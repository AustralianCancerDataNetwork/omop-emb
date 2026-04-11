from __future__ import annotations

import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Optional, Sequence, Type, cast, Dict, Tuple 

import numpy as np
from numpy import ndarray
from omop_emb.backends.config import BackendType
from sqlalchemy import Engine, insert, select, func
from sqlalchemy.orm import Session
import logging
from dataclasses import dataclass, field
import h5py

from omop_alchemy.cdm.model.vocabulary import Concept
from omop_emb.backends.registry import ModelRegistry

from ..errors import EmbeddingBackendConfigurationError
from .faiss_sql import (
    FAISSConceptIDEmbeddingRegistry, 
    create_faiss_embedding_registry_table, 
    add_concept_ids_to_faiss_registry,
    q_concept_ids_with_embeddings
)
from ..config import (
    IndexType,
    MetricType,
    ENV_OMOP_EMB_FAISS_INDEX_DIR,
    get_supported_metrics_for_backend_index,
)
from .storage_manager import EmbeddingStorageManager
from ..base import EmbeddingBackend, require_registered_model
from ..embedding_utils import (
    EmbeddingBackendCapabilities,
    EmbeddingConceptFilter,
    EmbeddingModelRecord,
    NearestConceptMatch,
    get_similarity_from_distance
)

logger = logging.getLogger(__name__)

FAISS_METADATA_HNSW_NUM_NEIGHBORS = "hnsw_num_neighbors"
FAISS_METADATA_HNSW_EF_SEARCH = "hnsw_ef_search"
FAISS_METADATA_HNSW_EF_CONSTRUCTION = "hnsw_ef_construction"
DEFAULT_HNSW_NUM_NEIGHBORS = 32
DEFAULT_HNSW_EF_SEARCH = 64
DEFAULT_HNSW_EF_CONSTRUCTION = 200


def _coerce_positive_int(*, value: Optional[object], field_name: str, default: int) -> int:
    if value is None:
        return default
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer. Got {value!r}.") from exc
    if resolved <= 0:
        raise ValueError(f"{field_name} must be a positive integer. Got {resolved}.")
    return resolved


def build_faiss_index_metadata(
    *,
    index_type: IndexType,
    existing_metadata: Optional[Mapping[str, object]] = None,
    hnsw_num_neighbors: Optional[int] = None,
    hnsw_ef_search: Optional[int] = None,
    hnsw_ef_construction: Optional[int] = None,
) -> dict[str, object]:
    metadata = dict(existing_metadata or {})
    for key in (
        FAISS_METADATA_HNSW_NUM_NEIGHBORS,
        FAISS_METADATA_HNSW_EF_SEARCH,
        FAISS_METADATA_HNSW_EF_CONSTRUCTION,
    ):
        metadata.pop(key, None)

    if index_type == IndexType.HNSW:
        metadata[FAISS_METADATA_HNSW_NUM_NEIGHBORS] = _coerce_positive_int(
            value=hnsw_num_neighbors,
            field_name=FAISS_METADATA_HNSW_NUM_NEIGHBORS,
            default=DEFAULT_HNSW_NUM_NEIGHBORS,
        )
        metadata[FAISS_METADATA_HNSW_EF_SEARCH] = _coerce_positive_int(
            value=hnsw_ef_search,
            field_name=FAISS_METADATA_HNSW_EF_SEARCH,
            default=DEFAULT_HNSW_EF_SEARCH,
        )
        metadata[FAISS_METADATA_HNSW_EF_CONSTRUCTION] = _coerce_positive_int(
            value=hnsw_ef_construction,
            field_name=FAISS_METADATA_HNSW_EF_CONSTRUCTION,
            default=DEFAULT_HNSW_EF_CONSTRUCTION,
        )

    return metadata


def resolve_faiss_hnsw_parameters(
    metadata: Optional[Mapping[str, object]],
) -> tuple[int, int, int]:
    metadata = metadata or {}
    return (
        _coerce_positive_int(
            value=metadata.get(FAISS_METADATA_HNSW_NUM_NEIGHBORS),
            field_name=FAISS_METADATA_HNSW_NUM_NEIGHBORS,
            default=DEFAULT_HNSW_NUM_NEIGHBORS,
        ),
        _coerce_positive_int(
            value=metadata.get(FAISS_METADATA_HNSW_EF_SEARCH),
            field_name=FAISS_METADATA_HNSW_EF_SEARCH,
            default=DEFAULT_HNSW_EF_SEARCH,
        ),
        _coerce_positive_int(
            value=metadata.get(FAISS_METADATA_HNSW_EF_CONSTRUCTION),
            field_name=FAISS_METADATA_HNSW_EF_CONSTRUCTION,
            default=DEFAULT_HNSW_EF_CONSTRUCTION,
        ),
    )

@dataclass
class FaissMetadata:
    """Strongly typed metadata required for the FAISS backend to operate."""
    index_file_path: str
    metadata_bucket: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes to a flat dictionary for SQLAlchemy JSON storage."""
        # We flatten it so it looks clean in the database
        result = self.metadata_bucket.copy()
        result["index_file_path"] = self.index_file_path
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FaissMetadata":
        """Safely reconstructs the object from the database JSON."""
        data_copy = dict(data)
        
        # Pop the required FAISS fields, raising a clear error if missing
        try:
            index_path = data_copy.pop("index_file_path")
        except KeyError as e:
            raise ValueError(
                f"Corrupted FAISS metadata in registry. Missing required key: {e}"
            )

        # Everything else goes back into user_metadata
        return cls(
            index_file_path=str(index_path),
            metadata_bucket=data_copy
        )


class FaissEmbeddingBackend(EmbeddingBackend[FAISSConceptIDEmbeddingRegistry]):
    """
    File-based FAISS embedding backend.

    First-pass design
    -----------------
    - Registry is stored as JSON under a configurable base directory.
    - Embeddings are stored as an HDF5 matrix on disk alongside a FAISS index.
    - OMOP concept metadata and filter application still use SQLAlchemy.
    - Nearest-neighbor search uses FAISS when possible and falls back to
      in-memory cosine ranking for filtered subsets.

    Notes
    -----
    This backend is intentionally conservative and optimized for clarity over
    extreme scale. In particular, updates currently rewrite the per-model numpy
    arrays and rebuild the FAISS index, which is acceptable for a first pass
    but not ideal for very large incremental workloads.
    """

    DEFAULT_FAISS_DIR = ".omop_emb/faiss"

    def __init__(self, base_dir: Optional[str | os.PathLike[str]] = None):
        super().__init__()
        self.base_dir = Path(
            base_dir
            or os.getenv(ENV_OMOP_EMB_FAISS_INDEX_DIR)
            or FaissEmbeddingBackend.DEFAULT_FAISS_DIR
        ).expanduser()
        self.embedding_storage_managers: Dict[str, EmbeddingStorageManager] = {}

    @property
    def backend_type(self) -> BackendType:
        return BackendType.FAISS

    @property
    def capabilities(self) -> EmbeddingBackendCapabilities:
        return EmbeddingBackendCapabilities(
            stores_embeddings=True,
            supports_incremental_upsert=True,
            supports_nearest_neighbor_search=True,
            supports_server_side_similarity=False,
            supports_filter_by_concept_ids=True,
            supports_filter_by_domain=True,
            supports_filter_by_vocabulary=True,
            supports_filter_by_standard_flag=True,
            requires_explicit_index_refresh=False,
        )

    def initialise_store(self, engine) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        return super().initialise_store(engine)

    def _create_storage_table(self, engine: Engine, entry: ModelRegistry) -> Type[FAISSConceptIDEmbeddingRegistry]:
        return create_faiss_embedding_registry_table(engine=engine, model_registry_entry=entry)
    
    def get_safe_model_dir(self, model_name: str) -> Path:
        return self.base_dir / self.safe_model_name(model_name)

    def has_stale_model_artifacts(self, model_name: str) -> bool:
        model_dir = self.get_safe_model_dir(model_name)
        if not model_dir.exists():
            return False
        return any(model_dir.iterdir())
    
    def get_storage_manager(
        self, 
        model_name: str,
        dimensions: int,
        index_type: IndexType,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> EmbeddingStorageManager:
        self.register_storage_manager(
            model_name=model_name,
            dimensions=dimensions,
            index_type=index_type,
            metadata=metadata,
        )
        return self.embedding_storage_managers[model_name]
    
    def register_storage_manager(
        self,
        model_name: str,
        dimensions: int,
        index_type: IndexType,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Registers a storage manager for the given model if not already registered. This ensures that the necessary on-disk structures are in place for the model's embeddings and index."""

        hnsw_num_neighbors, hnsw_ef_search, hnsw_ef_construction = resolve_faiss_hnsw_parameters(metadata)
        existing_manager = self.embedding_storage_managers.get(model_name)
        if (
            existing_manager is None
            or existing_manager.dimensions != dimensions
            or existing_manager.hnsw_num_neighbors != hnsw_num_neighbors
            or existing_manager.hnsw_ef_search != hnsw_ef_search
            or existing_manager.hnsw_ef_construction != hnsw_ef_construction
        ):
            logger.info(f"Registering new storage manager for model '{model_name}' with dimensions={dimensions}, index_type={index_type}")
            self.embedding_storage_managers[model_name] = EmbeddingStorageManager(
                file_dir=self.get_safe_model_dir(model_name), 
                dimensions=dimensions,
                backend_type=self.backend_type,
                hnsw_num_neighbors=hnsw_num_neighbors,
                hnsw_ef_search=hnsw_ef_search,
                hnsw_ef_construction=hnsw_ef_construction,
            )

    def update_model_index_configuration(
        self,
        *,
        session: Session,
        model_name: str,
        index_type: IndexType,
        metadata: Mapping[str, object],
    ) -> EmbeddingModelRecord:
        row = session.scalar(
            select(ModelRegistry).where(
                ModelRegistry.model_name == model_name,
                ModelRegistry.backend_type == self.backend_type,
            )
        )
        if row is None:
            raise ValueError(
                f"Embedding model '{model_name}' is not registered in the FAISS backend."
            )

        previous_index_type = row.index_type
        row.index_type = index_type
        row.details = dict(metadata)
        session.add(row)
        session.commit()
        session.refresh(row)
        self.embedding_storage_managers.pop(model_name, None)

        if previous_index_type != index_type:
            obsolete_index_dir = self.get_safe_model_dir(model_name) / f"index_{previous_index_type.value}"
            if obsolete_index_dir.exists():
                logger.info(
                    "Deleting obsolete FAISS index directory for model '%s': %s",
                    model_name,
                    obsolete_index_dir,
                )
                shutil.rmtree(obsolete_index_dir)

        return self._record_from_registry_row(row)

    def _validate_storage_consistency(
        self,
        *,
        session: Session,
        model_name: str,
        storage_manager: EmbeddingStorageManager,
    ) -> None:
        embedding_table = self._get_embedding_table(session, model_name)
        registry_count = session.scalar(select(func.count()).select_from(embedding_table))
        storage_count = storage_manager.get_count()
        if registry_count != storage_count:
            raise RuntimeError(
                f"FAISS storage is inconsistent for model '{model_name}': SQL registry has "
                f"{registry_count} concept_ids but HDF5 storage has {storage_count} vectors. "
                "Re-run with `--overwrite-model-registration` or rebuild after cleanup."
            )

    def register_model(
        self,
        engine: Engine,
        model_name: str,
        dimensions: int,
        *,
        index_type: IndexType,
        metadata: Mapping[str, object] = {},
    ) -> EmbeddingModelRecord:

        # Create the storage for the model on disk
        safe_name = self.safe_model_name(model_name)
        model_dir = self.base_dir / safe_name
        model_dir.mkdir(parents=False, exist_ok=True)
        logger.info(f"Created model directory: {model_dir}")

        # NOTE: not sure if still necessary as the manager takes care of it all.
        #faiss_metadata = FaissMetadata(
        #    index_file_path=str(self._index_path(model_name)),
        #    metadata_bucket=dict(metadata),
        #)
        return super().register_model(
            engine=engine,
            model_name=model_name,
            dimensions=dimensions,
            index_type=index_type,
            #metadata=faiss_metadata.to_dict(),
            metadata=metadata
        )

    def delete_model(
        self,
        *,
        engine: Engine,
        session: Session,
        model_name: str,
    ) -> bool:
        deleted = super().delete_model(engine=engine, session=session, model_name=model_name)
        self.embedding_storage_managers.pop(model_name, None)
        model_dir = self.get_safe_model_dir(model_name)
        if model_dir.exists():
            logger.info("Deleting FAISS model directory for '%s': %s", model_name, model_dir)
            shutil.rmtree(model_dir)
        return deleted

    @require_registered_model
    def upsert_embeddings(
        self,
        session: Session,
        model_name: str,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
        embeddings: ndarray,
        metric_type: Optional[MetricType] = None,
    ) -> None:
        """
        Insert or update vector embeddings for a collection of OMOP concept IDs.

        This method presents an interface for persisting generated embeddings. 
        The FAISS backend implementation stages the provided embeddings in an HDF5 file and updates the corresponding FAISS index on disk. 
        Additionally, it ensures that the mapping between concept IDs and their positions in the FAISS index is maintained in a SQLAlchemy-managed registry table.

        Parameters
        ----------
        session : sqlalchemy.orm.Session
            The active database session used for transactional persistence and 
            model metadata updates.
        model_name : str
            The unique identifier or name of the embedding model (e.g., 
            'text-embedding-3-small').
        model_record : EmbeddingModelRecord
            A record object containing metadata, dimensions, and configuration 
            specific to the embedding model being processed.
        concept_ids : Sequence[int]
            A sequence of OMOP standard concept IDs corresponding to the 
            ordered rows in the embeddings array.
        embeddings : numpy.ndarray
            A 2D array of shape (n_concepts, n_dimensions) containing the 
            generated vector representations.
        metric_type : Optional[MetricType]
            The similarity metric type (e.g., COSINE, EUCLIDEAN) that should be 
            associated with the stored embeddings for accurate nearest neighbor 
            search behavior. If None, the index will not be built and the raw
            embeddings are only stored in the HDF5 file for later use.

        Returns
        -------
        None

        Notes
        -----
        - Indices in FAISS are directly tied to a metric for optimisation. Providing
        no metric_type means we won't be able to create the index and will only store the raw embeddings.
        The index will be created upon :meth:`~FaissEmbeddingBackend.get_nearest_concepts`, where a `metric_type` is required.
        """

        concept_id_tuple = tuple(concept_ids)
        storage_manager = self.get_storage_manager(
            model_name=model_name,
            dimensions=model_record.dimensions,
            index_type=model_record.index_type,
            metadata=model_record.metadata,
        )
        self._validate_storage_consistency(
            session=session,
            model_name=model_name,
            storage_manager=storage_manager,
        )
        self.validate_embeddings_and_concept_ids(
            concept_ids=concept_id_tuple,
            embeddings=embeddings,
            dimensions=model_record.dimensions,
        )

        # Try to add first before we damage the on-disk stuff
        try:
            add_concept_ids_to_faiss_registry(
                concept_ids=concept_id_tuple,
                session=session,
                registered_table=self._get_embedding_table(session, model_name),
            )
        except Exception as e:
            session.rollback()
            raise ValueError(
                f"Failed to add concept IDs to FAISS registry for model '{model_name}'. This may be due to a mismatch between the provided concept_ids and the existing entries in the database, or a database constraint violation. Original error: {str(e)}"
            ) from e
        
        storage_manager.append(
            concept_ids=np.array(concept_id_tuple, dtype=np.int64),
            embeddings=embeddings,
            index_type=model_record.index_type,
            metric_type=metric_type
        )

    @require_registered_model
    def get_nearest_concepts(
        self,
        session: Session,
        model_name: str,
        model_record: EmbeddingModelRecord,
        query_embeddings: np.ndarray,
        metric_type: MetricType,
        *,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        k: int = 10,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        
        storage_manager = self.get_storage_manager(
            model_name=model_name,
            dimensions=model_record.dimensions,
            index_type=model_record.index_type,
            metadata=model_record.metadata,
        )

        if storage_manager.get_count() == 0:
            return ()

        self._validate_storage_consistency(
            session=session,
            model_name=model_name,
            storage_manager=storage_manager,
        )
        self.validate_embeddings(embeddings=query_embeddings, dimensions=model_record.dimensions)
        embedding_table = self._get_embedding_table(session, model_name)

        if concept_filter is None or self._filter_is_empty(concept_filter):
            # Easier to not do any filter if all are allowed
            permitted_concept_ids = None
        else:
            filter_started_at = time.monotonic()
            q_permitted_concept_ids = (
                select(embedding_table.concept_id)
                .join(Concept, Concept.concept_id == embedding_table.concept_id)
            )
            q_permitted_concept_ids = concept_filter.apply(q_permitted_concept_ids)
            permitted_concept_ids = np.array(
                [int(row.concept_id) for row in session.execute(q_permitted_concept_ids)],
                dtype=np.int64,
            )
            logger.info(
                "Resolved FAISS filter candidate set: model=%s count=%s elapsed=%.3fs",
                model_name,
                len(permitted_concept_ids),
                time.monotonic() - filter_started_at,
            )

        faiss_search_started_at = time.monotonic()
        distances, concept_ids = storage_manager.search(
            query_vector=query_embeddings,
            metric_type=metric_type,
            index_type=model_record.index_type,
            k=k,
            subset_concept_ids=permitted_concept_ids
        )
        logger.info(
            "Completed backend nearest-neighbor stage: model=%s queries=%s elapsed=%.3fs",
            model_name,
            query_embeddings.shape[0],
            time.monotonic() - faiss_search_started_at,
        )

        returned_ids = tuple(
            int(concept_id)
            for concept_id in np.unique(concept_ids)
            if int(concept_id) != -1
        )
        if not returned_ids:
            return tuple(() for _ in range(query_embeddings.shape[0]))

        hydration_started_at = time.monotonic()
        permitted_concept_ids_storage = {
            row.concept_id: row
            for row in session.execute(
                q_concept_ids_with_embeddings(
                    embedding_table=embedding_table,
                    concept_filter=EmbeddingConceptFilter(concept_ids=returned_ids),
                    limit=None,
                )
            )
        }
        logger.info(
            "Hydrated FAISS result metadata: model=%s concepts=%s elapsed=%.3fs",
            model_name,
            len(permitted_concept_ids_storage),
            time.monotonic() - hydration_started_at,
        )

        matches = []
        for concept_id_pery_query, distance_per_query in zip(concept_ids, distances):
            matches_per_query = []
            for concept_id, distance in zip(concept_id_pery_query, distance_per_query):
                if concept_id == -1:  # Skip empty for k>total_concepts 
                    continue
                row = permitted_concept_ids_storage.get(concept_id)
                if row is None:
                    logger.warning(f"Concept ID {concept_id} returned by FAISS search but not found in permitted concept IDs. This indicates a mismatch between the FAISS index and the database registry. Skipping this result.")
                    continue
                matches_per_query.append(NearestConceptMatch(
                    concept_id=int(concept_id),
                    concept_name=row.concept_name,
                    similarity=get_similarity_from_distance(distance, metric_type),
                    is_standard=bool(row.is_standard),
                    is_active=bool(row.is_active),
                ))
            matches.append(tuple(matches_per_query))
        return tuple(matches)

    @require_registered_model
    def refresh_model_index(self, session: Session, model_name: str, model_record: EmbeddingModelRecord) -> None:
        self.rebuild_model_indexes(
            session=session,
            model_name=model_name,
            model_record=model_record,
            metric_types=None,
        )

    @require_registered_model
    def rebuild_model_indexes(
        self,
        session: Session,
        model_name: str,
        model_record: EmbeddingModelRecord,
        metric_types: Optional[Sequence[MetricType]] = None,
        batch_size: int = 100_000,
    ) -> None:
        storage_manager = self.get_storage_manager(
            model_name=model_name,
            dimensions=model_record.dimensions,
            index_type=model_record.index_type,
            metadata=model_record.metadata,
        )
        self._validate_storage_consistency(
            session=session,
            model_name=model_name,
            storage_manager=storage_manager,
        )
        metrics = tuple(metric_types) if metric_types is not None else get_supported_metrics_for_backend_index(
            self.backend_type,
            model_record.index_type,
        )
        for metric_type in metrics:
            logger.info(
                "Rebuilding FAISS index for model '%s' with index_type=%s metric=%s",
                model_name,
                model_record.index_type,
                metric_type,
            )
            storage_manager.rebuild_index(
                index_type=model_record.index_type,
                metric_type=metric_type,
                batch_size=batch_size,
            )
    
    @require_registered_model
    def get_embeddings_by_concept_ids(
            self, 
            session: Session, 
            model_name: str, 
            model_record: EmbeddingModelRecord,
            concept_ids: Sequence[int]
        ) -> Mapping[int, Sequence[float]]:
        concept_id_tuple = tuple(concept_ids)
        if not concept_id_tuple:
            return {}
        
        concept_ids_np = np.array(concept_id_tuple, dtype=np.int64)

        storage_manager = self.get_storage_manager(
            model_name=model_name,
            dimensions=model_record.dimensions,
            index_type=model_record.index_type,
            metadata=model_record.metadata,
        )

        return storage_manager.get_embeddings_by_concept_ids(concept_ids=concept_ids_np)


    def _get_filtered_concept_ids(
        self,
        session: Session,
        concept_filter: EmbeddingConceptFilter,
    ) -> tuple[int, ...]:
        query = select(Concept.concept_id)
        if concept_filter is not None:
            query = concept_filter.apply(query)
        return tuple(int(row.concept_id) for row in session.execute(query))

    @staticmethod
    def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    @staticmethod
    def _filter_is_empty(concept_filter: EmbeddingConceptFilter) -> bool:
        return (
            concept_filter.concept_ids is None
            and concept_filter.domains is None
            and concept_filter.vocabularies is None
            and not concept_filter.require_standard
        )

    @staticmethod
    def _create_faiss_index(*, faiss, index_type: str, dimensions: int):
        index_type = str(getattr(index_type, "value", index_type))
        if index_type in {"IndexFlatIP", "flat", "flatip"}:
            return faiss.IndexFlatIP(dimensions)
        raise ValueError(
            f"Unsupported FAISS index_type={index_type!r} in first-pass backend. "
            "Currently supported: IndexFlatIP."
        )
