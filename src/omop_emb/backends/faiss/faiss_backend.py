from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Type, Dict, Tuple 

import numpy as np
from numpy import ndarray
from sqlalchemy import Engine
from sqlalchemy.orm import Session
import logging

from .faiss_sql import (
    FAISSConceptIDEmbeddingRegistry, 
    create_faiss_embedding_registry_table, 
    add_concept_ids_to_faiss_registry,
    q_concept_ids_with_embeddings
)
from omop_emb.config import BackendType, IndexType, MetricType, ProviderType
from .storage_manager import EmbeddingStorageManager
from ..base import EmbeddingBackend, require_registered_model
from omop_emb.utils.embedding_utils import (
    EmbeddingConceptFilter,
    NearestConceptMatch,
    get_similarity_from_distance
)
from omop_emb.model_registry import EmbeddingModelRecord

logger = logging.getLogger(__name__)


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
    
    def __init__(
        self,
        storage_base_dir: Optional[str | Path] = None,
        registry_db_name: Optional[str] = None,
    ):
        """Initialize FAISS backend storage and registry wiring.

        Parameters
        ----------
        storage_base_dir : str | Path, optional
            Base directory for FAISS files and local metadata.
            Resolution is handled by ``EmbeddingBackend``:
            explicit argument, then ``OMOP_EMB_BASE_STORAGE_DIR``, then backend default.
        registry_db_name : str, optional
            Optional local model-registry database filename.
        """
        super().__init__(
            storage_base_dir=storage_base_dir,
            registry_db_name=registry_db_name,
        )
        self.embedding_storage_managers: Dict[str, EmbeddingStorageManager] = {}

    @property
    def backend_type(self) -> BackendType:
        return BackendType.FAISS

    def _create_storage_table(self, engine: Engine, model_record: EmbeddingModelRecord) -> Type[FAISSConceptIDEmbeddingRegistry]:
        return create_faiss_embedding_registry_table(engine=engine, model_record=model_record)
    
    def get_safe_model_dir(self, model_name: str) -> Path:
        return self.storage_base_dir / self._embedding_model_registry.safe_model_name(model_name)
    
    def get_storage_manager(
        self, 
        model_record: EmbeddingModelRecord,
    ) -> EmbeddingStorageManager:
        self.register_storage_manager(
            model_record=model_record,
        )
        return self.embedding_storage_managers[model_record.model_name]
    
    def register_storage_manager(
        self,
        model_record: EmbeddingModelRecord,
    ) -> None:
        """Registers a storage manager for the given model if not already registered. This ensures that the necessary on-disk structures are in place for the model's embeddings and index."""

        if model_record.model_name not in self.embedding_storage_managers:
            logger.info(f"Registering new storage manager for model '{model_record.model_name}' with dimensions={model_record.dimensions}, index_type={model_record.index_type}")
            self.embedding_storage_managers[model_record.model_name] = EmbeddingStorageManager(
                    file_dir=self.get_safe_model_dir(model_record.model_name), 
                    dimensions=model_record.dimensions,
                    backend_type=self.backend_type,
                )

    def register_model(
        self,
        engine: Engine,
        model_name: str,
        dimensions: int,
        *,
        provider_type: ProviderType,
        index_type: IndexType,
        metadata: Mapping[str, object] = {},
    ) -> EmbeddingModelRecord:
        """Register a model with FAISS backend.

        DB write happens first — if it raises (e.g., conflict), no directory is created.
        Directory creation is a side effect that happens only after successful registration.
        """
        # DB write first — if this raises, directory was never created
        record = super().register_model(
            engine=engine,
            model_name=model_name,
            provider_type=provider_type,
            dimensions=dimensions,
            index_type=index_type,
            metadata=metadata,
        )

        # Only create directory after successful registration
        safe_name = self._embedding_model_registry.safe_model_name(model_name)
        model_dir = self.storage_base_dir / safe_name
        model_dir.mkdir(parents=False, exist_ok=True)
        logger.info(f"Created model directory: {model_dir}")

        return record

    @require_registered_model
    def upsert_embeddings(
        self,
        *,
        session: Session,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        concept_ids: Sequence[int],
        embeddings: ndarray,
        _model_record: EmbeddingModelRecord,
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
            SQLAlchemy session bound to the OMOP CDM database.
        model_name : str
            Registered name of the embedding model.
        provider_type : ProviderType
            Provider type for the embedding model.
        index_type : IndexType
            Storage index type used for this model's embeddings.
        concept_ids : Sequence[int]
            Concept IDs aligned with the rows of ``embeddings``.
        embeddings : numpy.ndarray
            Embedding matrix of shape ``(n_concepts, D)``, where $D$ is the embedding dimensionality defined in the model registration.
        _model_record : EmbeddingModelRecord
            Internal registered-model record injected by ``@require_registered_model``.
        metric_type : Optional[MetricType]
            Optional metric type for the FAISS index. If provided, the index will be created or updated with the specified metric. If not provided, embeddings will be stored without an index, and nearest neighbor search will not be available until an index is created with a specified metric.

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
        self.validate_embeddings_and_concept_ids(
            concept_ids=concept_id_tuple,
            embeddings=embeddings,
            dimensions=_model_record.dimensions,
        )

        # Try to add first before we damage the on-disk stuff
        try:
            add_concept_ids_to_faiss_registry(
                concept_ids=concept_id_tuple,
                session=session,
                registered_table=self.get_embedding_table(
                    model_name=model_name,
                    index_type=index_type,
                    provider_type=provider_type,
                ),
            )
        except Exception as e:
            session.rollback()
            raise ValueError(
                f"Failed to add concept IDs to FAISS registry for model '{model_name}'. This may be due to a mismatch between the provided concept_ids and the existing entries in the database, or a database constraint violation. Original error: {str(e)}"
            ) from e
        
        storage_manager = self.get_storage_manager(_model_record)
        storage_manager.append(
            concept_ids=np.array(concept_id_tuple, dtype=np.int64),
            embeddings=embeddings,
            index_type=_model_record.index_type,
            metric_type=metric_type
        )

    @require_registered_model
    def get_nearest_concepts(
        self,
        *,
        session: Session,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        query_embeddings: np.ndarray,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        _model_record: EmbeddingModelRecord,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        
        storage_manager = self.get_storage_manager(_model_record)

        if storage_manager.get_count() == 0:
            return ()

        self.validate_embeddings(embeddings=query_embeddings, dimensions=_model_record.dimensions)
        q_permitted_concept_ids = q_concept_ids_with_embeddings(
            embedding_table=self.get_embedding_table(model_name=model_name, index_type=index_type, provider_type=provider_type),
            concept_filter=concept_filter,
            limit=None
        )

        permitted_concept_ids_storage = {
            row.concept_id: row for row in session.execute(q_permitted_concept_ids)
        }

        if concept_filter is None:
            # Easier to not do any filter if all are allowed
            permitted_concept_ids = None
            logger.debug(f"No concept filter provided. Setting number of returned nearest concepts (k) to default: {self.DEFAULT_K_NEAREST}")
            k = self.DEFAULT_K_NEAREST
        else:
            permitted_concept_ids = np.array(list(permitted_concept_ids_storage.keys()), dtype=np.int64)
            if concept_filter.limit is not None:
                k = concept_filter.limit
            else:
                logger.debug(f"Concept filter provided without limit. Setting number of returned nearest concepts (k) to default: {self.DEFAULT_K_NEAREST}")
                k = self.DEFAULT_K_NEAREST

        distances, concept_ids = storage_manager.search(
            query_vector=query_embeddings,
            metric_type=metric_type,
            index_type=_model_record.index_type,
            k=k,
            subset_concept_ids=permitted_concept_ids
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

                similarity = get_similarity_from_distance(distance.item(), metric_type)
                if not isinstance(similarity, float):
                    raise RuntimeError(f"Expected similarity to be a float, got {type(similarity)}")
                matches_per_query.append(NearestConceptMatch(
                    concept_id=int(concept_id),
                    concept_name=row.concept_name,
                    similarity=similarity,
                    is_standard=bool(row.is_standard),
                    is_active=bool(row.is_active),
                ))
            matches.append(tuple(matches_per_query))

        matches_tuple = tuple(matches)
        self.validate_nearest_concepts_output(matches_tuple, k, query_embeddings=query_embeddings)
        return matches_tuple

    @require_registered_model
    def get_embeddings_by_concept_ids(
        self, 
        *,
        session: Session,
        model_name: str, 
        provider_type: ProviderType,
        index_type: IndexType,
        concept_ids: Sequence[int],
        _model_record: EmbeddingModelRecord
        ) -> Mapping[int, Sequence[float]]:
        concept_id_tuple = tuple(concept_ids)
        if not concept_id_tuple:
            return {}
        
        concept_ids_np = np.array(concept_id_tuple, dtype=np.int64)

        storage_manager = self.get_storage_manager(_model_record)

        return storage_manager.get_embeddings_by_concept_ids(concept_ids=concept_ids_np)
