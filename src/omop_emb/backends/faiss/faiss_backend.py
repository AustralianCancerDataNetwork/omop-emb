from __future__ import annotations

from pathlib import Path
from typing import (
    Mapping, 
    Optional, 
    Sequence, 
    Type, 
    Dict, 
    Tuple,
    Any
)

import numpy as np
from numpy import ndarray
from sqlalchemy import Engine, select, func
from sqlalchemy.orm import Session
import logging
import shutil

from .faiss_sql import (
    FAISSConceptIDEmbeddingRegistry,
    create_faiss_embedding_registry_table,
    delete_faiss_embedding_registry_table,
    add_concept_ids_to_faiss_registry,
    q_concept_ids_with_embeddings,
    q_concept_ids_with_embeddings_without_metadata,
)
from omop_emb.config import BackendType, IndexType, MetricType, ProviderType
from .index_config import (
    IndexConfig,
    index_config_from_index_type_and_metadata,
    INDEX_CONFIG_METADATA_KEY
)
from .storage_manager import EmbeddingStorageManager
from ..base import EmbeddingBackend, require_registered_model
from omop_emb.utils.embedding_utils import (
    EmbeddingConceptFilter,
    NearestConceptMatch,
    get_similarity_from_distance,
)
from omop_emb.model_registry import EmbeddingModelRecord

logger = logging.getLogger(__name__)


class FaissEmbeddingBackend(EmbeddingBackend[FAISSConceptIDEmbeddingRegistry]):
    """FAISS backend backed by HDF5 vector storage and SQL concept-id registry tables.

    Notes
    -----
    - The FAISS index files and HDF5 embedding files are stored in the filesystem,
      while the mapping between concept IDs and their positions in the FAISS index
      is stored in a SQLAlchemy registry table.
    - The HDF5 files store the raw embeddings and are used to populate the FAISS
      index. When new embeddings are upserted, they are appended to the HDF5 file
      and the FAISS index is updated accordingly. The registry table is also updated
      with any new concept IDs to maintain the mapping.

    Parameters
    ----------
    storage_base_dir : str | Path, optional
        Base directory for FAISS files and local metadata.
        Resolution is handled by ``EmbeddingBackend``: explicit argument, then
        ``OMOP_EMB_BASE_STORAGE_DIR``, then backend default.
    registry_db_name : str, optional
        Optional local model-registry database filename. If not provided, the
        registry will be created in the default location defined by
        ``EmbeddingBackend``, which resolves to ~/.omop_emb/ by default.
    """

    def __init__(
        self,
        storage_base_dir: Optional[str | Path] = None,
        registry_db_name: Optional[str] = None,
    ):
        super().__init__(
            storage_base_dir=storage_base_dir,
            registry_db_name=registry_db_name,
        )
        self._embedding_storage_managers: Dict[str, EmbeddingStorageManager] = {}

    @property
    def backend_type(self) -> BackendType:
        return BackendType.FAISS

    def _create_storage_table(self, engine: Engine, model_record: EmbeddingModelRecord) -> Type[FAISSConceptIDEmbeddingRegistry]:
        return create_faiss_embedding_registry_table(engine=engine, model_record=model_record)

    def get_safe_model_dir(self, model_name: str) -> Path:
        return self.storage_base_dir / self._embedding_model_registry.safe_model_name(model_name)

    def has_stale_model_artifacts(self, model_name: str) -> bool:
        model_dir = self.get_safe_model_dir(model_name)
        return model_dir.exists() and any(model_dir.iterdir())

    # ------------------------------------------------------------------
    # Storage manager lifecycle
    # ------------------------------------------------------------------

    def get_storage_manager(self, model_record: EmbeddingModelRecord) -> EmbeddingStorageManager:
        self.register_storage_manager(model_record)
        return self._embedding_storage_managers[model_record.model_name]

    def register_storage_manager(self, model_record: EmbeddingModelRecord) -> None:
        """Ensure an in-memory storage manager exists for *model_record* obtained from the ModelRegistry.
        If it doesn't exist yet, create a new one using the persisted index configuration from the registry metadata.
        """
        existing = self._embedding_storage_managers.get(model_record.model_name)

        if not existing:
            logger.info(
                f"Registering storage manager for '{model_record.model_name}' "
                f"(dimensions={model_record.dimensions})"
            )
            self._embedding_storage_managers[model_record.model_name] = EmbeddingStorageManager(
                file_dir=self.get_safe_model_dir(model_record.model_name),
                dimensions=model_record.dimensions,
                backend_type=self.backend_type,
            )

    # ------------------------------------------------------------------
    # Model registration / deletion / configuration update
    # ------------------------------------------------------------------

    def register_model(
        self,
        engine: Engine,
        model_name: str,
        dimensions: int,
        *,
        provider_type: ProviderType,
        index_config: IndexConfig,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> EmbeddingModelRecord:
        """Register a model with the FAISS backend.

        HNSW parameters are serialized into the registry metadata so they can be
        reconstructed faithfully after a process restart — without the caller
        having to supply them again.

        The DB write happens first; the model directory is only created on success.
        """
        metadata = metadata or {}
        persisted_metadata: dict[str, object] = {
            INDEX_CONFIG_METADATA_KEY: index_config.to_dict(),
            **(self._validate_external_metadata(metadata)),
        }
        record = super().register_model(
            engine=engine,
            model_name=model_name,
            provider_type=provider_type,
            dimensions=dimensions,
            index_type=index_config.index_type,
            metadata=persisted_metadata,
        )

        model_dir = self.get_safe_model_dir(model_name)
        model_dir.mkdir(parents=False, exist_ok=True)
        logger.info(f"Created model directory: {model_dir}")

        self.register_storage_manager(record)
        return record

    def delete_model(
        self,
        engine: Engine,
        model_name: str,
        *,
        provider_type: ProviderType,
        index_type: IndexType,
    ) -> None:
        """Fully remove a registered model and all its artifacts.

        Drops the SQL embedding-registry table, deletes the HDF5 and FAISS files
        from disk, removes all in-memory caches, and deletes the registry row.
        """
        record = self.get_registered_model(
            model_name=model_name,
            provider_type=provider_type,
            index_type=index_type,
        )
        if record is None:
            raise ValueError(
                f"Model '{model_name}' with provider='{provider_type.value}' and "
                f"index_type='{index_type.value}' is not registered."
            )

        # Drop OMOP CDM embedding table
        cache_key = self._get_embedding_table_cache_key(model_name, provider_type, index_type)
        embedding_table = self._embedding_table_cache.pop(cache_key, None)
        if embedding_table is not None:
            delete_faiss_embedding_registry_table(engine, record)


        # Evict in-memory manager and delete all associated files (HDF5 + FAISS indices)
        self._embedding_storage_managers.pop(model_name, None)
        model_dir = self.get_safe_model_dir(model_name)
        if model_dir.exists():
            shutil.rmtree(model_dir)

        # Remove registry row
        self._embedding_model_registry.delete_model(
            model_name=model_name,
            provider_type=provider_type,
            backend_type=self.backend_type,
            index_type=index_type,
        )
        logger.info(f"Deleted model '{model_name}' from registry.")

    def update_model_index_configuration(
        self,
        model_name: str,
        *,
        provider_type: ProviderType,
        index_type: IndexType,
        index_config: IndexConfig,
    ) -> EmbeddingModelRecord:
        """Persist updated index configuration parameters for an existing model.

        Only metadata (e.g. HNSW ``num_neighbors``, ``ef_search``) is changed.
        The in-memory storage manager is evicted so the next access re-creates 
        it with the new config. Callers should follow up with ``rebuild_model_indexes`` 
        to apply the new parameters to the on-disk FAISS index files.
        """
        if index_config.index_type != index_type:
            raise ValueError(
                f"index_config.index_type ({index_config.index_type!r}) must match "
                f"the registered index_type ({index_type!r}). "
                "Use delete_model + register_model to change the index type."
            )

        new_metadata = {INDEX_CONFIG_METADATA_KEY: index_config.to_dict()}
        new_record = self._embedding_model_registry.update_model_metadata(
            model_name=model_name,
            provider_type=provider_type,
            backend_type=self.backend_type,
            index_type=index_type,
            metadata=new_metadata,
        )

        # Evict — will be re-created (with new config) on next get_storage_manager call
        self._embedding_storage_managers.pop(model_name, None)
        return new_record

    # ------------------------------------------------------------------
    # Consistency / maintenance helpers
    # ------------------------------------------------------------------

    def _validate_storage_consistency(
        self,
        session: Session,
        model_record: EmbeddingModelRecord,
    ) -> None:
        """Raise if the SQL registry row count differs from the HDF5 vector count."""
        embedding_table = self.get_embedding_table(
            model_name=model_record.model_name,
            provider_type=model_record.provider_type,
            index_type=model_record.index_type,
        )
        db_count: int = session.scalar(
            select(func.count()).select_from(embedding_table)
        ) or 0
        hdf5_count = self.get_storage_manager(model_record).get_count()
        if db_count != hdf5_count:
            raise RuntimeError(
                f"Storage consistency error for model '{model_record.model_name}': "
                f"SQL registry has {db_count} entries but HDF5 storage has {hdf5_count}. "
                "Manual reconciliation is required."
            )

    @require_registered_model
    def initialise_indexes(
        self,
        *,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        metric_types: Sequence[MetricType],
        _model_record: EmbeddingModelRecord,
        batch_size: int = 100_000,
    ) -> None:
        """Load or build FAISS indexes for the given metric types from HDF5 storage.

        Call this on process start to warm up indexes before the first search query.
        If index files already exist on disk they are loaded; otherwise they are
        built from the HDF5 embeddings and saved.
        """
        config = index_config_from_index_type_and_metadata(
            _model_record.index_type, _model_record.metadata
        )
        storage_manager = self.get_storage_manager(_model_record)
        for metric_type in metric_types:
            storage_manager.create_index_for_metric(
                metric_type=metric_type,
                index_config=config,
                batch_size=batch_size,
            )

    def rebuild_model_indexes(
        self,
        model_record: EmbeddingModelRecord,
        metric_types: Sequence[MetricType],
        batch_size: int = 100_000,
    ) -> None:
        """Delete and rebuild the FAISS index files for the given metric types.

        Use this after calling ``update_model_index_configuration`` to apply the
        new parameters, or any time the HDF5 embeddings have changed in a way that
        makes the existing index stale.
        """
        config = index_config_from_index_type_and_metadata(
            model_record.index_type, model_record.metadata
        )
        storage_manager = self.get_storage_manager(model_record)
        for metric_type in metric_types:
            storage_manager.rebuild_index_for_metric(
                metric_type=metric_type,
                index_config=config,
                batch_size=batch_size,
            )

    @staticmethod
    def _validate_external_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Expected metadata to be a mapping type, got {type(metadata)}")
        
        reserved_keys = {INDEX_CONFIG_METADATA_KEY}
        if any(key in reserved_keys for key in metadata):
            raise ValueError(f"Metadata contains reserved keys: {reserved_keys & metadata.keys()}")
        return metadata

    # ------------------------------------------------------------------
    # Core read/write operations
    # ------------------------------------------------------------------

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
        """Insert or update vector embeddings for a collection of OMOP concept IDs.

        Parameters
        ----------
        session : Session
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
            Embedding matrix of shape ``(n_concepts, D)``.
        _model_record : EmbeddingModelRecord
            Injected by ``@require_registered_model``.
        metric_type : MetricType, optional
            If provided, the FAISS index for this metric is created/updated.
            Without a metric the raw embeddings are stored but no index is built;
            nearest-neighbor search will trigger index construction on first call.
        """
        concept_id_tuple = tuple(concept_ids)
        self.validate_embeddings_and_concept_ids(
            concept_ids=concept_id_tuple,
            embeddings=embeddings,
            dimensions=_model_record.dimensions,
        )

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
                f"Failed to add concept IDs to FAISS registry for model '{model_name}'. "
                "This may be due to duplicate concept IDs or a database constraint violation. "
                f"Original error: {e}"
            ) from e

        config = index_config_from_index_type_and_metadata(
            _model_record.index_type, _model_record.metadata
        )
        storage_manager = self.get_storage_manager(_model_record)
        storage_manager.append(
            concept_ids=np.array(concept_id_tuple, dtype=np.int64),
            embeddings=embeddings,
            index_config=config if metric_type is not None else None,
            metric_type=metric_type,
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
        config = index_config_from_index_type_and_metadata(
            _model_record.index_type, _model_record.metadata
        )
        storage_manager = self.get_storage_manager(_model_record)

        if storage_manager.get_count() == 0:
            return ()

        storage_manager.create_index_for_metric(
            metric_type=metric_type,
            index_config=config,
        )

        self.validate_embeddings(embeddings=query_embeddings, dimensions=_model_record.dimensions)

        embedding_table = self.get_embedding_table(
            model_name=model_name,
            index_type=index_type,
            provider_type=provider_type,
        )

        if concept_filter is None:
            permitted_concept_ids: Optional[np.ndarray] = None
            k = self.DEFAULT_K_NEAREST
        else:
            permitted_rows = session.execute(
                q_concept_ids_with_embeddings_without_metadata(embedding_table, concept_filter)
            ).all()
            permitted_concept_ids = np.array(
                [row.concept_id for row in permitted_rows], dtype=np.int64
            )
            k = concept_filter.limit if concept_filter.limit is not None else self.DEFAULT_K_NEAREST

        distances, returned_ids = storage_manager.search(
            query_vector=query_embeddings,
            metric_type=metric_type,
            index_type=index_type,
            k=k,
            subset_concept_ids=permitted_concept_ids,
        )

        # Fetch metadata for only the concept_ids FAISS actually returned
        unique_returned = np.unique(returned_ids[returned_ids != -1])
        concept_meta = {}
        if unique_returned.size > 0:
            meta_filter = EmbeddingConceptFilter(concept_ids=tuple(int(x) for x in unique_returned))
            meta_rows = session.execute(
                q_concept_ids_with_embeddings(embedding_table, meta_filter)
            ).all()
            concept_meta = {row.concept_id: row for row in meta_rows}

        matches = []
        for concept_id_per_query, distance_per_query in zip(returned_ids, distances):
            matches_per_query = []
            for concept_id, distance in zip(concept_id_per_query, distance_per_query):
                if concept_id == -1:
                    continue
                row = concept_meta.get(int(concept_id))
                if row is None:
                    logger.warning(
                        f"Concept ID {concept_id} returned by FAISS but not found in "
                        "concept metadata. Skipping."
                    )
                    continue
                similarity = get_similarity_from_distance(distance.item(), metric_type)
                if not isinstance(similarity, float):
                    raise RuntimeError(f"Expected float similarity, got {type(similarity)}")
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
        _model_record: EmbeddingModelRecord,
    ) -> Mapping[int, Sequence[float]]:
        concept_id_tuple = tuple(concept_ids)
        if not concept_id_tuple:
            return {}

        storage_manager = self.get_storage_manager(_model_record)
        return storage_manager.get_embeddings_by_concept_ids(
            concept_ids=np.array(concept_id_tuple, dtype=np.int64)
        )
