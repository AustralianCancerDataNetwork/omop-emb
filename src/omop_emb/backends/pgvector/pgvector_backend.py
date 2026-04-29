from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Type, Tuple

from numpy import ndarray
import logging
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from .pgvector_sql import (
    q_embedding_nearest_concepts,
    q_embedding_vectors_by_concept_ids,
    PGVectorConceptIDEmbeddingTable,
    EMBEDDING_COLUMN_NAME,
    create_pg_embedding_table,
    delete_pg_embedding_table,
    add_embeddings_to_registered_table,
)
from .pgvector_index_manager import (
    PGVectorBaseIndexManager,
    PGVectorFlatIndexManager,
    PGVectorHNSWIndexManager,
)
from omop_emb.config import BackendType, MetricType, IndexType, ProviderType
from omop_emb.backends.base_backend import EmbeddingBackend, require_registered_model
from omop_emb.backends.index_config import (
    IndexConfig,
    FlatIndexConfig,
    HNSWIndexConfig,
    index_config_from_index_type_and_metadata,
)
from omop_emb.utils.embedding_utils import (
    EmbeddingConceptFilter,
    NearestConceptMatch,
)
from omop_emb.model_registry import EmbeddingModelRecord

logger = logging.getLogger(__name__)

_PGVECTOR_MAX_VECTOR_DIMENSIONS = 2_000  # pgvector 'vector' type hard limit

_INDEX_MANAGER_FOR_CONFIG: dict[type[IndexConfig], type[PGVectorBaseIndexManager]] = {
    FlatIndexConfig: PGVectorFlatIndexManager,
    HNSWIndexConfig: PGVectorHNSWIndexManager,
}


class PGVectorEmbeddingBackend(EmbeddingBackend[PGVectorConceptIDEmbeddingTable]):
    """pgvector-backed embedding backend for PostgreSQL databases.

    Stores and retrieves embedding vectors using the ``pgvector`` Postgres
    extension.  All nearest-neighbor search is delegated to the database via
    SQL.  SQL index lifecycle (HNSW ``CREATE / DROP / REBUILD``) is managed by
    per-model ``PGVectorBaseIndexManager`` instances.

    Notes
    -----
    ``storage_base_dir`` is used for the local model-registry metadata database
    even though the vectors themselves live in Postgres.
    """

    def __init__(
        self,
        storage_base_dir=None,
        registry_db_name=None,
    ):
        super().__init__(
            storage_base_dir=storage_base_dir,
            registry_db_name=registry_db_name,
        )
        self._pgvector_index_managers: Dict[str, PGVectorBaseIndexManager] = {}

    # ------------------------------------------------------------------
    # Backend identity
    # ------------------------------------------------------------------

    @property
    def backend_type(self) -> BackendType:
        return BackendType.PGVECTOR

    def _create_storage_table(
        self, engine: Engine, model_record: EmbeddingModelRecord
    ) -> Type[PGVectorConceptIDEmbeddingTable]:
        table = create_pg_embedding_table(engine=engine, model_record=model_record)
        self._register_index_manager(engine=engine, model_record=model_record)
        return table

    def _delete_storage_table(self, engine: Engine, model_record: EmbeddingModelRecord) -> None:
        self._pgvector_index_managers.pop(model_record.model_name, None)
        delete_pg_embedding_table(engine=engine, model_record=model_record)

    # ------------------------------------------------------------------
    # Model registration
    # ------------------------------------------------------------------

    def register_model(
        self,
        *,
        engine: Engine,
        model_name: str,
        dimensions: int,
        provider_type: ProviderType,
        index_config: IndexConfig,
        metadata=None,
    ) -> "EmbeddingModelRecord":
        if dimensions > _PGVECTOR_MAX_VECTOR_DIMENSIONS:
            raise ValueError(
                f"pgvector 'vector' type supports at most {_PGVECTOR_MAX_VECTOR_DIMENSIONS:,} dimensions, "
                f"but model '{model_name}' requests {dimensions}. "
                "'halfvec' or 'bit' are not yet supported."
            )
        return super().register_model(
            engine=engine,
            model_name=model_name,
            dimensions=dimensions,
            provider_type=provider_type,
            index_config=index_config,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Store lifecycle
    # ------------------------------------------------------------------

    def pre_initialise_store(self, engine: Engine) -> None:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector CASCADE;"))

    # ------------------------------------------------------------------
    # Index manager lifecycle
    # ------------------------------------------------------------------

    def _register_index_manager(
        self, engine: Engine, model_record: EmbeddingModelRecord
    ) -> None:
        """Create and cache a pgvector index manager for *model_record* if not already present."""
        if model_record.model_name in self._pgvector_index_managers:
            return

        index_config = index_config_from_index_type_and_metadata(
            model_record.index_type, model_record.metadata
        )
        manager_cls = _INDEX_MANAGER_FOR_CONFIG.get(type(index_config))
        if manager_cls is None:
            raise ValueError(
                f"No pgvector index manager for index config type {type(index_config).__name__}."
            )
        manager = manager_cls(
            engine=engine,
            tablename=model_record.storage_identifier,
            embedding_column=EMBEDDING_COLUMN_NAME,
            index_config=index_config,
        )
        self._pgvector_index_managers[model_record.model_name] = manager
        logger.info(
            f"Registered pgvector index manager for '{model_record.model_name}' "
            f"({index_config.index_type.value})."
        )

    def get_index_manager(self, model_name: str) -> PGVectorBaseIndexManager:
        """Return the cached index manager for *model_name*, or raise if not registered."""
        manager = self._pgvector_index_managers.get(model_name)
        if manager is None:
            raise ValueError(
                f"No pgvector index manager for model '{model_name}'. "
                "Ensure the store is initialised and the model is registered."
            )
        return manager

    def update_model_index_configuration(
        self,
        model_name: str,
        *,
        provider_type: ProviderType,
        index_type: IndexType,
        index_config: IndexConfig,
    ) -> EmbeddingModelRecord:
        """Persist updated index configuration and refresh the in-memory index manager.

        The base implementation updates the registry metadata.  This override
        additionally evicts the stale in-memory manager so the next call to
        ``rebuild_indexes`` creates a fresh manager from the new config.
        """
        new_record = super().update_model_index_configuration(
            model_name=model_name,
            provider_type=provider_type,
            index_type=index_type,
            index_config=index_config,
        )
        self._pgvector_index_managers.pop(model_name, None)
        return new_record

    @require_registered_model
    def initialise_indexes(
        self,
        *,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        metric_types: Sequence[MetricType],
        _model_record: EmbeddingModelRecord,
    ) -> None:
        """Ensure SQL indices exist for each metric in *metric_types*.

        Idempotent: already-existing indices are skipped.  Call this on process
        startup to create any missing indices after model registration.
        """
        manager = self.get_index_manager(model_name)
        for metric_type in metric_types:
            manager.load_or_create(metric_type)

    @require_registered_model
    def rebuild_model_indexes(
        self,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        *,
        engine: Engine,
        metric_types: Sequence[MetricType],
        batch_size: int = 100_000,
        _model_record: EmbeddingModelRecord,
    ) -> None:
        """Drop and recreate SQL indices for each metric in *metric_types*.

        Use this after calling ``update_model_index_configuration`` to apply
        new HNSW parameters (``m``, ``ef_construction``) to the on-disk index.
        Postgres rebuilds index content from the table automatically.
        ``batch_size`` is accepted for interface compatibility but unused.
        """
        if model_name not in self._pgvector_index_managers:
            self._register_index_manager(engine=engine, model_record=_model_record)
        manager = self.get_index_manager(model_name)
        for metric_type in metric_types:
            manager.rebuild_index(metric_type)

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
        concept_id_tuple = tuple(concept_ids)

        self.validate_embeddings_and_concept_ids(
            concept_ids=concept_id_tuple,
            embeddings=embeddings,
            dimensions=_model_record.dimensions,
        )

        table = self.get_embedding_table(
            model_name=model_name,
            index_type=index_type,
            provider_type=provider_type,
        )

        existing = self._get_existing_concept_ids(session, concept_id_tuple, table)
        if existing:
            raise ValueError(
                f"concept_ids already present in registry for model '{model_name}': {existing}. "
                "Existing concept_ids cannot be overwritten."
            )

        add_embeddings_to_registered_table(
            session=session,
            concept_ids=concept_id_tuple,
            embeddings=embeddings,
            registered_table=table,
        )

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

        embedding_table = self.get_embedding_table(
            model_name=model_name,
            index_type=index_type,
            provider_type=provider_type,
        )
        query = q_embedding_vectors_by_concept_ids(
            embedding_table=embedding_table,
            concept_ids=concept_id_tuple,
        )

        return_dict = {
            int(row.concept_id): list(row.embedding)
            for row in session.execute(query)
        }

        missing_ids = set(concept_id_tuple) - set(return_dict.keys())
        if missing_ids:
            raise ValueError(
                f"Requested concept IDs {missing_ids} not found in the database for model '{model_name}'."
            )
        return return_dict

    @require_registered_model
    def get_nearest_concepts(
        self,
        *,
        session: Session,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        query_embeddings: ndarray,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        _model_record: EmbeddingModelRecord,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        embedding_table = self.get_embedding_table(
            model_name=model_name,
            index_type=index_type,
            provider_type=provider_type,
        )
        self.validate_embeddings(embeddings=query_embeddings, dimensions=_model_record.dimensions)

        # Set ef_search for HNSW queries so the session uses the configured value.
        if index_type == IndexType.HNSW:
            manager = self.get_index_manager(model_name)
            if isinstance(manager, PGVectorHNSWIndexManager):
                session.execute(
                    text(f"SET hnsw.ef_search = {manager.index_config.ef_search}")
                )

        # Guarantee that concept_filter has a limit set for K nearest neighbors
        if concept_filter is None or concept_filter.limit is None:
            logger.debug(
                f"No concept filter limit provided. "
                f"Setting k to default: {self.DEFAULT_K_NEAREST}"
            )
            if concept_filter is None:
                concept_filter = EmbeddingConceptFilter(limit=self.DEFAULT_K_NEAREST)
            else:
                concept_filter = EmbeddingConceptFilter(
                    concept_ids=concept_filter.concept_ids,
                    domains=concept_filter.domains,
                    vocabularies=concept_filter.vocabularies,
                    require_standard=concept_filter.require_standard,
                    limit=self.DEFAULT_K_NEAREST,
                )

        query_list = query_embeddings.tolist()
        query = q_embedding_nearest_concepts(
            embedding_table=embedding_table,
            query_embeddings=query_list,
            metric_type=metric_type,
            concept_filter=concept_filter,
        )

        rows = session.execute(query).all()
        results = [[] for _ in range(len(query_list))]

        for row in rows:
            results[row.q_id].append(
                NearestConceptMatch(
                    concept_id=int(row.concept_id),
                    concept_name=row.concept_name,
                    similarity=float(row.similarity),
                    is_standard=bool(row.is_standard),
                    is_active=bool(row.is_active),
                )
            )

        matches_tuple = tuple(tuple(matches) for matches in results)

        k = concept_filter.limit
        if k is None:
            raise RuntimeError(
                "Internal error: concept_filter.limit should have been set by this point."
            )
        self.validate_nearest_concepts_output(matches_tuple, k, query_embeddings=query_embeddings)
        return matches_tuple
