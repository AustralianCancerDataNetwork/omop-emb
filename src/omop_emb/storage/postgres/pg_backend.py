"""pgvector-backed embedding backend.

Architecture changes from the old design
-----------------------------------------
* Two mandatory engines: ``emb_engine`` (pgvector instance) and
  ``omop_cdm_engine`` (user's OMOP CDM).
* The registry lives in ``emb_engine``, not SQLite.
* ``_initialise_store()`` is called in ``__init__`` — there is no separate
  setup call the user must remember.
* ANN search is split into two steps: a pgvector-only LATERAL query followed
  by a CDM metadata fetch.  This supports cross-database deployments where the
  CDM uses a dialect other than Postgres.
* Dimensions > 2 000 automatically use the ``halfvec`` column type (float16).
"""
from __future__ import annotations

import logging
from typing import Dict, Mapping, Optional, Sequence, Tuple, Type

from numpy import ndarray
from sqlalchemy import Engine, text

from omop_emb.config import (
    BackendType,
    IndexType,
    MetricType,
    ProviderType,
    VectorColumnType,
    vector_column_type_for_dimensions,
)
from omop_emb.storage.base import EmbeddingBackend, require_registered_model
from omop_emb.storage.index_config import (
    FlatIndexConfig,
    HNSWIndexConfig,
    IndexConfig,
    index_config_from_index_type_and_metadata,
)
from omop_emb.storage.postgres.pg_index_manager import (
    PGVectorBaseIndexManager,
    PGVectorFlatIndexManager,
    PGVectorHNSWIndexManager,
)
from omop_emb.storage.postgres.pg_registry import (
    EmbeddingModelRecord,
    PostgresRegistryManager,
)
from omop_emb.storage.postgres.pg_sql import (
    EMBEDDING_COLUMN_NAME,
    PGVectorConceptIDEmbeddingTable,
    create_pg_embedding_table,
    delete_pg_embedding_table,
    q_add_embeddings_to_registered_table,
    q_embedding_vectors_by_concept_ids,
    q_nearest_concept_ids,
    get_distance,
)
from omop_emb.utils.embedding_utils import (
    EmbeddingConceptFilter,
    NearestConceptMatch,
    get_similarity_from_distance,
)

logger = logging.getLogger(__name__)

_INDEX_MANAGER_FOR_CONFIG: dict[type[IndexConfig], type[PGVectorBaseIndexManager]] = {
    FlatIndexConfig: PGVectorFlatIndexManager,
    HNSWIndexConfig: PGVectorHNSWIndexManager,
}


class PGVectorEmbeddingBackend(EmbeddingBackend[PGVectorConceptIDEmbeddingTable]):
    """pgvector-backed embedding backend.

    Parameters
    ----------
    emb_engine : Engine
        SQLAlchemy engine for the **dedicated pgvector Postgres instance**.
        Both embedding tables and the model registry are stored here.
    omop_cdm_engine : Engine
        SQLAlchemy engine for the **user's OMOP CDM** (read-only).  May be
        any SQLAlchemy-supported dialect; used only for concept metadata queries.
    """

    def __init__(self, emb_engine: Engine, omop_cdm_engine: Engine) -> None:
        self._registry = PostgresRegistryManager(emb_engine=emb_engine)
        self._pgvector_index_managers: Dict[str, PGVectorBaseIndexManager] = {}
        super().__init__(emb_engine=emb_engine, omop_cdm_engine=omop_cdm_engine)

    # ------------------------------------------------------------------
    # Backend identity
    # ------------------------------------------------------------------

    @property
    def backend_type(self) -> BackendType:
        return BackendType.PGVECTOR

    # ------------------------------------------------------------------
    # Store lifecycle
    # ------------------------------------------------------------------

    def pre_initialise_store(self) -> None:
        """Ensure the ``vector`` extension exists in the pgvector instance."""
        with self.emb_engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector CASCADE;"))

    # ------------------------------------------------------------------
    # Storage table management
    # ------------------------------------------------------------------

    def _create_storage_table(
        self,
        model_record: EmbeddingModelRecord,
    ) -> Type[PGVectorConceptIDEmbeddingTable]:
        table = create_pg_embedding_table(engine=self.emb_engine, model_record=model_record)
        try:
            self._register_index_manager(model_record=model_record)
        except Exception:
            delete_pg_embedding_table(engine=self.emb_engine, model_record=model_record)
            raise
        return table

    def _delete_storage_table(self, model_record: EmbeddingModelRecord) -> None:
        self._pgvector_index_managers.pop(model_record.model_name, None)
        delete_pg_embedding_table(engine=self.emb_engine, model_record=model_record)

    # ------------------------------------------------------------------
    # Model registration
    # ------------------------------------------------------------------

    def register_model(
        self,
        *,
        model_name: str,
        dimensions: int,
        provider_type: ProviderType,
        index_config: IndexConfig,
        metadata=None,
    ) -> EmbeddingModelRecord:
        # Validate dimensions against the maximum supported by halfvec.
        # vector_column_type_for_dimensions raises if > 4 000.
        vector_column_type_for_dimensions(dimensions)
        return super().register_model(
            model_name=model_name,
            dimensions=dimensions,
            provider_type=provider_type,
            index_config=index_config,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Index manager lifecycle
    # ------------------------------------------------------------------

    def _register_index_manager(self, model_record: EmbeddingModelRecord) -> None:
        if model_record.model_name in self._pgvector_index_managers:
            return
        index_config = index_config_from_index_type_and_metadata(
            model_record.index_type, model_record.metadata
        )
        manager_cls = _INDEX_MANAGER_FOR_CONFIG.get(type(index_config))
        if manager_cls is None:
            raise ValueError(
                f"No pgvector index manager for config type {type(index_config).__name__}."
            )
        manager = manager_cls(
            emb_engine=self.emb_engine,
            tablename=model_record.storage_identifier,
            embedding_column=EMBEDDING_COLUMN_NAME,
            index_config=index_config,
            dimensions=model_record.dimensions,
        )
        self._pgvector_index_managers[model_record.model_name] = manager

    def get_index_manager(self, model_name: str) -> PGVectorBaseIndexManager:
        manager = self._pgvector_index_managers.get(model_name)
        if manager is None:
            raise ValueError(
                f"No pgvector index manager for model '{model_name}'. "
                "Ensure the model is registered."
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
        new_record = super().update_model_index_configuration(
            model_name=model_name,
            provider_type=provider_type,
            index_type=index_type,
            index_config=index_config,
        )
        self._pgvector_index_managers.pop(model_name, None)
        logger.warning(
            f"Index configuration for '{model_name}' updated. "
            "SQL indexes are NOT rebuilt automatically — call rebuild_model_indexes()."
        )
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
        """Ensure SQL indices exist for each metric in *metric_types* (idempotent)."""
        if model_name not in self._pgvector_index_managers:
            self._register_index_manager(model_record=_model_record)
        manager = self.get_index_manager(model_name)
        for metric_type in metric_types:
            manager.load_or_create(metric_type)

    def _rebuild_model_indexes_impl(
        self,
        model_record: EmbeddingModelRecord,
        *,
        metric_types: Sequence[MetricType],
        batch_size: int = 100_000,
    ) -> None:
        if model_record.model_name not in self._pgvector_index_managers:
            self._register_index_manager(model_record=model_record)
        manager = self.get_index_manager(model_record.model_name)
        for metric_type in metric_types:
            manager.rebuild_index(metric_type)

    # ------------------------------------------------------------------
    # Core write operations
    # ------------------------------------------------------------------

    def _upsert_embeddings_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
        embeddings: ndarray,
        metric_type: Optional[MetricType] = None,
    ) -> None:
        concept_id_tuple = tuple(concept_ids)
        self.validate_embeddings_and_concept_ids(
            concept_ids=concept_id_tuple,
            embeddings=embeddings,
            dimensions=model_record.dimensions,
        )
        table = self.get_embedding_table(
            model_name=model_record.model_name,
            index_type=model_record.index_type,
            provider_type=model_record.provider_type,
        )
        existing = self._get_existing_concept_ids(concept_id_tuple, table)
        if existing:
            raise ValueError(
                f"concept_ids already present for model '{model_record.model_name}': {existing}. "
                "Existing embeddings cannot be overwritten."
            )
        stmt = q_add_embeddings_to_registered_table(
            concept_ids=concept_id_tuple,
            embeddings=embeddings,
            registered_table=table,
        )
        try:
            with self.emb_session_factory.begin() as session:
                session.execute(stmt)
        except Exception as exc:
            logger.error(
                f"Failed to upsert embeddings for model '{model_record.model_name}': {exc}"
            )
            raise

    # ------------------------------------------------------------------
    # Core read operations
    # ------------------------------------------------------------------

    def _get_embeddings_by_concept_ids_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
    ) -> Mapping[int, Sequence[float]]:
        concept_id_tuple = tuple(concept_ids)
        if not concept_id_tuple:
            return {}
        table = self.get_embedding_table(
            model_name=model_record.model_name,
            index_type=model_record.index_type,
            provider_type=model_record.provider_type,
        )
        stmt = q_embedding_vectors_by_concept_ids(
            embedding_table=table,
            concept_ids=concept_id_tuple,
        )
        with self.emb_session_factory() as session:
            result = {int(row.concept_id): list(row.embedding) for row in session.execute(stmt)}

        missing = set(concept_id_tuple) - set(result.keys())
        if missing:
            raise ValueError(
                f"Concept IDs {missing} not found for model '{model_record.model_name}'."
            )
        return result

    def _get_nearest_concepts_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        query_embeddings: ndarray,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        self.validate_embeddings(query_embeddings, model_record.dimensions)

        # Resolve k
        k = (concept_filter.limit if concept_filter else None) or self.DEFAULT_K_NEAREST

        # Set ef_search for HNSW at session level
        if model_record.index_type == IndexType.HNSW:
            manager = self.get_index_manager(model_record.model_name)
            if isinstance(manager, PGVectorHNSWIndexManager):
                with self.emb_session_factory.begin() as session:
                    session.execute(
                        text(f"SET hnsw.ef_search = {manager.index_config.ef_search}")
                    )

        # Step 1: Pre-filter via CDM if the filter has CDM criteria
        candidate_ids = self._get_candidate_concept_ids_from_cdm(concept_filter)

        # Step 2: ANN search on pgvector instance
        table = self.get_embedding_table(
            model_name=model_record.model_name,
            index_type=model_record.index_type,
            provider_type=model_record.provider_type,
        )
        stmt = q_nearest_concept_ids(
            embedding_table=table,
            query_embeddings=query_embeddings.tolist(),
            metric_type=metric_type,
            k=k,
            candidate_concept_ids=candidate_ids,
        )
        with self.emb_session_factory() as session:
            ann_rows = session.execute(stmt).all()

        # Step 3: Fetch concept metadata from CDM
        unique_ids = {row.concept_id for row in ann_rows}
        concept_metadata = self._fetch_concept_metadata(unique_ids)

        # Step 4: Build NearestConceptMatch objects
        results: list[list[NearestConceptMatch]] = [[] for _ in range(len(query_embeddings))]
        for row in ann_rows:
            meta = concept_metadata.get(row.concept_id)
            if meta is None:
                logger.debug(
                    f"concept_id {row.concept_id} not found in CDM — skipping."
                )
                continue
            similarity = get_similarity_from_distance(float(row.distance), metric_type)
            results[row.q_id].append(
                NearestConceptMatch(
                    concept_id=int(row.concept_id),
                    concept_name=meta.concept_name,
                    similarity=float(similarity),
                    is_standard=meta.standard_concept in ("S", "C"),
                    is_active=meta.invalid_reason not in ("D", "U"),
                )
            )

        matches = tuple(tuple(r) for r in results)
        self.validate_nearest_concepts_output(matches, k, query_embeddings)
        return matches
