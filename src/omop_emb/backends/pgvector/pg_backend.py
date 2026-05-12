from __future__ import annotations

import logging
from typing import Mapping, Optional, Sequence, Tuple, Type

from numpy import ndarray
from sqlalchemy import Engine, select, text, create_engine

try:
    from pgvector.sqlalchemy import Vector
except ImportError as _e:
    raise ImportError(
        "pgvector is not installed. Install it with: pip install omop-emb[pgvector]"
    ) from _e

from omop_emb.config import BackendType, MetricType, ProviderType
from omop_emb.backends.base_backend import (
    ConceptEmbeddingRecord,
    EmbeddingBackend,
)
from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig, IndexConfig
from omop_emb.backends.pgvector.pg_index_manager import (
    PGVectorBaseIndexManager,
    PGVectorFlatIndexManager,
    PGVectorHNSWIndexManager,
)
from omop_emb.backends.embedding_table import (
    create_pg_embedding_table, 
    load_pg_embedding_table
)
from omop_emb.backends.pgvector.pg_sql import (
    EMBEDDING_COLUMN_NAME,
    drop_pg_embedding_table,
    get_distance,
    q_all_concept_ids,
    q_nearest_concept_ids,
    q_upsert_embeddings,
    q_create_extension_pgvector,
    table_exists,
)
from omop_emb.backends.db_utils import setup_concept_filter_temps, temp_filter_table
from omop_emb.model_registry import EmbeddingModelRecord
from omop_emb.utils.embedding_utils import (
    EmbeddingConceptFilter,
    NearestConceptMatch,
    get_similarity_from_distance,
    vector_column_type_for_dimensions
)

logger = logging.getLogger(__name__)

_INDEX_MANAGER_FOR_CONFIG: dict[type[IndexConfig], type[PGVectorBaseIndexManager]] = {
    FlatIndexConfig: PGVectorFlatIndexManager,
    HNSWIndexConfig: PGVectorHNSWIndexManager,
}


class PGVectorEmbeddingBackend(EmbeddingBackend):
    """pgvector-backed embedding backend.

    Both embedding tables and the model registry live in the same Postgres
    instance. Supports ``FLAT`` and ``HNSW`` index types and all pgvector
    distance metrics.

    Parameters
    ----------
    emb_engine : Engine
        SQLAlchemy engine connected to the pgvector database.

    Notes
    -----
    Call :meth:`pre_initialise_store` once to create the ``vector`` extension
    if it does not already exist. This is called automatically during
    ``__init__``.
    """

    def __init__(self, emb_engine: Engine) -> None:
        self._index_managers: dict[str, PGVectorBaseIndexManager] = {}
        super().__init__(emb_engine=emb_engine)

    @classmethod
    def from_db_url(cls, db_url: str) -> PGVectorEmbeddingBackend:
        """Create a pgvector embedding backend from a database URL.

        Parameters
        ----------
        db_url : str
            Database URL in SQLAlchemy format, e.g. ``postgresql://user:pass@host:port/dbname``.

        Returns
        -------
        PGVectorEmbeddingBackend
        """
        engine = create_engine(db_url, echo=False)
        return cls(emb_engine=engine)

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
        """Create the pgvector extension if it does not exist."""
        with self.emb_engine.begin() as conn:
            conn.execute(q_create_extension_pgvector())

    # ------------------------------------------------------------------
    # Storage table management
    # ------------------------------------------------------------------

    def _storage_table_exists(self, model_record: EmbeddingModelRecord) -> bool:
        return table_exists(self.emb_engine, model_record.storage_identifier)

    def _get_storage_table_descriptor(self, model_record: EmbeddingModelRecord) -> type:
        return load_pg_embedding_table(model_record=model_record)

    def _create_storage_table(self, model_record: EmbeddingModelRecord) -> type:
        return create_pg_embedding_table(engine=self.emb_engine, model_record=model_record)

    def _delete_storage_table(self, model_record: EmbeddingModelRecord) -> None:
        self._index_managers.pop(model_record.storage_identifier, None)
        drop_pg_embedding_table(engine=self.emb_engine, model_record=model_record)

    # ------------------------------------------------------------------
    # Model registration override
    # ------------------------------------------------------------------

    def register_model(
        self,
        *,
        model_name: str,
        dimensions: int,
        provider_type: ProviderType,
        index_config: Optional[IndexConfig] = None,
        metadata=None,
    ) -> EmbeddingModelRecord:
        """Register a model, validating pgvector dimensionality limits.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        dimensions : int
            Embedding vector dimensionality.
        provider_type : ProviderType
            Provider that serves the model.
        index_config : IndexConfig, optional
            Defaults to ``FlatIndexConfig()`` when not provided.
        metadata : Mapping[str, object], optional
            Free-form operational metadata.

        Returns
        -------
        EmbeddingModelRecord

        Raises
        ------
        ValueError
            If ``dimensions`` exceeds the pgvector halfvec limit of 4 000.
        """
        vector_column_type_for_dimensions(dimensions)  # validates halfvec limit
        return super().register_model(
            model_name=model_name,
            dimensions=dimensions,
            provider_type=provider_type,
            index_config=index_config,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _rebuild_index_impl(
        self, *, model_record: EmbeddingModelRecord, index_config: IndexConfig
    ) -> None:
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
        metric = index_config.metric_type or MetricType.COSINE
        manager.rebuild_index(metric)
        self._index_managers[model_record.storage_identifier] = manager

    def get_index_manager(self, storage_identifier: str) -> Optional[PGVectorBaseIndexManager]:
        """Return the active index manager for a table, or ``None`` if not set.

        Parameters
        ----------
        storage_identifier : str
            Physical table name.

        Returns
        -------
        PGVectorBaseIndexManager or None
        """
        return self._index_managers.get(storage_identifier)

    # ------------------------------------------------------------------
    # Core write operations
    # ------------------------------------------------------------------

    def _upsert_embeddings_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        records: Sequence[ConceptEmbeddingRecord],
        embeddings: ndarray,
    ) -> None:
        self.validate_embeddings_and_records(
            embeddings=embeddings,
            records=records,
            dimensions=model_record.dimensions,
        )
        table = self._table_cache[model_record.storage_identifier]
        stmt = q_upsert_embeddings(
            records=records,
            embeddings=embeddings,
            registered_table=table,
        )
        try:
            with self.emb_session_factory.begin() as session:
                session.execute(stmt)
        except Exception as exc:
            logger.error(
                "Failed to upsert embeddings for '%s': %s", model_record.model_name, exc
            )
            raise

    # ------------------------------------------------------------------
    # Core read operations
    # ------------------------------------------------------------------

    def _get_embeddings_by_concept_ids_impl(
        self,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
    ) -> Mapping[int, Sequence[float]]:
        if not concept_ids:
            return {}
        table = self._table_cache[model_record.storage_identifier]
        with self.emb_session_factory.begin() as session:
            with temp_filter_table(
                session, list(concept_ids), "BIGINT", table_name="_tmp_emb_cids", dialect=self.emb_engine.dialect.name
            ) as temp_table_name:
                rows = session.execute(
                    select(table.concept_id, table.embedding)
                    .where(text(f'concept_id IN (SELECT id FROM "{temp_table_name}")'))
                ).all()
        result = {int(row[0]): list(row[1]) for row in rows}
        missing = set(concept_ids) - set(result.keys())
        if missing:
            raise ValueError(
                f"Concept IDs {missing} not found for model '{model_record.model_name}'."
            )
        return result

    def _get_nearest_concepts_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        metric_type: MetricType,
        query_embeddings: ndarray,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        k: int = EmbeddingBackend.DEFAULT_K_NEAREST,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        self.validate_embeddings(query_embeddings, model_record.dimensions)

        manager = self._index_managers.get(model_record.storage_identifier)
        table = self._table_cache[model_record.storage_identifier]

        with self.emb_session_factory.begin() as session:
            if isinstance(manager, PGVectorHNSWIndexManager):
                session.execute(
                    text(f"SET hnsw.ef_search = {manager.index_config.ef_search}")
                )

            if concept_filter is not None:
                setup_concept_filter_temps(session, concept_filter, "postgresql")

            stmt = q_nearest_concept_ids(
                embedding_table=table,
                query_embeddings=query_embeddings.tolist(),
                metric_type=metric_type,
                k=k,
                concept_filter=concept_filter,
            )
            ann_rows = session.execute(stmt).all()

        results: list[list[NearestConceptMatch]] = [[] for _ in range(len(query_embeddings))]
        for row in ann_rows:
            similarity = get_similarity_from_distance(float(row.distance), metric_type)
            results[row.q_id].append(
                NearestConceptMatch(
                    concept_id=int(row.concept_id),
                    similarity=float(similarity),
                    is_standard=bool(row.is_standard),
                )
            )

        return tuple(tuple(r) for r in results)

    # ------------------------------------------------------------------
    # Utility queries
    # ------------------------------------------------------------------

    def _has_any_embeddings_impl(self, *, model_record: EmbeddingModelRecord) -> bool:
        table = self._table_cache[model_record.storage_identifier]
        with self.emb_session_factory() as session:
            return session.execute(select(table.concept_id).limit(1)).first() is not None

    def _get_all_stored_concept_ids_impl(self, *, model_record: EmbeddingModelRecord) -> set[int]:
        table = self._table_cache[model_record.storage_identifier]
        with self.emb_session_factory() as session:
            return {row[0] for row in session.execute(q_all_concept_ids(table))}

    def _get_concept_filter_metadata_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
    ) -> Mapping[int, Mapping[str, object]]:
        if not concept_ids:
            return {}
        table = self._table_cache[model_record.storage_identifier]
        with self.emb_session_factory.begin() as session:
            with temp_filter_table(
                session, list(concept_ids), "BIGINT", table_name="_tmp_cfm_cids", dialect=self.emb_engine.dialect.name
            ) as temp_table_name:
                rows = session.execute(
                    select(
                        table.concept_id,
                        table.domain_id,
                        table.vocabulary_id,
                        table.is_standard,
                        table.is_valid,
                    ).where(text(f'concept_id IN (SELECT id FROM "{temp_table_name}")'))
                ).all()
        return {
            int(row[0]): {
                "domain_id": row[1] or "",
                "vocabulary_id": row[2] or "",
                "is_standard": bool(row[3]),
                "is_valid": bool(row[4]),
            }
            for row in rows
        }
