from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from functools import cached_property

from numpy import ndarray
from oa_configurator import (
    Dialect,
    Role,
    guard_schema_provenance_for,
    physical_schema_of,
    qualified,
)
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session

try:
    from pgvector.sqlalchemy import Vector  # noqa: F401
except ImportError as _e:
    raise ImportError(
        "pgvector is not installed. Install it with: pip install omop-emb[pgvector]"
    ) from _e

from omop_emb.backends.base_backend import (
    CompactionResult,
    ConceptEmbeddingRecord,
    EmbeddingBackend,
    writes,
)
from omop_emb.backends.embedding_table import (
    PGEmbeddingTable,
)
from omop_emb.backends.index_config import HNSWIndexConfig, IndexConfig
from omop_emb.backends.pgvector.pg_sql import (
    create_pg_embedding_table,
    delete_embedding_rows,
    drop_pg_embedding_table,
    hnsw_index_ddl,
    installed_pgvector_version,
    pg_embedding_table_descriptor,
    q_all_concept_ids,
    q_embedding_count_by_vocabulary,
    query_concept_filter_metadata,
    query_concept_ids_matching_filter,
    query_embeddings_by_ids,
    query_nearest_concept_ids,
    table_exists,
    upsert_embedding_rows,
)
from omop_emb.config import BackendType, MetricType
from omop_emb.model_registry import EmbeddingModelRecord
from omop_emb.utils.embedding_utils import (
    EmbeddingArray,
    EmbeddingConceptFilter,
    NearestConceptMatch,
    get_similarity_from_distance,
    vector_column_type_for_dimensions,
)

logger = logging.getLogger(__name__)


def _validate_pg_registration(dimensions: int) -> None:
    """Validate pgvector dimensions, including the halfvec limit."""
    vector_column_type_for_dimensions(dimensions)


def _create_vector_extension(dbapi_connection, _connection_record) -> None:
    """Connect-event callable: ensure the pgvector extension exists on this connection.

    Passed as an ``extensions`` callable to ``ResolvedDatabase.create_engine()``
    (see its docstring). ``CREATE EXTENSION IF NOT EXISTS`` is atomic and
    idempotent, so re-running it on every connection is safe and cheap.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector CASCADE;")
        dbapi_connection.commit()
    finally:
        cursor.close()


def _set_read_only(dbapi_connection, _connection_record) -> None:
    """Connect-event callable: make every transaction on this connection read-only.

    Passed as an ``extensions`` callable by ``open_vector_store_reader()``.
    PostgreSQL then rejects any DDL or write, temporary tables included.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY;")
        dbapi_connection.commit()
    finally:
        cursor.close()


class PGVectorEmbeddingBackend(EmbeddingBackend[type[PGEmbeddingTable]]):
    """pgvector-backed embedding backend.

    Both embedding tables and the model registry live in the same Postgres
    instance. Supports ``FLAT`` and ``HNSW`` index types and all pgvector
    distance metrics.

    Parameters
    ----------
    emb_engine : Engine
        SQLAlchemy engine connected to the pgvector database. On the writer
        path, ``_create_vector_extension`` is attached as an ``extensions``
        callable and creates the ``vector`` extension; the reader path runs no
        DDL and relies on it already existing.
    writable : bool, optional
        Forwarded to ``EmbeddingBackend``.
    """

    # Instance variables
    _MAX_EF_SEARCH = 1000
    _filtered_hnsw_warned = False

    # ------------------------------------------------------------------
    # Backend identity
    # ------------------------------------------------------------------

    @property
    def backend_type(self) -> BackendType:
        return BackendType.PGVECTOR

    @property
    def dialect(self) -> str:
        return Dialect.POSTGRESQL

    @property
    def default_flat_metric(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Storage table management
    # ------------------------------------------------------------------

    def _storage_table_exists(self, model_record: EmbeddingModelRecord) -> bool:
        return table_exists(self.emb_engine, model_record.storage_identifier)

    def _get_storage_table_descriptor(
        self, model_record: EmbeddingModelRecord
    ) -> type[PGEmbeddingTable]:
        return pg_embedding_table_descriptor(model_record=model_record)

    def _create_storage_table(self, model_record: EmbeddingModelRecord) -> type[PGEmbeddingTable]:
        return create_pg_embedding_table(
            engine=self.emb_engine,
            model_record=model_record,
        )

    def _delete_storage_table(self, model_record: EmbeddingModelRecord) -> None:
        drop_pg_embedding_table(engine=self.emb_engine, model_record=model_record)

    # ------------------------------------------------------------------
    # Model registration override
    # ------------------------------------------------------------------

    @writes
    def register_model(
        self,
        *,
        model_name: str,
        dimensions: int,
        provider_type: str,
        index_config: IndexConfig | None = None,
        metadata: Mapping[str, object] | None = None,
        registered_at: datetime | None = None,
    ) -> EmbeddingModelRecord:
        """Register a model, validating pgvector dimensionality limits.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        dimensions : int
            Embedding vector dimensionality.
        provider_type : str
            omop-llm provider key that serves the model.
        index_config : IndexConfig, optional
            Defaults to ``FlatIndexConfig()`` when not provided.
        metadata : Mapping[str, object], optional
            Free-form operational metadata.
        registered_at : datetime, optional
            Backdate ``created_at``/``updated_at`` to this timestamp instead
            of "now". Only applies to a brand-new registration.

        Returns
        -------
        EmbeddingModelRecord

        Raises
        ------
        ValueError
            If ``dimensions`` exceeds the pgvector halfvec limit of 4 000, or
            for any reason :meth:`EmbeddingBackend.register_model` raises
            (a reserved metadata key, or a non-``FlatIndexConfig`` index).
        ModelRegistrationConflictError
            If the model is already registered with a different dimensionality.
        """
        _validate_pg_registration(dimensions)
        return super().register_model(
            model_name=model_name,
            dimensions=dimensions,
            provider_type=provider_type,
            index_config=index_config,
            metadata=metadata,
            registered_at=registered_at,
        )

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _rebuild_index_impl(
        self, *, model_record: EmbeddingModelRecord, index_config: IndexConfig, vacuum: bool
    ) -> None:
        """Drop every physical index on the table, then create the HNSW index index_config describes.

        With ``vacuum``, ``VACUUM (ANALYZE)`` runs on the table afterwards so
        the planner sees current statistics. It does not shrink the table on
        disk, see ``compact_storage``.
        """
        with (
            self.emb_engine.begin() as conn,
            guard_schema_provenance_for(conn, schema_tag=Role.PRIMARY),
        ):
            for statement in self.drop_index_sql(model_record.model_name):
                conn.execute(text(statement))
            if isinstance(index_config, HNSWIndexConfig):
                conn.execute(text(hnsw_index_ddl(self.emb_engine, model_record, index_config)))
        logger.info(
            f"Rebuilt '{model_record.storage_identifier}' with index type '{index_config.index_type.value}'."
        )
        if vacuum:
            self._vacuum_analyze(model_record)

    def _vacuum_analyze(self, model_record: EmbeddingModelRecord) -> None:
        """Run ``VACUUM (ANALYZE)`` on the model's table."""
        self._run_vacuum(model_record, "ANALYZE")

    def _compact_storage_impl(self, *, model_record: EmbeddingModelRecord) -> CompactionResult:
        """Run ``VACUUM FULL`` on the model's table, which rewrites it and rebuilds its indexes.

        Plain ``VACUUM`` only makes dead space reusable, so the table keeps its
        size on disk. ``VACUUM FULL`` holds an exclusive lock on the table.
        """
        size_before = self._relation_size(model_record)
        self._run_vacuum(model_record, "FULL")
        return CompactionResult(size_before, self._relation_size(model_record))

    def _qualified_table(self, model_record: EmbeddingModelRecord) -> str:
        return qualified(
            self.emb_engine,
            model_record.storage_identifier,
            physical_schema=physical_schema_of(self.emb_engine),
        )

    def _run_vacuum(self, model_record: EmbeddingModelRecord, options: str) -> None:
        """Run ``VACUUM`` with *options* on the model's table; PostgreSQL forbids it inside a transaction."""
        table = self._qualified_table(model_record)
        with (
            self.emb_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn,
            guard_schema_provenance_for(conn, schema_tag=Role.PRIMARY),
        ):
            conn.execute(text(f"VACUUM ({options}) {table}"))

    def _relation_size(self, model_record: EmbeddingModelRecord) -> int:
        """Return the size in bytes of the model's table with its TOAST data and indexes."""
        table = self._qualified_table(model_record)
        with (
            self.emb_engine.connect() as conn,
            guard_schema_provenance_for(conn, schema_tag=Role.PRIMARY),
        ):
            return int(
                conn.execute(
                    text("SELECT pg_total_relation_size(CAST(:table AS regclass))"),
                    {"table": table},
                ).scalar_one()
            )

    def physical_indexes(self, model_name: str) -> tuple[str, ...]:
        """Return existing indexes for this model's table without creating or
        changing them."""
        record = self.get_registered_model(model_name=model_name)
        if record is None:
            return ()
        physical_schema = physical_schema_of(self.emb_engine)
        expected_prefix = f"idx_{record.storage_identifier}_"
        return tuple(
            str(item["name"])
            for item in inspect(self.emb_engine).get_indexes(
                record.storage_identifier,
                schema=physical_schema,
            )
            if str(item["name"]).startswith(expected_prefix)
        )

    # ------------------------------------------------------------------
    # Core write operations
    # ------------------------------------------------------------------

    def _upsert_embeddings_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        records: Sequence[ConceptEmbeddingRecord],
        embeddings: EmbeddingArray,
    ) -> None:
        table = self._storage_table(model_record)
        try:
            with self.emb_session_factory.begin() as session:
                upsert_embedding_rows(
                    session=session,
                    records=records,
                    embeddings=embeddings,
                    registered_table=table,
                )
        except Exception as exc:
            logger.error("Failed to upsert embeddings for '%s': %s", model_record.model_name, exc)
            raise

    def _delete_embeddings_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
    ) -> int:
        table = self._storage_table(model_record)
        with self.emb_session_factory.begin() as session:
            return delete_embedding_rows(
                session=session, embedding_table=table, concept_ids=concept_ids
            )

    # ------------------------------------------------------------------
    # Core read operations
    # ------------------------------------------------------------------

    def _get_embeddings_by_concept_ids_impl(
        self,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
    ) -> Mapping[int, ndarray]:
        if not concept_ids:
            return {}
        table = self._storage_table(model_record)
        with self.emb_session_factory() as session:
            result = query_embeddings_by_ids(
                session=session, embedding_table=table, concept_ids=concept_ids
            )
        missing = set(concept_ids) - set(result.keys())
        if missing:
            raise ValueError(
                f"Concept IDs {missing} not found for model '{model_record.model_name}'."
            )
        return result

    @cached_property
    def _supports_iterative_scan(self) -> bool:
        "Iterative index scans were added in pgvector 0.8.0"
        with self.emb_session_factory() as session:
            return installed_pgvector_version(session) >= (0, 8, 0)

    def _configure_hnsw_scan(
        self, session: Session, *, ef_search: int, k: int, filtered: bool
    ) -> None:
        """Set the HNSW scan parameters for the current transaction.

        With pgvector 0.8.0 or newer the index keeps scanning until ``k`` rows
        pass the filter, so ``ef_search`` is used as configured. The scan runs
        in ``relaxed_order`` and the caller sorts the rows, because
        ``strict_order`` skips every row closer than the last one it returned
        and so loses neighbours found in later batches. Older versions stop at
        ``ef_search`` candidates, so it is raised to at least ``k`` (capped at
        1000) and a filtered query logs a one-time warning that its results
        may be incomplete.

        Parameters
        ----------
        session : Session
            Session whose open transaction receives the settings.
        ef_search : int
            Configured ``ef_search`` of the model.
        k : int
            Number of neighbours requested.
        filtered : bool
            Whether the query carries a concept filter.
        """
        if self._supports_iterative_scan:
            session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
        else:
            ef_search = min(max(ef_search, k), self._MAX_EF_SEARCH)
            if filtered and not self._filtered_hnsw_warned:
                self._filtered_hnsw_warned = True
                logger.warning(
                    "Filtered HNSW queries may return fewer than k results on pgvector "
                    "below 0.8.0. Upgrade the extension to enable iterative index scans."
                )
        session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))

    def _get_nearest_concepts_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        metric_type: MetricType,
        query_embeddings: EmbeddingArray,
        concept_filter: EmbeddingConceptFilter | None = None,
        k: int = EmbeddingBackend.DEFAULT_K_NEAREST,
    ) -> tuple[tuple[NearestConceptMatch, ...], ...]:
        table = self._storage_table(model_record)

        with self.emb_session_factory.begin() as session:
            if isinstance(model_record.index_config, HNSWIndexConfig):
                self._configure_hnsw_scan(
                    session,
                    ef_search=model_record.index_config.ef_search,
                    k=k,
                    filtered=concept_filter is not None,
                )

            ann_rows = [
                query_nearest_concept_ids(
                    session=session,
                    embedding_table=table,
                    query_embedding=query_embedding,
                    metric_type=metric_type,
                    k=k,
                    concept_filter=concept_filter,
                )
                for query_embedding in query_embeddings
            ]

        results = [
            [
                NearestConceptMatch(
                    concept_id=int(row.concept_id),
                    similarity=float(
                        get_similarity_from_distance(float(row.distance), metric_type)
                    ),
                    domain_id=row.domain_id,
                    vocabulary_id=row.vocabulary_id,
                    is_standard=bool(row.is_standard),
                    is_active=bool(row.is_valid),
                )
                for row in sorted(rows, key=lambda row: row.distance)
            ]
            for rows in ann_rows
        ]

        return tuple(tuple(r) for r in results)

    # ------------------------------------------------------------------
    # Utility queries
    # ------------------------------------------------------------------

    def _has_any_embeddings_impl(self, *, model_record: EmbeddingModelRecord) -> bool:
        table = self._storage_table(model_record)
        with self.emb_session_factory() as session:
            return session.execute(select(table.concept_id).limit(1)).first() is not None

    def _get_all_stored_concept_ids_impl(self, *, model_record: EmbeddingModelRecord) -> set[int]:
        table = self._storage_table(model_record)
        with self.emb_session_factory() as session:
            return {row[0] for row in session.execute(q_all_concept_ids(table))}

    def _get_concept_filter_metadata_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
    ) -> Mapping[int, ConceptEmbeddingRecord]:
        if not concept_ids:
            return {}
        table = self._storage_table(model_record)
        concept_filter = EmbeddingConceptFilter(concept_ids=tuple(concept_ids))
        with self.emb_session_factory() as session:
            rows = query_concept_filter_metadata(
                session=session,
                embedding_table=table,
                concept_filter=concept_filter,
            )
        return {
            int(row[0]): ConceptEmbeddingRecord(
                concept_id=int(row[0]),
                domain_id=row[1] or "",
                vocabulary_id=row[2] or "",
                is_standard=bool(row[3]),
                is_valid=bool(row[4]),
            )
            for row in rows
        }

    def _get_concept_ids_matching_filter_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        concept_filter: EmbeddingConceptFilter,
    ) -> set[int]:
        table = self._storage_table(model_record)
        with self.emb_session_factory() as session:
            return query_concept_ids_matching_filter(
                session=session,
                embedding_table=table,
                concept_filter=concept_filter,
            )

    def _get_embedding_count_by_vocabulary_impl(
        self, *, model_record: EmbeddingModelRecord
    ) -> Mapping[str, int]:
        table = self._storage_table(model_record)
        with self.emb_session_factory() as session:
            rows = session.execute(q_embedding_count_by_vocabulary(table)).all()
        return {row[0]: int(row[1]) for row in rows}
