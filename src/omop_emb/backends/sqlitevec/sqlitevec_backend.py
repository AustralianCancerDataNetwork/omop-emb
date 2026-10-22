"""sqlite-vec embedding backend (default, no external database required).

All data lives in a single .db file. vec0 virtual tables (embeddings) and
the registry table share the same file and SQLAlchemy engine.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from numpy import ndarray
from oa_configurator import Dialect
from sqlalchemy import Engine, MetaData, Table, text
from sqlalchemy.exc import OperationalError

try:
    import sqlite_vec
except ImportError as _e:
    raise ImportError(
        "sqlite-vec is not installed. Install it with: pip install sqlite-vec"
    ) from _e

from omop_emb.backends.base_backend import (
    CompactionResult,
    ConceptEmbeddingRecord,
    EmbeddingBackend,
)
from omop_emb.backends.index_config import FlatIndexConfig, IndexConfig
from omop_emb.backends.sqlitevec.sqlitevec_sql import (
    database_size,
    ddl_create_vec0,
    ddl_drop_vec0,
    declared_metric,
    dml_delete_rows,
    dml_upsert_rows,
    freelist_pages,
    query_all_concept_ids,
    query_concept_ids_matching_filter,
    query_embedding_count_by_vocabulary,
    query_embeddings_by_ids,
    query_has_any,
    query_knn_batch,
    query_metadata_by_ids,
    reclaimable_chunks,
    recreate_vec0,
    sqlite_vec_table_descriptor,
    table_exists,
    vacuum_database,
)
from omop_emb.config import BackendType, MetricType
from omop_emb.model_registry import EmbeddingModelRecord
from omop_emb.utils.embedding_utils import (
    EmbeddingArray,
    EmbeddingConceptFilter,
    NearestConceptMatch,
    get_similarity_from_distance,
)
from omop_emb.utils.errors import LegacyVectorTableError, UnsupportedIndexConfigError

logger = logging.getLogger(__name__)


def _load_sqlite_vec(dbapi_connection, _connection_record) -> None:
    """Connect-event callable: load the sqlite-vec extension on this connection.

    Passed as an ``extensions`` callable to ``ResolvedDatabase.create_engine()``
    (see its docstring). Loading a SQLite extension is inherently a
    per-connection operation, so this is safe to run on every new connection.
    """
    dbapi_connection.enable_load_extension(True)
    sqlite_vec.load(dbapi_connection)
    dbapi_connection.enable_load_extension(False)


def _set_query_only(dbapi_connection, _connection_record) -> None:
    """Connect-event callable: make this connection reject every write.

    Passed as an ``extensions`` callable by ``open_vector_store_reader()``.
    SQLite then rejects any DDL or write, temporary tables included.
    """
    dbapi_connection.execute("PRAGMA query_only = ON")


class SQLiteVecEmbeddingBackend(EmbeddingBackend[Table]):
    """sqlite-vec embedding backend.

    All data lives in a single .db file. vec0 virtual tables (embeddings) and
    the registry table share the same file and SQLAlchemy engine.

    Notes
    -----
    - Only the ``FLAT`` index type is supported. Each model is locked to one
      metric (L2, cosine or L1, cosine by default), declared on its vec0 table
      and recorded in the registry. ``rebuild_index`` recreates the table from
      the stored vectors when the metric differs or the table was created
      without a declared metric.
    - vec0 stores rows in fixed-size chunks and does not fill the free slots
      of an earlier chunk, so deleted or rewritten rows leave space that
      ``VACUUM`` alone does not return. ``compact_storage`` recreates the
      table to return it.
    - vec0 is a SQLite virtual table type the SQLAlchemy ORM cannot map, but
      it accepts ordinary SQL once created. ``_table_cache`` therefore stores
      a Core :class:`~sqlalchemy.Table` used for every query except the initial
      ``CREATE VIRTUAL TABLE``.
    """

    def __init__(
        self,
        emb_engine: Engine,
        *,
        writable: bool = True,
    ) -> None:
        self._sqlite_vec_metadata = MetaData()
        super().__init__(emb_engine=emb_engine, writable=writable)

    # ------------------------------------------------------------------
    # Backend identity
    # ------------------------------------------------------------------

    @property
    def backend_type(self) -> BackendType:
        return BackendType.SQLITEVEC

    @property
    def dialect(self) -> str:
        return Dialect.SQLITE

    @property
    def default_flat_metric(self) -> MetricType:
        return MetricType.COSINE

    # ------------------------------------------------------------------
    # Storage table management
    # ------------------------------------------------------------------

    def _storage_table_exists(self, model_record: EmbeddingModelRecord) -> bool:
        return table_exists(self.emb_engine, model_record.storage_identifier)

    def _legacy_table_error(self, model_record: EmbeddingModelRecord) -> LegacyVectorTableError:
        declared = declared_metric(self.emb_engine, model_record.storage_identifier)
        registered = model_record.metric_type
        return LegacyVectorTableError(
            f"The sqlite-vec table '{model_record.storage_identifier}' for model "
            f"'{model_record.model_name}' declares metric '{declared.value if declared else 'none'}' "
            f"but the registry has '{registered.value if registered else 'none'}'. "
            "Recreate it from the stored vectors with "
            f"`omop-emb maintenance rebuild-index --model {model_record.model_name}` "
            "(add `--metric-type` to choose the metric, cosine by default)."
        )

    def _get_storage_table_descriptor(self, model_record: EmbeddingModelRecord) -> Table:
        declared = declared_metric(self.emb_engine, model_record.storage_identifier)
        if declared is None or declared != model_record.metric_type:
            raise self._legacy_table_error(model_record)
        return sqlite_vec_table_descriptor(
            model_record.storage_identifier, self._sqlite_vec_metadata
        )

    def _create_storage_table(self, model_record: EmbeddingModelRecord) -> Table:
        metric_type = model_record.metric_type
        if metric_type is None:
            raise self._legacy_table_error(model_record)
        ddl = ddl_create_vec0(
            table_name=model_record.storage_identifier,
            dimensions=model_record.dimensions,
            metric_type=metric_type,
        )
        with self.emb_engine.begin() as conn:
            conn.execute(text(ddl))
        return sqlite_vec_table_descriptor(
            model_record.storage_identifier, self._sqlite_vec_metadata
        )

    def _delete_storage_table(self, model_record: EmbeddingModelRecord) -> None:
        with self.emb_engine.begin() as conn:
            conn.execute(text(ddl_drop_vec0(model_record.storage_identifier)))

    # ------------------------------------------------------------------
    # Index management (sqlite-vec supports FLAT only; rebuild recreates a table whose metric differs)
    # ------------------------------------------------------------------

    def _rebuild_index_impl(
        self, *, model_record: EmbeddingModelRecord, index_config: IndexConfig, vacuum: bool
    ) -> None:
        if not isinstance(index_config, FlatIndexConfig):
            raise UnsupportedIndexConfigError(
                f"sqlite-vec only supports FLAT indexes. Got: {type(index_config).__name__}."
            )
        metric_type = index_config.metric_type
        if metric_type is None:
            raise ValueError("sqlite-vec needs a metric_type for the table.")
        if declared_metric(self.emb_engine, model_record.storage_identifier) != metric_type:
            count = recreate_vec0(
                self.emb_engine,
                model_record.storage_identifier,
                model_record.dimensions,
                metric_type,
            )
            self._table_cache.pop(model_record.storage_identifier, None)
            logger.info(
                f"Recreated '{model_record.storage_identifier}' with the {metric_type.value} metric ({count} rows)."
            )
            if vacuum:
                self._vacuum_database()

    def _compact_storage_impl(self, *, model_record: EmbeddingModelRecord) -> CompactionResult:
        """Recreate the table when that frees chunks, then ``VACUUM`` the file if pages are free.

        Deleted rows leave vec0 chunks partly empty and ``VACUUM`` alone does
        not return them, so the table is recreated from its stored vectors.
        Nothing is rewritten when the table has no chunk to free.
        """
        table_name = model_record.storage_identifier
        metric_type = declared_metric(self.emb_engine, table_name)
        if metric_type is None:
            raise self._legacy_table_error(model_record)
        size_before = database_size(self.emb_engine)
        if reclaimable_chunks(self.emb_engine, table_name):
            recreate_vec0(self.emb_engine, table_name, model_record.dimensions, metric_type)
            self._table_cache.pop(table_name, None)
        if freelist_pages(self.emb_engine):
            self._vacuum_database()
        return CompactionResult(size_before, database_size(self.emb_engine))

    def _vacuum_database(self) -> None:
        try:
            vacuum_database(self.emb_engine)
        except OperationalError as exc:
            logger.warning(
                f"VACUUM failed ({exc.orig}), so the database file keeps the freed space. "
                "Run VACUUM on it once enough disk space is available."
            )

    def physical_indexes(self, model_name: str) -> tuple[str, ...]:
        """vec0 is always a flat scan; there is no secondary physical index."""
        return ()

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
        with self.emb_session_factory.begin() as session:
            dml_upsert_rows(
                session=session,
                table=table,
                records=records,
                embeddings=embeddings,
            )

    def _delete_embeddings_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
    ) -> int:
        table = self._storage_table(model_record)
        with self.emb_session_factory.begin() as session:
            return dml_delete_rows(session=session, table=table, concept_ids=concept_ids)

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
                session=session,
                table=table,
                concept_ids=concept_ids,
            )
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
        query_embeddings: EmbeddingArray,
        concept_filter: EmbeddingConceptFilter | None = None,
        k: int = EmbeddingBackend.DEFAULT_K_NEAREST,
    ) -> tuple[tuple[NearestConceptMatch, ...], ...]:
        table = self._storage_table(model_record)
        with self.emb_session_factory() as session:
            batches = query_knn_batch(
                session=session,
                table=table,
                query_vectors=list(query_embeddings),
                k=k,
                concept_filter=concept_filter,
            )

        results = [
            tuple(
                NearestConceptMatch(
                    concept_id=row.concept_id,
                    similarity=get_similarity_from_distance(row.distance, metric_type),
                    domain_id=row.domain_id,
                    vocabulary_id=row.vocabulary_id,
                    is_standard=bool(row.is_standard),
                    is_active=bool(row.is_valid),
                )
                for row in rows
            )
            for rows in batches
        ]
        return tuple(results)

    # ------------------------------------------------------------------
    # Utility queries
    # ------------------------------------------------------------------

    def _has_any_embeddings_impl(self, *, model_record: EmbeddingModelRecord) -> bool:
        table = self._storage_table(model_record)
        with self.emb_session_factory() as session:
            return query_has_any(session=session, table=table)

    def _get_all_stored_concept_ids_impl(self, *, model_record: EmbeddingModelRecord) -> set[int]:
        table = self._storage_table(model_record)
        with self.emb_session_factory() as session:
            return query_all_concept_ids(session=session, table=table)

    def _get_concept_filter_metadata_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
    ) -> Mapping[int, ConceptEmbeddingRecord]:
        if not concept_ids:
            return {}
        table = self._storage_table(model_record)
        with self.emb_session_factory() as session:
            rows = query_metadata_by_ids(session=session, table=table, concept_ids=concept_ids)
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
                table=table,
                concept_filter=concept_filter,
            )

    def _get_embedding_count_by_vocabulary_impl(
        self, *, model_record: EmbeddingModelRecord
    ) -> Mapping[str, int]:
        table = self._storage_table(model_record)
        with self.emb_session_factory() as session:
            return query_embedding_count_by_vocabulary(session=session, table=table)
