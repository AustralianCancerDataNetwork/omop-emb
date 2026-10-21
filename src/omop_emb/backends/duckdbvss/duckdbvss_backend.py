"""DuckDB VSS embedding backend (optional).

Requires: ``pip install omop-emb[duckdbvss]``

All data lives in a single DuckDB file (or :memory:). The embedding column
is a fixed-size ``FLOAT[N]`` array, and the HNSW index is built by DuckDB's
``vss`` extension, loaded on every connection via ``_install_vss``.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import duckdb
from numpy import ndarray
from oa_configurator import (
    Dialect,
    ResolvedDatabase,
    Role,
    guard_schema_provenance_for,
    physical_schema_of,
    qualified,
)
from sqlalchemy import inspect, select, text
from sqlalchemy.engine import make_url

from omop_emb.backends.base_backend import (
    CompactionResult,
    ConceptEmbeddingRecord,
    EmbeddingBackend,
)
from omop_emb.backends.duckdbvss.duckdbvss_sql import (
    create_duckdbvss_embedding_table,
    database_block_bytes,
    delete_embedding_rows,
    drop_duckdbvss_embedding_table,
    duckdbvss_embedding_table_descriptor,
    hnsw_index_ddl,
    q_all_concept_ids,
    q_embedding_count_by_vocabulary,
    query_concept_filter_metadata,
    query_concept_ids_matching_filter,
    query_embeddings_by_ids,
    query_knn_batch,
    table_exists,
    upsert_embedding_rows,
)
from omop_emb.backends.embedding_table import DuckDBVSSEmbeddingTable
from omop_emb.backends.index_config import HNSWIndexConfig, IndexConfig
from omop_emb.config import BackendType, MetricType
from omop_emb.model_registry import EmbeddingModelRecord
from omop_emb.utils.embedding_utils import (
    EmbeddingArray,
    EmbeddingConceptFilter,
    NearestConceptMatch,
    get_similarity_from_distance,
)

logger = logging.getLogger(__name__)


def _install_vss(dbapi_connection, _connection_record) -> None:
    """Connect-event callable: load the VSS extension and enable its
    experimental persistence flag on this connection.

    Passed as an ``extensions`` callable to ``ResolvedDatabase.create_engine()``
    (see its docstring). Loading an extension is inherently per-connection,
    so this is safe to run on every new connection. 
    
    Notes
    -----
    - Tries a bare ``LOAD`` first and only falls back to ``INSTALL`` 
        - ``INSTALL`` needs network access the first time but is cached
        locally for subsequent connections.
    """
    cursor = dbapi_connection.cursor()
    try:
        try:
            cursor.execute("LOAD vss;")
        except duckdb.Error:
            cursor.execute("INSTALL vss; LOAD vss;")
        cursor.execute("SET hnsw_enable_experimental_persistence = true;")
        dbapi_connection.commit()
    finally:
        cursor.close()


def _reader_database_for_missing_file(database: ResolvedDatabase) -> ResolvedDatabase:
    """database, or an ephemeral in-memory substitute if its DuckDB file doesn't exist yet.

    ``read_only=True`` can't open a missing file, but a read-only reader
    must never create one either. An ephemeral in-memory database reports
    "nothing registered yet" through every existing backend method on its
    own, so no separate stub reader type is needed.
    """
    path = make_url(database.connection.url).database
    if path is None or path == ":memory:" or Path(path).exists():
        return database
    memory_url = "duckdb:///:memory:"
    connection = dataclasses.replace(
        database.connection, url=memory_url, safe_url=memory_url, _engine_url=make_url(memory_url)
    )
    return dataclasses.replace(database, connection=connection)


class DuckDBVSSEmbeddingBackend(EmbeddingBackend[type[DuckDBVSSEmbeddingTable]]):
    """DuckDB VSS-backed embedding backend.

    Supports ``FLAT`` and ``HNSW`` index types with ``L2`` and ``COSINE``
    distance metrics. DuckDB VSS has no L1 array function, unlike pgvector.

    Notes
    -----
    KNN queries run one statement per query vector: DuckDB has no
    lateral-join equivalent of pgvector's multi-vector batching.

    A filtered KNN query against an HNSW-indexed table always exact-scans
    the filtered rows instead of using the index (see
    ``duckdbvss_sql._build_filtered_exact_scan_stmt``): DuckDB's built-in
    ``vss`` extension returns too few rows, often zero, on filtered queries
    regardless of ``ef_search``, with no tunable fix.

    Compaction only runs ``CHECKPOINT``: DuckDB has no table-scoped
    reclaim, and the only thing that shrinks the file (``COPY FROM
    DATABASE`` to a fresh one) rewrites the whole physical file, which can
    host more than one schema's data here, so it's out of scope for
    compacting a single model's table.
    """

    @property
    def backend_type(self) -> BackendType:
        return BackendType.DUCKDBVSS

    @property
    def dialect(self) -> str:
        return Dialect.DUCKDB

    @property
    def default_flat_metric(self) -> None:
        """``None``: FLAT queries choose their metric per query, as on pgvector."""
        return None

    # ------------------------------------------------------------------
    # Storage table management
    # ------------------------------------------------------------------

    def _storage_table_exists(self, model_record: EmbeddingModelRecord) -> bool:
        return table_exists(self.emb_engine, model_record.storage_identifier)

    def _get_storage_table_descriptor(
        self, model_record: EmbeddingModelRecord
    ) -> type[DuckDBVSSEmbeddingTable]:
        return duckdbvss_embedding_table_descriptor(model_record=model_record)

    def _create_storage_table(
        self, model_record: EmbeddingModelRecord
    ) -> type[DuckDBVSSEmbeddingTable]:
        return create_duckdbvss_embedding_table(engine=self.emb_engine, model_record=model_record)

    def _delete_storage_table(self, model_record: EmbeddingModelRecord) -> None:
        drop_duckdbvss_embedding_table(engine=self.emb_engine, model_record=model_record)

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _rebuild_index_impl(
        self, *, model_record: EmbeddingModelRecord, index_config: IndexConfig, vacuum: bool
    ) -> None:
        """Drop every physical index on the table, then create the HNSW index index_config describes.

        With ``vacuum``, ``ANALYZE`` runs on the table afterwards so the
        planner sees current statistics. It does not reclaim disk space,
        see ``compact_storage``.
        """
        table_ref = qualified(
            self.emb_engine, model_record.storage_identifier, physical_schema=physical_schema_of(self.emb_engine)
        )
        with self.emb_engine.begin() as conn:
            with guard_schema_provenance_for(conn, schema_tag=Role.PRIMARY):
                for statement in self.drop_index_sql(model_record.model_name):
                    conn.execute(text(statement))
                if isinstance(index_config, HNSWIndexConfig):
                    conn.execute(text(hnsw_index_ddl(self.emb_engine, model_record, index_config)))
                if vacuum:
                    conn.execute(text(f"ANALYZE {table_ref}"))
        logger.info(
            f"Rebuilt '{model_record.storage_identifier}' with index type '{index_config.index_type.value}'."
        )

    def _compact_storage_impl(self, *, model_record: EmbeddingModelRecord) -> CompactionResult:
        """Checkpoint the database; this is the only reclaim DuckDB offers here.

        DuckDB only merges a row group's blocks once roughly a quarter of
        its rows are deleted, and never shrinks the file itself. Database copy
        would allow shrinking but writes a new file including other schemas
        and is therefore out of scope.

        A database's very first checkpoint ever can read as a jump from 0 in
        the returned sizes, since blocks aren't accounted for until something
        checkpoints them.
        """
        size_before = database_block_bytes(self.emb_engine)
        with self.emb_engine.begin() as conn:
            conn.execute(text("CHECKPOINT"))
        return CompactionResult(size_before, database_block_bytes(self.emb_engine))

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
                record.storage_identifier, schema=physical_schema,
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
            logger.error(
                "Failed to upsert embeddings for '%s': %s", model_record.model_name, exc
            )
            raise

    def _delete_embeddings_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
    ) -> int:
        table = self._storage_table(model_record)
        with self.emb_session_factory.begin() as session:
            return delete_embedding_rows(session=session, embedding_table=table, concept_ids=concept_ids)

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
            result = query_embeddings_by_ids(session=session, embedding_table=table, concept_ids=concept_ids)
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
            if isinstance(model_record.index_config, HNSWIndexConfig):
                session.execute(text(f"SET hnsw_ef_search = {int(model_record.index_config.ef_search)}"))

            batches = query_knn_batch(
                session=session,
                embedding_table=table,
                query_vectors=list(query_embeddings),
                metric_type=metric_type,
                k=k,
                dimensions=model_record.dimensions,
                concept_filter=concept_filter,
                exact_scan_if_filtered=isinstance(model_record.index_config, HNSWIndexConfig),
            )

        results = [
            tuple(
                NearestConceptMatch(
                    concept_id=int(row.concept_id),
                    similarity=float(get_similarity_from_distance(float(row.distance), metric_type)),
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
            return session.execute(select(table.concept_id).limit(1)).first() is not None

    def _get_all_stored_concept_ids_impl(
        self, *, model_record: EmbeddingModelRecord
    ) -> set[int]:
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
