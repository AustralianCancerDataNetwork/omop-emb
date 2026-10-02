"""sqlite-vec embedding backend (default, no external database required).

All data lives in a single .db file. vec0 virtual tables (embeddings) and
the registry table share the same file and SQLAlchemy engine.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy import ndarray
from oa_configurator import Dialect, ResolvedDatabase
from sqlalchemy import Engine, MetaData, Table, text

try:
    import sqlite_vec
except ImportError as _e:
    raise ImportError(
        "sqlite-vec is not installed. Install it with: pip install sqlite-vec"
    ) from _e

from omop_emb.config import BackendType, MetricType
from omop_emb.backends.base_backend import (
    ConceptEmbeddingRecord,
    EmbeddingBackend,
)
from omop_emb.backends.index_config import FlatIndexConfig, IndexConfig
from omop_emb.backends.sqlitevec.sqlitevec_sql import (
    ddl_create_vec0,
    ddl_drop_vec0,
    dml_upsert_rows,
    query_all_concept_ids,
    query_concept_filter_metadata,
    query_concept_ids_matching_filter,
    query_embedding_count_by_vocabulary,
    query_embeddings_by_ids,
    query_has_any,
    query_knn_batch,
    table_exists,
    sqlite_vec_table_descriptor,
)
from omop_emb.model_registry import EmbeddingModelRecord
from omop_emb.utils.embedding_utils import (
    EmbeddingConceptFilter,
    NearestConceptMatch,
    get_similarity_from_distance,
)

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
    - Only ``FLAT`` index type is supported. Supports ``L2``, ``COSINE``, and
    ``L1`` distance metrics.
    - vec0 is a SQLite virtual table type the SQLAlchemy ORM cannot map, but
      it accepts ordinary SQL once created. ``_table_cache`` therefore stores
      a Core :class:`~sqlalchemy.Table` used for every query except the initial 
      ``CREATE VIRTUAL TABLE``.
    """

    def __init__(
        self,
        emb_engine: Engine,
        *,
        resolved: ResolvedDatabase | None = None,
        writable: bool = True,
    ) -> None:
        self._sqlite_vec_metadata = MetaData()
        super().__init__(emb_engine=emb_engine, resolved=resolved, writable=writable)

    # ------------------------------------------------------------------
    # Backend identity
    # ------------------------------------------------------------------

    @property
    def backend_type(self) -> BackendType:
        return BackendType.SQLITEVEC

    @property
    def dialect(self) -> str:
        return Dialect.SQLITE

    # ------------------------------------------------------------------
    # Storage table management
    # ------------------------------------------------------------------

    def _storage_table_exists(self, model_record: EmbeddingModelRecord) -> bool:
        return table_exists(self.emb_engine, model_record.storage_identifier)

    def _get_storage_table_descriptor(self, model_record: EmbeddingModelRecord) -> Table:
        return sqlite_vec_table_descriptor(model_record.storage_identifier, self._sqlite_vec_metadata)

    def _create_storage_table(self, model_record: EmbeddingModelRecord) -> Table:
        ddl = ddl_create_vec0(
            table_name=model_record.storage_identifier,
            dimensions=model_record.dimensions,
        )
        with self.emb_engine.begin() as conn:
            conn.execute(text(ddl))
        return sqlite_vec_table_descriptor(model_record.storage_identifier, self._sqlite_vec_metadata)

    def _delete_storage_table(self, model_record: EmbeddingModelRecord) -> None:
        with self.emb_engine.begin() as conn:
            conn.execute(text(ddl_drop_vec0(model_record.storage_identifier)))

    # ------------------------------------------------------------------
    # Index management (sqlite-vec supports FLAT only)
    # ------------------------------------------------------------------

    def _rebuild_index_impl(
        self, *, model_record: EmbeddingModelRecord, index_config: IndexConfig
    ) -> None:
        if not isinstance(index_config, FlatIndexConfig):
            raise ValueError(
                f"sqlite-vec only supports FLAT indexes. Got: {type(index_config).__name__}."
            )
        # vec0 is always a flat scan -> no DDL needed.

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
        embeddings: ndarray,
    ) -> None:
        self.validate_embeddings_and_records(
            embeddings=embeddings,
            records=records,
            dimensions=model_record.dimensions,
        )
        table = self._storage_table(model_record)
        with self.emb_session_factory.begin() as session:
            dml_upsert_rows(
                session=session,
                table=table,
                records=records,
                embeddings=embeddings.astype(np.float32),
            )

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
        query_embeddings: ndarray,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        k: int = EmbeddingBackend.DEFAULT_K_NEAREST,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        self.validate_embeddings(query_embeddings, model_record.dimensions)

        table = self._storage_table(model_record)
        with self.emb_session_factory() as session:
            batches = query_knn_batch(
                session=session,
                table=table,
                query_vectors=[v.astype(np.float32) for v in query_embeddings],
                metric_type=metric_type,
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

    def _get_all_stored_concept_ids_impl(
        self, *, model_record: EmbeddingModelRecord
    ) -> set[int]:
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
        concept_filter = EmbeddingConceptFilter(concept_ids=tuple(concept_ids))
        with self.emb_session_factory() as session:
            rows = query_concept_filter_metadata(
                session=session,
                table=table,
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
                table=table,
                concept_filter=concept_filter,
            )

    def _get_embedding_count_by_vocabulary_impl(
        self, *, model_record: EmbeddingModelRecord
    ) -> Mapping[str, int]:
        table = self._storage_table(model_record)
        with self.emb_session_factory() as session:
            return query_embedding_count_by_vocabulary(session=session, table=table)
