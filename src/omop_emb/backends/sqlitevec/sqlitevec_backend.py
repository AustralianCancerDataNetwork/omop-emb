"""sqlite-vec embedding backend (default, no external database required).

All data lives in a single .db file. vec0 virtual tables (embeddings) and
the registry table share the same file and SQLAlchemy engine.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy import ndarray
from sqlalchemy import Engine, MetaData, Table, create_engine, event, text

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
    query_concept_ids_matching_filter,
    query_embeddings_by_ids,
    query_filter_metadata_by_ids,
    query_has_any,
    query_knn,
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


def create_sqlitevec_engine(db_path: str) -> Engine:
    """Create a SQLAlchemy engine for sqlite-vec with the extension pre-loaded.

    Parameters
    ----------
    db_path : str
        File path to the SQLite database, or ``':memory:'`` for an in-memory
        database.

    Returns
    -------
    Engine
    """
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    @event.listens_for(engine, "connect")
    def _load_sqlite_vec(dbapi_conn, _connection_record):
        dbapi_conn.enable_load_extension(True)
        sqlite_vec.load(dbapi_conn)
        dbapi_conn.enable_load_extension(False)

    return engine


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

    def __init__(self, emb_engine: Engine) -> None:
        self._sqlite_vec_metadata = MetaData()
        super().__init__(emb_engine=emb_engine)

    @classmethod
    def from_path(cls, db_path: str) -> "SQLiteVecEmbeddingBackend":
        """Construct a backend from a database file path.

        Parameters
        ----------
        db_path : str
            File path to the SQLite database, or ``':memory:'`` for testing.

        Returns
        -------
        SQLiteVecEmbeddingBackend
        """
        return cls(emb_engine=create_sqlitevec_engine(db_path))

    # ------------------------------------------------------------------
    # Backend identity
    # ------------------------------------------------------------------

    @property
    def backend_type(self) -> BackendType:
        return BackendType.SQLITEVEC

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
            metric_type=model_record.metric_type,
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
        with self.emb_session_factory.begin() as session:
            dml_upsert_rows(
                session=session,
                table=table,
                records=records,
                embeddings=embeddings.astype(np.float32),
                dialect=self.emb_engine.dialect.name,
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
        table = self._table_cache[model_record.storage_identifier]
        with self.emb_session_factory() as session:
            result = query_embeddings_by_ids(
                session=session,
                table=table,
                concept_ids=concept_ids,
                dialect=self.emb_engine.dialect.name,
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

        table = self._table_cache[model_record.storage_identifier]
        results: list[tuple[NearestConceptMatch, ...]] = []
        with self.emb_session_factory() as session:
            for query_vec in query_embeddings:
                rows = query_knn(
                    session=session,
                    table=table,
                    query_vector=query_vec.astype(np.float32),
                    metric_type=metric_type,
                    k=k,
                    concept_filter=concept_filter,
                )
                matches = tuple(
                    NearestConceptMatch(
                        concept_id=concept_id,
                        similarity=get_similarity_from_distance(distance, metric_type),
                        is_standard=bool(is_standard),
                    )
                    for concept_id, distance, is_standard in rows
                )
                results.append(matches)

        return tuple(results)

    # ------------------------------------------------------------------
    # Utility queries
    # ------------------------------------------------------------------

    def _has_any_embeddings_impl(self, *, model_record: EmbeddingModelRecord) -> bool:
        table = self._table_cache[model_record.storage_identifier]
        with self.emb_session_factory() as session:
            return query_has_any(session=session, table=table)

    def _get_all_stored_concept_ids_impl(
        self, *, model_record: EmbeddingModelRecord
    ) -> set[int]:
        table = self._table_cache[model_record.storage_identifier]
        with self.emb_session_factory() as session:
            return query_all_concept_ids(session=session, table=table)

    def _get_concept_filter_metadata_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
    ) -> Mapping[int, Mapping[str, object]]:
        table = self._table_cache[model_record.storage_identifier]
        with self.emb_session_factory() as session:
            return query_filter_metadata_by_ids(
                session=session,
                table=table,
                concept_ids=concept_ids,
                dialect=self.emb_engine.dialect.name,
            )

    def _get_concept_ids_matching_filter_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        concept_filter: EmbeddingConceptFilter,
    ) -> set[int]:
        if concept_filter.is_empty():
            return self._get_all_stored_concept_ids_impl(model_record=model_record)
        table = self._table_cache[model_record.storage_identifier]
        with self.emb_session_factory() as session:
            return query_concept_ids_matching_filter(
                session=session,
                table=table,
                concept_filter=concept_filter,
            )
