"""sqlite-vec embedding backend (default, no external database required).

All data lives in a single .db file. vec0 virtual tables (embeddings) and
the registry table share the same file and SQLAlchemy engine.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy import ndarray
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import sessionmaker

try:
    import sqlite_vec
except ImportError as _e:
    raise ImportError(
        "sqlite-vec is not installed. Install it with: pip install sqlite-vec"
    ) from _e

from omop_emb.config import BackendType, MetricType, ProviderType
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
    query_embeddings_by_ids,
    query_has_any,
    query_knn,
)
from omop_emb.model_registry import EmbeddingModelRecord
from omop_emb.utils.embedding_utils import (
    EmbeddingConceptFilter,
    NearestConceptMatch,
    get_similarity_from_distance,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SQLiteVecTableSpec:
    """Descriptor for a sqlite-vec vec0 virtual table.

    Attributes
    ----------
    table_name : str
        Physical table name (the registry ``storage_identifier``).
    dimensions : int
        Embedding vector length.
    metric_type : MetricType, optional
        Metric baked into the ``distance_metric=`` DDL column suffix, or
        ``None`` for FLAT tables (per-query metric via ``vec_distance_*``).
    """

    table_name: str
    dimensions: int
    metric_type: Optional[MetricType]


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


class SQLiteVecBackend(EmbeddingBackend):
    """sqlite-vec embedding backend.

    All data lives in a single .db file. vec0 virtual tables (embeddings) and
    the registry table share the same file and SQLAlchemy engine.

    Notes
    -----
    Only ``FLAT`` index type is supported. Supports ``L2`` and ``COSINE``
    distance metrics.
    """

    @classmethod
    def from_path(cls, db_path: str) -> "SQLiteVecBackend":
        """Construct a backend from a database file path.

        Parameters
        ----------
        db_path : str
            File path to the SQLite database, or ``':memory:'`` for testing.

        Returns
        -------
        SQLiteVecBackend
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

    def _create_storage_table(self, model_record: EmbeddingModelRecord) -> SQLiteVecTableSpec:
        ddl = ddl_create_vec0(
            table_name=model_record.storage_identifier,
            dimensions=model_record.dimensions,
            metric_type=model_record.metric_type,
        )
        with self.emb_engine.begin() as conn:
            conn.execute(text(ddl))
        return SQLiteVecTableSpec(
            table_name=model_record.storage_identifier,
            dimensions=model_record.dimensions,
            metric_type=model_record.metric_type,
        )

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
        # vec0 is always a flat scan — no DDL needed.

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
        with self.emb_session_factory.begin() as session:
            dml_upsert_rows(
                session=session,
                table_name=model_record.storage_identifier,
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
        with self.emb_session_factory() as session:
            result = query_embeddings_by_ids(
                session=session,
                table_name=model_record.storage_identifier,
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

        results: list[tuple[NearestConceptMatch, ...]] = []
        with self.emb_session_factory() as session:
            for query_vec in query_embeddings:
                rows = query_knn(
                    session=session,
                    table_name=model_record.storage_identifier,
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
        with self.emb_session_factory() as session:
            return query_has_any(session=session, table_name=model_record.storage_identifier)

    def _get_all_stored_concept_ids_impl(self, *, model_record: EmbeddingModelRecord) -> set[int]:
        with self.emb_session_factory() as session:
            return query_all_concept_ids(session=session, table_name=model_record.storage_identifier)
