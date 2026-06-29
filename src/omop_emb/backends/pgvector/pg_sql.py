"""SQL helpers for the pgvector backend.

The embedding table carries filter columns alongside the vector:
  - ``domain_id``     TEXT    (OMOP domain)
  - ``vocabulary_id`` TEXT    (OMOP vocabulary)
  - ``is_standard``   BOOLEAN (standard_concept in ('S','C') → True)

These are populated at upsert time by the caller and enable efficient
pre-filtering during KNN without re-querying the OMOP CDM.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Union

from numpy import ndarray
from sqlalchemy import Engine, Integer, Select, inspect as sa_inspect, literal, select, text, TextClause
from sqlalchemy.sql import cast, column, values
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.orm import mapped_column

from omop_emb.config import MetricType
from omop_emb.backends.base_backend import ConceptEmbeddingRecord
from omop_emb.backends.db_utils import apply_concept_filter_where
from omop_emb.backends.embedding_table import EMBEDDING_COLUMN_NAME, EmbeddingTableBase, PGEmbeddingTable
from omop_emb.model_registry import EmbeddingModelRecord
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter

logger = logging.getLogger(__name__)


def table_exists(engine: Engine, table_name: str) -> bool:
    """Return ``True`` if *table_name* exists in the current Postgres schema."""
    return sa_inspect(engine).has_table(table_name)

def create_pg_embedding_table(
    engine: Engine, model_record: EmbeddingModelRecord
) -> type[PGEmbeddingTable]:
    """Create a pgvector embedding table and return its ORM class.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine for the pgvector database.
    model_record : EmbeddingModelRecord

    Returns
    -------
    type[PGEmbeddingTable]
        SQLAlchemy ORM class mapped to the newly created table.

    Notes
    -----
    Uses ``halfvec(N)`` for dimensions greater than 2 000 and ``vector(N)``
    otherwise. Caching is handled by ``_ensure_storage_table`` in the backend
    base class; this function always issues DDL.
    """
    table_cls = pg_embedding_table_descriptor(model_record)
    EmbeddingTableBase.metadata.create_all(engine, tables=[table_cls.__table__])  # type: ignore[arg-type]
    return table_cls


def drop_pg_embedding_table(engine: Engine, model_record: EmbeddingModelRecord) -> None:
    """Drop the physical embedding table from the database.

    Parameters
    ----------
    engine : Engine
    model_record : EmbeddingModelRecord
    """
    tablename = model_record.storage_identifier
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{tablename}"'))
    logger.info(f"Dropped embedding table '{tablename}'.")


# ---------------------------------------------------------------------------
# DML helpers
# ---------------------------------------------------------------------------


def q_upsert_embeddings(
    records: Sequence[ConceptEmbeddingRecord],
    embeddings: ndarray,
    registered_table: type[PGEmbeddingTable],
):
    """Build an INSERT ... ON CONFLICT DO UPDATE statement for embedding rows.

    Parameters
    ----------
    records : Sequence[ConceptEmbeddingRecord]
        Concept metadata rows, one per embedding.
    embeddings : ndarray
        Float32 array of shape ``(N, D)``.
    registered_table : type[PGEmbeddingTable]
        ORM class for the target embedding table.

    Returns
    -------
    Insert
        PostgreSQL upsert statement ready for execution.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    insert_values = [
        {
            "concept_id": rec.concept_id,
            "domain_id": rec.domain_id,
            "vocabulary_id": rec.vocabulary_id,
            "is_standard": rec.is_standard,
            "is_valid": rec.is_valid,
            EMBEDDING_COLUMN_NAME: emb.tolist(),
        }
        for rec, emb in zip(records, embeddings)
    ]
    stmt = pg_insert(registered_table).values(insert_values)
    return stmt.on_conflict_do_update(
        index_elements=["concept_id"],
        set_={
            "domain_id": stmt.excluded.domain_id,
            "vocabulary_id": stmt.excluded.vocabulary_id,
            "is_standard": stmt.excluded.is_standard,
            "is_valid": stmt.excluded.is_valid,
            EMBEDDING_COLUMN_NAME: getattr(stmt.excluded, EMBEDDING_COLUMN_NAME),
        },
    )


def q_all_concept_ids(embedding_table: type[PGEmbeddingTable]) -> Select:
    """Build a SELECT for all concept IDs in an embedding table.

    Parameters
    ----------
    embedding_table : type[PGEmbeddingTable]

    Returns
    -------
    Select
    """
    return select(embedding_table.concept_id)


def q_create_extension_pgvector() -> TextClause:
    """Return a SQL statement to create the pgvector extension if it doesn't exist."""
    return text("CREATE EXTENSION IF NOT EXISTS vector CASCADE;")


# ---------------------------------------------------------------------------
# ANN query
# ---------------------------------------------------------------------------


def q_nearest_concept_ids(
    embedding_table: type[PGEmbeddingTable],
    query_embeddings: List[List[float]],
    metric_type: MetricType,
    k: int,
    concept_filter: Optional[EmbeddingConceptFilter] = None,
) -> Select:
    """Build a pgvector ANN query returning the nearest concept IDs per query.

    Parameters
    ----------
    embedding_table : type[PGEmbeddingTable]
        ORM class for the embedding table.
    query_embeddings : list[list[float]]
        List of ``Q`` query vectors, each of length ``D``.
    metric_type : MetricType
    k : int
        Maximum number of results per query.
    concept_filter : EmbeddingConceptFilter, optional

    Returns
    -------
    Select
        Columns: ``q_id`` (int), ``concept_id`` (int), ``is_standard`` (bool), ``distance`` (float).
        Result shape is ``(Q*K, 4)`` before the caller re-groups by ``q_id``.

    Notes
    -----
    Uses a lateral join so all queries are batched in a single round-trip.
    """
    from pgvector.sqlalchemy import Vector  # optional dependency

    query_data = [(i, q) for i, q in enumerate(query_embeddings)]
    query_v = values(
        column("q_id", Integer),
        column("q_vec", Vector),
        name="queries",
    ).data(query_data)

    query_vector_cast = cast(query_v.c.q_vec, Vector)
    distance = get_distance(embedding_table, query_vector_cast, metric_type)

    inner_stmt = (
        select(
            embedding_table.concept_id,
            embedding_table.is_standard,
            distance.label("distance"),
        )
        .order_by(distance)
        .limit(k)
    )

    if concept_filter is not None:
        inner_stmt = apply_concept_filter_where(
            inner_stmt, sa_inspect(embedding_table).columns, concept_filter
        )
        if concept_filter.limit is not None:
            inner_stmt = inner_stmt.limit(concept_filter.limit)

    lateral_subq = inner_stmt.lateral("top_k")

    return (
        select(
            query_v.c.q_id,
            lateral_subq.c.concept_id,
            lateral_subq.c.is_standard,
            lateral_subq.c.distance,
        )
        .select_from(query_v)
        .join(lateral_subq, literal(True))
    )


def q_concept_ids_matching_filter(
    embedding_table: type[PGEmbeddingTable], concept_filter: EmbeddingConceptFilter
) -> Select:
    """Build a query returning every ``concept_id`` satisfying *concept_filter*.

    Notes
    -----
    Caller must have already called :func:`~omop_emb.backends.db_utils.setup_concept_filter_temps`
    in the same transaction.
    """
    stmt = select(embedding_table.concept_id)
    return apply_concept_filter_where(stmt, sa_inspect(embedding_table).columns, concept_filter)


def q_concept_filter_metadata(
    embedding_table: type[PGEmbeddingTable], concept_filter: EmbeddingConceptFilter
) -> Select:
    """Build a query returning filter metadata columns for every concept ID
    satisfying concept_filter.

    Notes
    -----
    Caller must have already called :func:`~omop_emb.backends.db_utils.setup_concept_filter_temps`
    in the same transaction.
    """
    stmt = select(
        embedding_table.concept_id,
        embedding_table.domain_id,
        embedding_table.vocabulary_id,
        embedding_table.is_standard,
        embedding_table.is_valid,
    )
    return apply_concept_filter_where(stmt, sa_inspect(embedding_table).columns, concept_filter)


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------


def get_distance(
    embedding_table: type[PGEmbeddingTable],
    text_embedding: Union[list[float], ColumnElement],
    metric: MetricType,
) -> ColumnElement:
    """Return a SQLAlchemy distance expression for the given metric.

    Parameters
    ----------
    embedding_table : type[PGEmbeddingTable]
        ORM class whose ``embedding`` column is used. The column itself
        isn't declared on ``PGEmbeddingTable`` (its type depends on
        dimensionality), hence ``getattr``.
    text_embedding : list[float] or ColumnElement
        Query vector.
    metric : MetricType

    Returns
    -------
    ColumnElement
        Distance column expression.

    Raises
    ------
    ValueError
        If ``metric`` is not supported by pgvector.
    """
    embedding_col = getattr(embedding_table, EMBEDDING_COLUMN_NAME)
    if metric == MetricType.COSINE:
        return embedding_col.cosine_distance(text_embedding)
    elif metric == MetricType.L2:
        return embedding_col.l2_distance(text_embedding)
    elif metric == MetricType.L1:
        return embedding_col.l1_distance(text_embedding)
    elif metric == MetricType.HAMMING:
        raise ValueError(
            "HAMMING distance requires a 'bit' column type which is not currently "
            "supported. Use L2, COSINE, L1, or JACCARD."
        )
    elif metric == MetricType.JACCARD:
        raise ValueError(
            "JACCARD distance requires a 'bit' column type which is not currently "
            "supported. Use L2, COSINE, or L1."
        )
    else:
        raise ValueError(f"Unsupported metric: {metric.value}")


def pg_embedding_table_descriptor(model_record: EmbeddingModelRecord) -> type[PGEmbeddingTable]:
    """Return the SQLAlchemy ORM class descriptor for a pgvector embedding table."""
    from omop_emb.utils.embedding_utils import (
        VectorColumnType,
        vector_column_type_for_dimensions,
    )
    from pgvector.sqlalchemy import VECTOR, HALFVEC  # optional dependency

    tablename = model_record.storage_identifier
    dimensions = model_record.dimensions
    col_type = vector_column_type_for_dimensions(dimensions)
    emb_col = mapped_column(
        HALFVEC(dimensions)
        if col_type == VectorColumnType.HALFVEC
        else VECTOR(dimensions),
        nullable=False,
        index=False,
    )
    return type(
        f"PGEmbedding_{tablename}",
        (PGEmbeddingTable,),
        {
            "__tablename__": tablename,
            "__table_args__": {"extend_existing": True},
            "__module__": __name__,
            EMBEDDING_COLUMN_NAME: emb_col,
        },
    )