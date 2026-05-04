"""SQL helpers for the pgvector storage backend.

Key changes from the old design
--------------------------------
* ``create_pg_embedding_table`` is now aware of
  :class:`~omop_emb.config.VectorColumnType` and automatically uses
  ``halfvec`` for models with more than 2 000 dimensions.
* The ANN query (:func:`q_nearest_concept_ids`) no longer joins to the OMOP
  CDM ``Concept`` table.  It returns ``(q_id, concept_id, distance)`` rows
  from the pgvector instance only.  Concept metadata is fetched separately
  from the CDM in :meth:`~omop_emb.storage.base.EmbeddingBackend._fetch_concept_metadata`.
* ``ConceptIDEmbeddingBase`` no longer carries a foreign-key to ``Concept``
  (cross-database FKs are unsupported by Postgres).
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Type, Union

from numpy import ndarray
from pgvector.sqlalchemy import HalfVector, Vector
from sqlalchemy import Engine, Integer, Select, delete, literal, select
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import cast, column, values
from sqlalchemy.sql.elements import ColumnElement

from orm_loader.helpers import Base

from omop_emb.config import MetricType, VectorColumnType
from omop_emb.storage.base import ConceptIDEmbeddingBase
from omop_emb.storage.index_config import INDEX_CONFIG_METADATA_KEY
from omop_emb.storage.postgres.pg_registry import EmbeddingModelRecord
from omop_emb.utils.embedding_utils import get_similarity_from_distance

logger = logging.getLogger(__name__)

_PGVECTOR_TABLE_CACHE: dict[str, type["PGVectorConceptIDEmbeddingTable"]] = {}
EMBEDDING_COLUMN_NAME = "embedding"


class PGVectorConceptIDEmbeddingTable(ConceptIDEmbeddingBase, Base):
    """Abstract ORM base for pgvector embedding tables."""
    __abstract__ = True
    embedding: Mapped[list[float]]


def _vector_column_for_type(dimensions: int, vector_col_type: VectorColumnType):
    """Return a SQLAlchemy column definition for the given vector column type."""
    if vector_col_type == VectorColumnType.HALFVEC:
        return mapped_column(HalfVector(dimensions), nullable=False, index=False)
    return mapped_column(Vector(dimensions), nullable=False, index=False)


def create_pg_embedding_table(
    engine: Engine,
    model_record: EmbeddingModelRecord,
) -> Type[PGVectorConceptIDEmbeddingTable]:
    """Create (or retrieve cached) ORM class for a model's embedding table.

    Automatically selects ``halfvec`` when ``model_record.dimensions > 2 000``.
    """
    from omop_emb.config import vector_column_type_for_dimensions

    tablename = model_record.storage_identifier
    dimensions = model_record.dimensions

    cached = _PGVECTOR_TABLE_CACHE.get(tablename)
    if cached is not None:
        Base.metadata.create_all(engine, tables=[cached.__table__])  # type: ignore[arg-type]
        return cached

    vector_col_type = vector_column_type_for_dimensions(dimensions)
    embedding_col = _vector_column_for_type(dimensions, vector_col_type)

    table = type(
        f"PGVectorConceptIDEmbeddingTable_{tablename}",
        (PGVectorConceptIDEmbeddingTable,),
        {
            "__tablename__": tablename,
            EMBEDDING_COLUMN_NAME: embedding_col,
            "__table_args__": {"extend_existing": True},
            "__module__": __name__,
        },
    )
    Base.metadata.create_all(engine, tables=[table.__table__])  # type: ignore[arg-type]
    _PGVECTOR_TABLE_CACHE[tablename] = table

    logger.debug(
        f"Created embedding table '{tablename}' "
        f"(dim={dimensions}, col_type={vector_col_type.value}, index={model_record.index_type.value})"
    )
    return table


def delete_pg_embedding_table(
    engine: Engine,
    model_record: EmbeddingModelRecord,
) -> None:
    """Drop a model's embedding table from the database and evict it from the cache."""
    tablename = model_record.storage_identifier
    embedding_table = _PGVECTOR_TABLE_CACHE.pop(tablename, None)
    if embedding_table is not None:
        with engine.begin() as conn:
            conn.execute(delete(embedding_table))
        logger.info(
            f"Dropped embedding table '{tablename}' for model '{model_record.model_name}'."
        )


# ---------------------------------------------------------------------------
# DML helpers
# ---------------------------------------------------------------------------

def q_add_embeddings_to_registered_table(
    concept_ids: tuple[int, ...],
    embeddings: ndarray,
    registered_table: Type[PGVectorConceptIDEmbeddingTable],
):
    """Build an INSERT statement for embedding rows."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    insert_values = [
        {
            registered_table.concept_id.key: cid,
            registered_table.embedding.key: emb.tolist(),
        }
        for cid, emb in zip(concept_ids, embeddings)
    ]
    return pg_insert(registered_table).values(insert_values)


def q_embedding_vectors_by_concept_ids(
    embedding_table: Type[PGVectorConceptIDEmbeddingTable],
    concept_ids: Tuple[int, ...],
) -> Select:
    return (
        select(embedding_table.concept_id, embedding_table.embedding)
        .where(embedding_table.concept_id.in_(concept_ids))
    )


# ---------------------------------------------------------------------------
# ANN query — returns (q_id, concept_id, distance) from pgvector only
# ---------------------------------------------------------------------------

def q_nearest_concept_ids(
    embedding_table: Type[PGVectorConceptIDEmbeddingTable],
    query_embeddings: List[List[float]],
    metric_type: MetricType,
    k: int,
    candidate_concept_ids: Optional[Tuple[int, ...]] = None,
) -> Select:
    """Build a pgvector ANN query that returns the nearest concept IDs per query.

    The result does **not** include concept metadata (name, domain, etc.).
    Metadata is fetched separately from the CDM in
    :meth:`~omop_emb.storage.base.EmbeddingBackend._fetch_concept_metadata`.

    Parameters
    ----------
    embedding_table
        ORM class for the model's embedding table.
    query_embeddings
        List of query vectors, each of length D.
    metric_type
        Distance metric for ranking.
    k
        Number of nearest neighbors to return per query.
    candidate_concept_ids
        When provided, restricts the search to these concept IDs (pre-filtered
        from CDM).  ``None`` means search all stored concepts.

    Returns
    -------
    Select
        Columns: ``q_id`` (int), ``concept_id`` (int), ``distance`` (float).
    """
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
            distance.label("distance"),
        )
        .order_by(distance)
        .limit(k)
    )

    if candidate_concept_ids is not None:
        inner_stmt = inner_stmt.where(
            embedding_table.concept_id.in_(candidate_concept_ids)
        )

    lateral_subq = inner_stmt.lateral("top_k")

    return (
        select(
            query_v.c.q_id,
            lateral_subq.c.concept_id,
            lateral_subq.c.distance,
        )
        .select_from(query_v)
        .join(lateral_subq, literal(True))
    )


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def get_distance(
    embedding_table: Type[PGVectorConceptIDEmbeddingTable],
    text_embedding: Union[list[float], ColumnElement],
    metric: MetricType,
) -> ColumnElement:
    """Return the pgvector distance expression for *metric*."""
    if metric == MetricType.COSINE:
        return embedding_table.embedding.cosine_distance(text_embedding)
    elif metric == MetricType.L2:
        return embedding_table.embedding.l2_distance(text_embedding)
    elif metric == MetricType.L1:
        return embedding_table.embedding.l1_distance(text_embedding)
    elif metric == MetricType.HAMMING:
        return embedding_table.embedding.hamming_distance(text_embedding)
    elif metric == MetricType.JACCARD:
        return embedding_table.embedding.jaccard_distance(text_embedding)
    else:
        raise ValueError(f"Unsupported metric: {metric.value}")
