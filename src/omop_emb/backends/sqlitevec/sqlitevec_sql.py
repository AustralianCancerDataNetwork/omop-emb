"""SQL helpers for the sqlite-vec backend."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
from numpy import ndarray
from sqlalchemy import (
    Column, 
    Engine, 
    Integer, 
    MetaData, 
    Table, 
    bindparam, 
    delete, 
    func, 
    insert, 
    select, 
    text,
    LargeBinary
)
from sqlalchemy.orm import Session

from omop_emb.backends.base_backend import ConceptEmbeddingRecord
from omop_emb.backends.db_utils import (
    apply_concept_filter_where,
    setup_concept_filter_temps,
    temp_filter_table,
)
from omop_emb.backends.embedding_table import CONCEPT_METADATA_COLUMNS, EMBEDDING_COLUMN_NAME
from omop_emb.config import MetricType
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter

logger = logging.getLogger(__name__)

# Used by ddl_create_vec0 when baking a metric into the column definition.
_MATCH_METRIC_MAP = {
    MetricType.L2: "l2",
    MetricType.COSINE: "cosine",
}

# Used by query_knn for per-query metric selection via ORDER BY.
_QUERY_METRIC_FUNC = {
    MetricType.L2: "vec_distance_l2",
    MetricType.COSINE: "vec_distance_cosine",
    MetricType.L1: "vec_distance_l1",
}


def table_exists(engine: Engine, table_name: str) -> bool:
    """Return ``True`` if a table (or virtual table) named *table_name* exists.

    Queries ``sqlite_master`` which lists all objects (real tables, virtual
    tables, indexes, views) regardless of type.
    """
    with engine.connect() as conn:
        return (
            conn.execute(
                text("SELECT 1 FROM sqlite_master WHERE name = :n"),
                {"n": table_name},
            ).first()
            is not None
        )

def sqlite_vec_table_descriptor(table_name: str, metadata: MetaData) -> Table:
    """Return the Core ``Table`` descriptor for an existing sqlite-vec ``vec0`` table.

    Parameters
    ----------
    table_name : str
    metadata : MetaData

    Returns
    -------
    Table
        Usable for ``select``/``insert``/``delete`` against the virtual
        table. Does not issue DDL -- the ``vec0`` table must already exist.
    """
    columns = [
        Column(c.name, c.type_, primary_key=(c.name == "concept_id"))
        for c in CONCEPT_METADATA_COLUMNS
    ]
    columns.append(Column(EMBEDDING_COLUMN_NAME, LargeBinary))
    return Table(table_name, metadata, *columns, extend_existing=True)


def ddl_create_vec0(
    table_name: str,
    dimensions: int,
    metric_type: Optional[MetricType] = None,
) -> str:
    """Return DDL to create a vec0 virtual table for the given model.

    Parameters
    ----------
    table_name : str
        Physical table name (the registry ``storage_identifier``).
    dimensions : int
        Embedding vector length.
    metric_type : MetricType, optional
        When provided, bakes ``distance_metric=<metric>`` into the embedding
        column definition. This is intended for future ANN index types (e.g.
        sqlite-vec IVF/HNSW) that use the baked-in metric during index-
        accelerated scans. For FLAT tables pass ``None`` (default) and supply
        the metric per-query via ``vec_distance_*`` functions.

    Returns
    -------
    str
        ``CREATE VIRTUAL TABLE IF NOT EXISTS ...`` statement.

    Raises
    ------
    ValueError
        If ``metric_type`` is provided but not supported by the
        ``distance_metric=`` DDL syntax (only L2 and COSINE are accepted).
    """
    if metric_type is not None:
        metric_str = _MATCH_METRIC_MAP.get(metric_type)
        if metric_str is None:
            raise ValueError(
                f"sqlite-vec does not support baked-in metric '{metric_type.value}'. "
                f"Supported values for distance_metric=: {[m.value for m in _MATCH_METRIC_MAP]}"
            )
        embedding_col = f"{EMBEDDING_COLUMN_NAME} FLOAT[{dimensions}] distance_metric={metric_str}"
    else:
        embedding_col = f"{EMBEDDING_COLUMN_NAME} FLOAT[{dimensions}]"

    metadata_cols = ", ".join(f"{c.name} {c.vec0_ddl}" for c in CONCEPT_METADATA_COLUMNS)

    return (
        f'CREATE VIRTUAL TABLE IF NOT EXISTS "{table_name}" USING vec0('
        f"{metadata_cols}, {embedding_col})"
    )


def ddl_drop_vec0(table_name: str) -> str:
    """Return DDL to drop a vec0 virtual table.

    Parameters
    ----------
    table_name : str

    Returns
    -------
    str
        ``DROP TABLE IF EXISTS ...`` statement.
    """
    return f'DROP TABLE IF EXISTS "{table_name}"'


def dml_upsert_rows(
    session: Session,
    table: Table,
    records: Sequence[ConceptEmbeddingRecord],
    embeddings: ndarray,
    dialect: str = "sqlite",
) -> None:
    """Upsert embedding rows into a vec0 table.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session (must be in a transaction).
    table : Table
    records : Sequence[ConceptEmbeddingRecord]
        Concept metadata rows, one per embedding.
    embeddings : ndarray
        Float32 array of shape ``(N, D)``.

    Notes
    -----
    vec0 has no native UPDATE support so upsert is implemented as
    delete-then-insert.
    """
    concept_ids = [r.concept_id for r in records]

    with temp_filter_table(
        session, concept_ids, "INTEGER", table_name="_tmp_del_cids", dialect=dialect
    ) as temp_table_name:
        temp_table = Table(temp_table_name, MetaData(), Column("id", Integer))
        session.execute(
            delete(table).where(table.c.concept_id.in_(select(temp_table.c.id)))
        )

    session.execute(
        insert(table),
        [
            {
                "concept_id": rec.concept_id,
                "domain_id": rec.domain_id,
                "vocabulary_id": rec.vocabulary_id,
                "is_standard": rec.is_standard,
                "is_valid": rec.is_valid,
                EMBEDDING_COLUMN_NAME: _embedding_to_blob(emb),
            }
            for rec, emb in zip(records, embeddings)
        ],
    )


def query_knn(
    session: Session,
    table: Table,
    query_vector: ndarray,
    metric_type: MetricType,
    k: int,
    concept_filter: Optional[EmbeddingConceptFilter] = None,
) -> list[tuple[int, float, int]]:
    """Run a KNN query against a vec0 table using a per-query distance function.

    Parameters
    ----------
    session : Session
    table : Table
    query_vector : ndarray
        Float32 array of shape ``(D,)``.
    metric_type : MetricType
        Distance metric. Must be one of the keys in ``_QUERY_METRIC_FUNC``
        (L2, COSINE, L1).
    k : int
        Maximum number of results to return.
    concept_filter : EmbeddingConceptFilter, optional
        Row-level filters applied in the WHERE clause before ranking.

    Returns
    -------
    list[tuple[int, float, int]]
        Triples of ``(concept_id, distance, is_standard)`` ordered by distance ascending.

    Raises
    ------
    ValueError
        If ``metric_type`` is not supported for per-query distance functions.

    Notes
    -----
    Uses ``ORDER BY vec_distance_*(embedding, :emb) LIMIT k`` instead of the
    vec0 MATCH syntax. For FLAT (full-scan) tables the performance is identical
    and the metric can be chosen freely at call time.
    """
    dist_func_name = _QUERY_METRIC_FUNC.get(metric_type)
    if dist_func_name is None:
        raise ValueError(
            f"sqlite-vec does not support metric '{metric_type.value}' for FLAT queries. "
            f"Supported: {[m.value for m in _QUERY_METRIC_FUNC]}"
        )

    emb_blob = _embedding_to_blob(query_vector)
    distance = getattr(func, dist_func_name)(
        table.c[EMBEDDING_COLUMN_NAME], bindparam("q_emb", emb_blob)
    ).label("distance")

    stmt = select(table.c.concept_id, distance, table.c.is_standard).order_by(distance).limit(k)

    if concept_filter is not None:
        setup_concept_filter_temps(session, concept_filter, "sqlite")
        stmt = apply_concept_filter_where(stmt, table.c, concept_filter)

    rows = session.execute(stmt).all()
    return [(int(row[0]), float(row[1]), int(row[2])) for row in rows]


def query_concept_ids_matching_filter(
    session: Session, table: Table, concept_filter: EmbeddingConceptFilter
) -> set[int]:
    """Return every ``concept_id`` in `table` satisfying `concept_filter`.
    Used to build an exact FAISS pre-filter set, not for ranking.
    """
    setup_concept_filter_temps(session, concept_filter, "sqlite")
    stmt = apply_concept_filter_where(select(table.c.concept_id), table.c, concept_filter)
    rows = session.execute(stmt).all()
    return {int(row[0]) for row in rows}


def query_all_concept_ids(session: Session, table: Table) -> set[int]:
    """Return all concept IDs stored in a vec0 table.

    Parameters
    ----------
    session : Session
    table : Table

    Returns
    -------
    set[int]
    """
    rows = session.execute(select(table.c.concept_id)).all()
    return {int(row[0]) for row in rows}


def query_embeddings_by_ids(
    session: Session,
    table: Table,
    concept_ids: Sequence[int],
    dialect: str = "sqlite",
) -> dict[int, list[float]]:
    """Fetch embedding vectors for a set of concept IDs.

    Parameters
    ----------
    session : Session
    table : Table
    concept_ids : Sequence[int]

    Returns
    -------
    dict[int, list[float]]
        Mapping of concept ID to embedding vector.
    """
    with temp_filter_table(
        session,
        list(concept_ids),
        "INTEGER",
        table_name="_tmp_emb_cids",
        dialect=dialect,
    ) as temp_table_name:
        temp_table = Table(temp_table_name, MetaData(), Column("id", Integer))
        rows = session.execute(
            select(table.c.concept_id, table.c[EMBEDDING_COLUMN_NAME]).where(
                table.c.concept_id.in_(select(temp_table.c.id))
            )
        ).all()
    return {int(row[0]): _blob_to_embedding(row[1]) for row in rows}


def query_has_any(session: Session, table: Table) -> bool:
    """Return ``True`` if the vec0 table contains at least one row.

    Parameters
    ----------
    session : Session
    table : Table

    Returns
    -------
    bool
    """
    return session.execute(select(table.c.concept_id).limit(1)).first() is not None


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _embedding_to_blob(emb: ndarray) -> bytes:
    return emb.astype(np.float32).tobytes()


def _blob_to_embedding(blob: bytes) -> list[float]:
    return np.frombuffer(blob, dtype=np.float32).tolist()


def query_filter_metadata_by_ids(
    session: Session,
    table: Table,
    concept_ids: Sequence[int],
    dialect: str = "sqlite",
) -> dict[int, dict]:
    """Fetch filter metadata columns for a set of concept IDs.

    Parameters
    ----------
    session : Session
    table : Table
    concept_ids : Sequence[int]

    Returns
    -------
    dict[int, dict]
        ``{concept_id: {"domain_id": str, "vocabulary_id": str,
        "is_standard": bool, "is_valid": bool}}``
    """
    if not concept_ids:
        return {}
    concept_filter = EmbeddingConceptFilter(concept_ids=tuple(concept_ids))
    setup_concept_filter_temps(session, concept_filter, dialect)
    stmt = apply_concept_filter_where(
        select(
            table.c.concept_id,
            table.c.domain_id,
            table.c.vocabulary_id,
            table.c.is_standard,
            table.c.is_valid,
        ),
        table.c,
        concept_filter,
    )
    rows = session.execute(stmt).all()
    return {
        int(row[0]): {
            "domain_id": row[1] or "",
            "vocabulary_id": row[2] or "",
            "is_standard": bool(row[3]),
            "is_valid": bool(row[4]),
        }
        for row in rows
    }
