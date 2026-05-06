"""SQL helpers for the sqlite-vec backend."""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
from numpy import ndarray
from sqlalchemy import text
from sqlalchemy.orm import Session

from omop_emb.backends.base_backend import ConceptEmbeddingRecord
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
        embedding_col = f"embedding FLOAT[{dimensions}] distance_metric={metric_str}"
    else:
        embedding_col = f"embedding FLOAT[{dimensions}]"

    return (
        f'CREATE VIRTUAL TABLE IF NOT EXISTS "{table_name}" USING vec0('
        f"concept_id INTEGER PRIMARY KEY, "
        f"domain_id TEXT METADATA, "
        f"vocabulary_id TEXT METADATA, "
        f"is_standard INTEGER METADATA, "
        f"{embedding_col}"
        f")"
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
    table_name: str,
    records: Sequence[ConceptEmbeddingRecord],
    embeddings: ndarray,
) -> None:
    """Upsert embedding rows into a vec0 table.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session (must be in a transaction).
    table_name : str
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
    id_placeholders = ",".join(f":d{i}" for i in range(len(concept_ids)))
    session.execute(
        text(f'DELETE FROM "{table_name}" WHERE concept_id IN ({id_placeholders})'),
        {f"d{i}": cid for i, cid in enumerate(concept_ids)},
    )

    for rec, emb in zip(records, embeddings):
        session.execute(
            text(
                f'INSERT INTO "{table_name}" '
                f"(concept_id, domain_id, vocabulary_id, is_standard, embedding) "
                f"VALUES (:cid, :did, :vid, :std, :emb)"
            ),
            {
                "cid": rec.concept_id,
                "did": rec.domain_id,
                "vid": rec.vocabulary_id,
                "std": int(rec.is_standard),
                "emb": _embedding_to_blob(emb),
            },
        )


def query_knn(
    session: Session,
    table_name: str,
    query_vector: ndarray,
    metric_type: MetricType,
    k: int,
    concept_filter: Optional[EmbeddingConceptFilter] = None,
) -> list[tuple[int, float]]:
    """Run a KNN query against a vec0 table using a per-query distance function.

    Parameters
    ----------
    session : Session
    table_name : str
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
    list[tuple[int, float]]
        Pairs of ``(concept_id, distance)`` ordered by distance ascending.

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
    dist_func = _QUERY_METRIC_FUNC.get(metric_type)
    if dist_func is None:
        raise ValueError(
            f"sqlite-vec does not support metric '{metric_type.value}' for FLAT queries. "
            f"Supported: {[m.value for m in _QUERY_METRIC_FUNC]}"
        )

    emb_blob = _embedding_to_blob(query_vector)
    params: dict = {"emb": emb_blob, "k": k}
    where_clauses: list[str] = []

    if concept_filter is not None:
        if concept_filter.concept_ids is not None:
            ids = list(concept_filter.concept_ids)
            placeholders = ",".join(f":cid{i}" for i in range(len(ids)))
            where_clauses.append(f"concept_id IN ({placeholders})")
            params.update({f"cid{i}": cid for i, cid in enumerate(ids)})
        if concept_filter.domains is not None:
            domains = list(concept_filter.domains)
            placeholders = ",".join(f":dom{i}" for i in range(len(domains)))
            where_clauses.append(f"domain_id IN ({placeholders})")
            params.update({f"dom{i}": d for i, d in enumerate(domains)})
        if concept_filter.vocabularies is not None:
            vocabs = list(concept_filter.vocabularies)
            placeholders = ",".join(f":voc{i}" for i in range(len(vocabs)))
            where_clauses.append(f"vocabulary_id IN ({placeholders})")
            params.update({f"voc{i}": v for i, v in enumerate(vocabs)})
        if concept_filter.require_standard:
            where_clauses.append("is_standard = 1")

    where_str = f"WHERE {' AND '.join(where_clauses)} " if where_clauses else ""
    sql = text(
        f'SELECT concept_id, {dist_func}(embedding, :emb) as distance '
        f'FROM "{table_name}" '
        f"{where_str}"
        f"ORDER BY distance LIMIT :k"
    )
    rows = session.execute(sql, params).all()
    return [(int(row[0]), float(row[1])) for row in rows]


def query_all_concept_ids(session: Session, table_name: str) -> set[int]:
    """Return all concept IDs stored in a vec0 table.

    Parameters
    ----------
    session : Session
    table_name : str

    Returns
    -------
    set[int]
    """
    rows = session.execute(text(f'SELECT concept_id FROM "{table_name}"')).all()
    return {int(row[0]) for row in rows}


def query_embeddings_by_ids(
    session: Session,
    table_name: str,
    concept_ids: Sequence[int],
) -> dict[int, list[float]]:
    """Fetch embedding vectors for a set of concept IDs.

    Parameters
    ----------
    session : Session
    table_name : str
    concept_ids : Sequence[int]

    Returns
    -------
    dict[int, list[float]]
        Mapping of concept ID to embedding vector.
    """
    placeholders = ",".join(f":c{i}" for i in range(len(concept_ids)))
    rows = session.execute(
        text(f'SELECT concept_id, embedding FROM "{table_name}" WHERE concept_id IN ({placeholders})'),
        {f"c{i}": cid for i, cid in enumerate(concept_ids)},
    ).all()
    return {int(row[0]): _blob_to_embedding(row[1]) for row in rows}


def query_has_any(session: Session, table_name: str) -> bool:
    """Return ``True`` if the vec0 table contains at least one row.

    Parameters
    ----------
    session : Session
    table_name : str

    Returns
    -------
    bool
    """
    return session.execute(
        text(f'SELECT concept_id FROM "{table_name}" LIMIT 1')
    ).first() is not None


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _embedding_to_blob(emb: ndarray) -> bytes:
    return emb.astype(np.float32).tobytes()


def _blob_to_embedding(blob: bytes) -> list[float]:
    return np.frombuffer(blob, dtype=np.float32).tolist()
