"""SQL helpers for the sqlite-vec backend."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
from numpy import ndarray
from oa_configurator import Dialect
from sqlalchemy import (
    Column,
    Engine,
    MetaData,
    Row,
    Select,
    Table,
    bindparam,
    column,
    delete,
    func,
    insert,
    select,
    text,
    LargeBinary
)
from sqlalchemy.orm import Session

from omop_emb.backends.base_backend import ConceptEmbeddingRecord
from omop_emb.backends.db_utils import apply_concept_filter_where, in_values
from omop_emb.backends.embedding_table import CONCEPT_METADATA_COLUMNS, EMBEDDING_COLUMN_NAME
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter

logger = logging.getLogger(__name__)

# Largest k that a vec0 MATCH query accepts.
MAX_KNN_K = 4096

_COSINE_DECLARATION = "distance_metric=cosine"


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

def table_declares_cosine(engine: Engine, table_name: str) -> bool:
    """Return ``True`` if the vec0 table *table_name* was created with the cosine metric.

    Parameters
    ----------
    engine : Engine
    table_name : str

    Returns
    -------
    bool
        ``False`` for a table created without ``distance_metric=cosine``.
    """
    with engine.connect() as conn:
        ddl = conn.execute(
            text("SELECT sql FROM sqlite_master WHERE name = :n"), {"n": table_name}
        ).scalar_one()
    return _COSINE_DECLARATION in "".join(str(ddl).lower().split())


def recreate_vec0_with_cosine(engine: Engine, table_name: str, dimensions: int) -> int:
    """Recreate a vec0 table with the cosine metric, keeping every stored row.

    Parameters
    ----------
    engine : Engine
        Writable engine.
    table_name : str
        Physical table name (the registry ``storage_identifier``).
    dimensions : int
        Embedding vector length.

    Returns
    -------
    int
        Number of rows carried over.

    Notes
    -----
    A vec0 table cannot be altered or renamed, so rows are copied to a staging
    table, the original is recreated, and rows are copied back, in one
    transaction. The transaction is opened explicitly because the sqlite
    driver does not open one for DDL, so a failure would otherwise leave the
    staging table behind.
    """
    staging = f"{table_name}__rebuild"
    columns = ", ".join([*(c.name for c in CONCEPT_METADATA_COLUMNS), EMBEDDING_COLUMN_NAME])
    with engine.begin() as conn:
        conn.exec_driver_sql("BEGIN IMMEDIATE")
        conn.execute(text(ddl_drop_vec0(staging)))
        conn.execute(text(ddl_create_vec0(staging, dimensions)))
        conn.execute(text(f'INSERT INTO "{staging}"({columns}) SELECT {columns} FROM "{table_name}"'))
        conn.execute(text(ddl_drop_vec0(table_name)))
        conn.execute(text(ddl_create_vec0(table_name, dimensions)))
        conn.execute(text(f'INSERT INTO "{table_name}"({columns}) SELECT {columns} FROM "{staging}"'))
        count = conn.execute(text(f'SELECT count(*) FROM "{table_name}"')).scalar_one()
        conn.execute(text(ddl_drop_vec0(staging)))
    return int(count)


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
        table. Does not issue DDL: the ``vec0`` table must already exist.
    """
    columns = [
        Column(c.name, c.type_, primary_key=(c.name == "concept_id"))
        for c in CONCEPT_METADATA_COLUMNS
    ]
    columns.append(Column(EMBEDDING_COLUMN_NAME, LargeBinary))
    return Table(table_name, metadata, *columns, extend_existing=True)


def ddl_create_vec0(table_name: str, dimensions: int) -> str:
    """Return DDL to create a vec0 virtual table for the given model.

    The embedding column declares the cosine metric, which ``MATCH`` queries
    use. The metric cannot be changed without recreating the table.

    Parameters
    ----------
    table_name : str
        Physical table name (the registry ``storage_identifier``).
    dimensions : int
        Embedding vector length.

    Returns
    -------
    str
        ``CREATE VIRTUAL TABLE IF NOT EXISTS ...`` statement.
    """
    embedding_col = f"{EMBEDDING_COLUMN_NAME} FLOAT[{dimensions}] {_COSINE_DECLARATION}"
    metadata_cols = ", ".join(f"{c.name} {c.vec0_ddl}" for c in CONCEPT_METADATA_COLUMNS)

    return (
        f'CREATE VIRTUAL TABLE IF NOT EXISTS "{table_name}" USING vec0('
        f"{metadata_cols}, {embedding_col})"
    )


def vacuum_database(engine: Engine) -> None:
    """Run ``VACUUM`` so the pages freed by a table recreation shrink the file.

    Parameters
    ----------
    engine : Engine
        Writable engine. ``VACUUM`` needs free disk space for a second copy of
        the data while it runs and cannot run inside a transaction.
    """
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.exec_driver_sql("VACUUM")


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


def dml_delete_rows(
    session: Session,
    table: Table,
    concept_ids: Sequence[int],
) -> int:
    """Delete the rows of concept_ids from a vec0 table and return how many existed."""
    existing = session.execute(
        select(func.count()).select_from(table).where(
            in_values(table.c.concept_id, concept_ids, dialect=Dialect.SQLITE)
        )
    ).scalar_one()
    session.execute(
        delete(table).where(in_values(table.c.concept_id, concept_ids, dialect=Dialect.SQLITE))
    )
    return int(existing)


def dml_upsert_rows(
    session: Session,
    table: Table,
    records: Sequence[ConceptEmbeddingRecord],
    embeddings: ndarray,
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
    session.execute(
        delete(table).where(in_values(table.c.concept_id, concept_ids, dialect=Dialect.SQLITE))
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


def _build_knn_stmt(
    table: Table,
    query_vector: ndarray,
    k: int,
    concept_filter: Optional[EmbeddingConceptFilter] = None,
) -> Select:
    """Build a single-vector ``MATCH`` KNN statement. Internal; see :func:`query_knn_batch`."""
    stmt = (
        select(
            table.c.concept_id,
            column("distance"),
            table.c.domain_id,
            table.c.vocabulary_id,
            table.c.is_standard,
            table.c.is_valid,
        )
        .where(
            table.c[EMBEDDING_COLUMN_NAME].op("MATCH")(
                bindparam("q_emb", _embedding_to_blob(query_vector))
            ),
            column("k") == k,
        )
        .order_by(column("distance"))
    )

    if concept_filter is not None:
        stmt = apply_concept_filter_where(stmt, table.c, concept_filter, dialect=Dialect.SQLITE)

    return stmt


def query_knn_batch(
    session: Session,
    table: Table,
    query_vectors: Sequence[ndarray],
    k: int,
    concept_filter: Optional[EmbeddingConceptFilter] = None,
) -> list[Sequence[Row]]:
    """Run cosine KNN queries against a vec0 table, one per vector in *query_vectors*.

    Parameters
    ----------
    session : Session
    table : Table
    query_vectors : Sequence[ndarray]
        Each a float32 array of shape ``(D,)``.
    k : int
        Maximum number of results to return per vector, at most ``MAX_KNN_K``.
    concept_filter : EmbeddingConceptFilter, optional
        Row-level filters applied during the scan, so ``k`` rows pass the filter.

    Returns
    -------
    list[Sequence[Row]]
        One row sequence per query vector. Each row contains ``concept_id``,
        ``distance``, ``domain_id``, ``vocabulary_id``, ``is_standard``, and
        ``is_valid``, ordered by cosine distance ascending.

    Raises
    ------
    ValueError
        If ``k`` exceeds ``MAX_KNN_K``.

    Notes
    -----
    Uses the vec0 ``MATCH ... AND k = ?`` form with the metric declared on the
    table. sqlite-vec has no multi-vector form, so each vector is its own
    statement.
    """
    if k > MAX_KNN_K:
        raise ValueError(
            f"sqlite-vec returns at most {MAX_KNN_K} results per query. Got k={k}."
        )
    results: list[Sequence[Row]] = []
    for query_vector in query_vectors:
        stmt = _build_knn_stmt(table, query_vector, k, concept_filter)
        results.append(session.execute(stmt).all())
    return results


def query_concept_ids_matching_filter(
    session: Session,
    table: Table,
    concept_filter: EmbeddingConceptFilter,
) -> set[int]:
    """Return every ``concept_id`` in `table` satisfying `concept_filter`.
    Used to build an exact FAISS pre-filter set, not for ranking.
    """
    stmt = apply_concept_filter_where(select(table.c.concept_id), table.c, concept_filter, dialect=Dialect.SQLITE)
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


def query_embedding_count_by_vocabulary(session: Session, table: Table) -> dict[str, int]:
    """Return stored embedding counts grouped by vocabulary_id.

    Parameters
    ----------
    session : Session
    table : Table

    Returns
    -------
    dict[str, int]
        ``vocabulary_id`` to embedding count, for every vocabulary with at
        least one stored embedding.
    """
    stmt = (
        select(table.c.vocabulary_id, func.count(table.c.concept_id))
        .group_by(table.c.vocabulary_id)
    )
    rows = session.execute(stmt).all()
    return {row[0]: int(row[1]) for row in rows}


def query_embeddings_by_ids(
    session: Session,
    table: Table,
    concept_ids: Sequence[int],
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
    rows = session.execute(
        select(table.c.concept_id, table.c[EMBEDDING_COLUMN_NAME]).where(
            in_values(table.c.concept_id, concept_ids, dialect=Dialect.SQLITE)
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


def query_concept_filter_metadata(
    session: Session,
    table: Table,
    concept_filter: EmbeddingConceptFilter,
) -> Sequence[Row]:
    """Return filter metadata columns (raw rows) for every concept ID
    satisfying concept_filter.

    Columns: ``concept_id``, ``domain_id``, ``vocabulary_id``, ``is_standard``,
    ``is_valid``. Row-to-domain-object conversion is the caller's job.
    """
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
        dialect=Dialect.SQLITE,
    )
    return session.execute(stmt).all()
