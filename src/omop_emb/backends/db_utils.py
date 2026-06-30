"""Backend-agnostic database utilities shared across storage backends."""

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from sqlalchemy import Column, Integer, MetaData, Select, String, Table, select, text
from sqlalchemy.orm import Session
from sqlalchemy.sql.base import ColumnCollection

from omop_emb.utils.embedding_utils import EmbeddingConceptFilter

# Standardised temp table names used by setup_concept_filter_temps.
# Both backends reference these constants so the names stay in sync.
KNN_CIDS_TABLE = "_knn_cids"
KNN_DOMS_TABLE = "_knn_doms"
KNN_VOCS_TABLE = "_knn_vocs"

_TEMP_FILTER_METADATA = MetaData()
KNN_CIDS_CORE_TABLE = Table(KNN_CIDS_TABLE, _TEMP_FILTER_METADATA, Column("id", Integer))
KNN_DOMS_CORE_TABLE = Table(KNN_DOMS_TABLE, _TEMP_FILTER_METADATA, Column("id", String))
KNN_VOCS_CORE_TABLE = Table(KNN_VOCS_TABLE, _TEMP_FILTER_METADATA, Column("id", String))


def apply_concept_filter_where(
    stmt: Select,
    columns: ColumnCollection[str, Column[Any]],
    concept_filter: EmbeddingConceptFilter,
) -> Select:
    """Apply concept_filter's WHERE-clause constraints to stmt.

    Parameters
    ----------
    stmt : Select
    columns : ColumnCollection[str, Column[Any]]
        ``embedding_table.c`` (Core) or ``sa.inspect(embedding_table).columns``
        (ORM). Both expose the TEmbeddingTable columns as real ``Column`` objects.
    concept_filter : EmbeddingConceptFilter

    Notes
    -----
    Assumes :func:`setup_concept_filter_temps` has already populated the
    referenced temp tables in the same transaction.
    """
    if concept_filter.concept_ids is not None:
        stmt = stmt.where(columns.concept_id.in_(select(KNN_CIDS_CORE_TABLE.c.id)))
    if concept_filter.domains is not None:
        stmt = stmt.where(columns.domain_id.in_(select(KNN_DOMS_CORE_TABLE.c.id)))
    if concept_filter.vocabularies is not None:
        stmt = stmt.where(columns.vocabulary_id.in_(select(KNN_VOCS_CORE_TABLE.c.id)))
    if concept_filter.require_standard:
        stmt = stmt.where(columns.is_standard == True)  # noqa: E712
    if concept_filter.require_active:
        stmt = stmt.where(columns.is_valid == True)  # noqa: E712
    return stmt


@contextmanager
def temp_filter_table(
    session: Session,
    values: Sequence,
    col_type: str,
    table_name: str,
    *,
    dialect: str,
) -> Iterator[str]:
    """Bulk-insert *values* into a temporary table and yield its name.

    Avoids bind-parameter explosion from large IN clauses by moving the value
    list into a temporary table and joining against it.  Each row is sent as a
    separate protocol message (executemany), so there is no parameter limit
    regardless of list length.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session (must already be in a transaction).
    values : Sequence
        Values to load into the temp table's ``id`` column.
    col_type : str
        SQL column type: ``'INTEGER'`` or ``'BIGINT'`` for integer concept IDs,
        ``'TEXT'`` for domain / vocabulary strings.
    table_name : str
        Name of the temporary table to create.
    dialect : str
        ``'postgresql'`` or ``'sqlite'``.

    Yields
    ------
    table_name : str
         Use it in a ``JOIN`` or ``IN (SELECT id FROM …)`` clause.

    Notes
    -----
    **PostgreSQL**: ``CREATE TEMPORARY TABLE … ON COMMIT DROP``.  The table is
    dropped automatically when the enclosing transaction commits; no explicit
    cleanup is needed.

    **SQLite**: ``CREATE TEMPORARY TABLE IF NOT EXISTS … ; DELETE FROM …``.
    Temp tables are connection-scoped, so the idempotent pattern handles pooled
    connections safely.  The table is truncated before each use and cleaned up
    when the connection is returned to the pool.
    """
    if dialect == "postgresql":
        session.execute(
            text(
                f'CREATE TEMPORARY TABLE "{table_name}" (id {col_type}) ON COMMIT DROP'
            )
        )
    elif dialect == "sqlite":
        session.execute(
            text(f'CREATE TEMPORARY TABLE IF NOT EXISTS "{table_name}" (id {col_type})')
        )
        session.execute(text(f'DELETE FROM "{table_name}"'))
    else:
        raise ValueError(f"Unsupported dialect: {dialect}")

    if values:
        session.execute(
            text(f'INSERT INTO "{table_name}" (id) VALUES (:v)'),
            [{"v": v} for v in values],
        )

    yield table_name


def setup_concept_filter_temps(
    session: Session,
    concept_filter: EmbeddingConceptFilter,
    dialect: str,
) -> None:
    """Populate a temp table for each list-valued field in *concept_filter*.

    Creates (or resets for SQLite) one temp table per non-``None`` list field
    and bulk-inserts its values.  The created tables use the module-level
    constants :data:`KNN_CIDS_TABLE`, :data:`KNN_DOMS_TABLE`, and
    :data:`KNN_VOCS_TABLE` as their names.

    Callers determine which WHERE clauses to add by checking the corresponding
    ``concept_filter`` fields directly. No return value is needed because the
    filter already encodes which fields are set.

    Call this at the start of a transaction before running any query that needs
    these tables.  For PostgreSQL the tables are ``ON COMMIT DROP``; for SQLite
    they persist for the connection lifetime and are reset on each call.

    Parameters
    ----------
    session : Session
    concept_filter : EmbeddingConceptFilter
    dialect : str
        ``'postgresql'`` or ``'sqlite'``.
    """
    if concept_filter.concept_ids is not None:
        with temp_filter_table(
            session,
            list(concept_filter.concept_ids),
            "INTEGER",
            KNN_CIDS_TABLE,
            dialect=dialect,
        ):
            pass

    if concept_filter.domains is not None:
        with temp_filter_table(
            session,
            list(concept_filter.domains),
            "TEXT",
            KNN_DOMS_TABLE,
            dialect=dialect,
        ):
            pass

    if concept_filter.vocabularies is not None:
        with temp_filter_table(
            session,
            list(concept_filter.vocabularies),
            "TEXT",
            KNN_VOCS_TABLE,
            dialect=dialect,
        ):
            pass
