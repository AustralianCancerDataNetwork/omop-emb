"""Backend-agnostic database utilities shared across storage backends."""

import json
from typing import Any, Sequence

from oa_configurator import Dialect
from sqlalchemy import Column, ColumnElement, Integer, Select, any_, bindparam, func, select
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.sql.base import ColumnCollection

from omop_emb.utils.embedding_utils import EmbeddingConceptFilter


def in_values(column: Column[Any], values: Sequence[Any], *, dialect: str) -> ColumnElement[bool]:
    """Return ``column IN values`` with values sent as one bind parameter, whatever their number.

    PostgreSQL compares against one array parameter (``= ANY(:values)``);
    SQLite reads one JSON text parameter through ``json_each``. Both are
    semi-joins, so duplicate values never duplicate rows.

    Parameters
    ----------
    column : Column
        Integer or text column to filter.
    values : Sequence
        Values to match.
    dialect : str
        ``'postgresql'`` or ``'sqlite'``.

    Returns
    -------
    ColumnElement[bool]
    """
    cast_value = int if isinstance(column.type, Integer) else str
    plain = [cast_value(v) for v in values]
    if dialect == Dialect.POSTGRESQL:
        return column == any_(bindparam(None, plain, type_=ARRAY(column.type)))
    if dialect == Dialect.SQLITE:
        listed = func.json_each(bindparam(None, json.dumps(plain))).table_valued("value")
        return column.in_(select(listed.c.value))
    raise ValueError(f"Unsupported dialect: {dialect}")


def apply_concept_filter_where(
    stmt: Select,
    columns: ColumnCollection[str, Column[Any]],
    concept_filter: EmbeddingConceptFilter,
    *,
    dialect: str,
) -> Select:
    """Apply concept_filter's WHERE-clause constraints to stmt.

    Parameters
    ----------
    stmt : Select
    columns : ColumnCollection[str, Column[Any]]
        ``embedding_table.c`` (Core) or ``sa.inspect(embedding_table).columns``
        (ORM). Both expose the TEmbeddingTable columns as real ``Column`` objects.
    concept_filter : EmbeddingConceptFilter
    dialect : str
        ``'postgresql'`` or ``'sqlite'``.

    Returns
    -------
    Select
    """
    if concept_filter.concept_ids is not None:
        stmt = stmt.where(in_values(columns.concept_id, concept_filter.concept_ids, dialect=dialect))
    if concept_filter.domains is not None:
        stmt = stmt.where(in_values(columns.domain_id, concept_filter.domains, dialect=dialect))
    if concept_filter.vocabularies is not None:
        stmt = stmt.where(in_values(columns.vocabulary_id, concept_filter.vocabularies, dialect=dialect))
    if concept_filter.require_standard:
        stmt = stmt.where(columns.is_standard == True)  # noqa: E712
    if concept_filter.require_active:
        stmt = stmt.where(columns.is_valid == True)  # noqa: E712
    return stmt
