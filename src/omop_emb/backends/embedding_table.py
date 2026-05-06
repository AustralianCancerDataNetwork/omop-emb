from __future__ import annotations

from typing import Type

from sqlalchemy import Boolean, Engine, Integer, String
from sqlalchemy.orm import DeclarativeBase, mapped_column

from omop_emb.model_registry import EmbeddingModelRecord


class EmbeddingTableBase(DeclarativeBase):
    pass


class ConceptEmbeddingMixin:
    """Shared columns present in every embedding table across all backends.

    Notes
    -----
    The ``embedding`` column is added by backend-specific subclasses because
    the column type differs (pgvector ``Vector(N)`` vs sqlite-vec
    ``LargeBinary``). It is always named ``embedding`` so shared query
    helpers can reference it uniformly.

    Attributes
    ----------
    concept_id : int
        Primary key. OMOP concept ID.
    domain_id : str
        OMOP domain (e.g. ``'Condition'``, ``'Drug'``).
    vocabulary_id : str
        Source vocabulary (e.g. ``'SNOMED'``, ``'RxNorm'``).
    is_standard : bool
        ``True`` when ``standard_concept`` is ``'S'`` or ``'C'``.
    """

    concept_id = mapped_column(Integer, primary_key=True)
    domain_id = mapped_column(String, nullable=False)
    vocabulary_id = mapped_column(String, nullable=False)
    is_standard = mapped_column(Boolean, nullable=False)
    is_valid = mapped_column(Boolean, nullable=False, server_default="true")


def create_pg_embedding_table(
    engine: Engine,
    model_record: EmbeddingModelRecord,
) -> Type:
    """Create or retrieve a cached ORM class for a pgvector embedding table.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine for the pgvector database.
    model_record : EmbeddingModelRecord
        Registry record whose ``storage_identifier`` names the table and
        whose ``dimensions`` selects the column type.

    Returns
    -------
    type
        Dynamically generated SQLAlchemy ORM class mapped to the table.

    Notes
    -----
    Uses ``halfvec(N)`` for dimensions greater than 2 000 and ``vector(N)``
    otherwise, matching pgvector column-type limits.
    Results are cached in ``_PG_TABLE_CACHE`` keyed by ``storage_identifier``.
    """
    from omop_emb.utils.embedding_utils import (
        VectorColumnType, 
        vector_column_type_for_dimensions
    )
    from pgvector.sqlalchemy import VECTOR, HALFVEC  # optional dependency

    tablename = model_record.storage_identifier
    dimensions = model_record.dimensions

    if tablename in _PG_TABLE_CACHE:
        return _PG_TABLE_CACHE[tablename]

    col_type = vector_column_type_for_dimensions(dimensions)
    emb_col = mapped_column(
        HALFVEC(dimensions) if col_type == VectorColumnType.HALFVEC else VECTOR(dimensions),
        nullable=False,
        index=False,
    )

    table_cls = type(
        f"PGEmbedding_{tablename}",
        (ConceptEmbeddingMixin, EmbeddingTableBase),
        {
            "__tablename__": tablename,
            "__table_args__": {"extend_existing": True},
            "__module__": __name__,
            "embedding": emb_col,
        },
    )
    EmbeddingTableBase.metadata.create_all(engine, tables=[table_cls.__table__])  # type: ignore[arg-type]
    _PG_TABLE_CACHE[tablename] = table_cls
    return table_cls


def create_svec_embedding_table(model_record: EmbeddingModelRecord) -> Type:
    """Return an ORM class mapped to an existing sqlite-vec vec0 table.

    Parameters
    ----------
    model_record : EmbeddingModelRecord
        Registry record whose ``storage_identifier`` names the vec0 table.

    Returns
    -------
    type
        Dynamically generated SQLAlchemy ORM class.

    Notes
    -----
    The vec0 table itself is created separately via raw DDL (see
    ``sqlitevec_sql.ddl_create_vec0``). This ORM class is used for INSERT
    and DELETE only. KNN queries use raw SQL with the MATCH syntax.
    Results are cached in ``_SVEC_TABLE_CACHE`` keyed by ``storage_identifier``.
    """
    from sqlalchemy import LargeBinary

    tablename = model_record.storage_identifier

    if tablename in _SVEC_TABLE_CACHE:
        return _SVEC_TABLE_CACHE[tablename]

    table_cls = type(
        f"SVecEmbedding_{tablename}",
        (ConceptEmbeddingMixin, EmbeddingTableBase),
        {
            "__tablename__": tablename,
            "__table_args__": {"extend_existing": True},
            "__module__": __name__,
            "embedding": mapped_column(LargeBinary, nullable=False),
        },
    )
    _SVEC_TABLE_CACHE[tablename] = table_cls
    return table_cls


_PG_TABLE_CACHE: dict[str, type] = {}
_SVEC_TABLE_CACHE: dict[str, type] = {}
