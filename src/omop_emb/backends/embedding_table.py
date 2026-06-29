from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass

from sqlalchemy import Boolean, Integer, String
from sqlalchemy.orm import DeclarativeBase, MappedColumn, mapped_column
from sqlalchemy.sql.type_api import TypeEngine


@dataclass(frozen=True)
class EmbeddingColumnSpec:
    """One concept-metadata column shared by every embedding table backend.

    Parameters
    ----------
    name : str
    type_ : type[TypeEngine]
        SQLAlchemy Core type used to build sqlite-vec's ``Table``.
    vec0_ddl : str
        DDL fragment after the column name in ``CREATE VIRTUAL TABLE ... USING vec0(...)``.
    """

    name: str
    type_: type[TypeEngine]
    vec0_ddl: str


CONCEPT_METADATA_COLUMNS: tuple[EmbeddingColumnSpec, ...] = (
    EmbeddingColumnSpec("concept_id", Integer, "INTEGER PRIMARY KEY"),
    EmbeddingColumnSpec("domain_id", String, "TEXT METADATA"),
    EmbeddingColumnSpec("vocabulary_id", String, "TEXT METADATA"),
    EmbeddingColumnSpec("is_standard", Boolean, "BOOLEAN METADATA"),
    EmbeddingColumnSpec("is_valid", Boolean, "BOOLEAN METADATA DEFAULT 1"),
)

EMBEDDING_COLUMN_NAME = "embedding"


def _check_columns_match_spec(cls: type) -> type:
    """Class decorator: assert cls's declared columns match CONCEPT_METADATA_COLUMNS.

    Works for both dataclasses (checked via ``fields()``) and SQLAlchemy
    mapped classes (checked via their ``MappedColumn`` attributes).
    """
    if is_dataclass(cls):
        declared = {f.name for f in fields(cls)}
    else:
        declared = {name for name, value in vars(cls).items() if isinstance(value, MappedColumn)}
    expected = {c.name for c in CONCEPT_METADATA_COLUMNS}
    assert declared == expected, (
        f"{cls.__name__} columns {declared} do not match CONCEPT_METADATA_COLUMNS {expected}."
    )
    return cls


@_check_columns_match_spec
@dataclass(frozen=True)
class ConceptEmbeddingRecord:
    """Concept metadata for a single embedding upsert row.

    Populated from the OMOP CDM by the caller (interface layer) before being
    passed to the backend.

    Attributes
    ----------
    concept_id : int
        OMOP concept ID.
    domain_id : str
        OMOP domain (e.g. ``'Condition'``, ``'Drug'``).
    vocabulary_id : str
        Source vocabulary (e.g. ``'SNOMED'``, ``'RxNorm'``).
    is_standard : bool
        ``True`` if ``standard_concept`` is ``'S'`` or ``'C'``.
    """

    concept_id: int
    domain_id: str
    vocabulary_id: str
    is_standard: bool
    is_valid: bool = True


class EmbeddingTableBase(DeclarativeBase):
    pass


@_check_columns_match_spec
class ConceptEmbeddingMixin:
    """Shared columns present in every embedding table across all backends.

    Notes
    -----
    The ``embedding`` column is added by backend-specific subclasses because
    the column type differs (pgvector ``Vector(N)`` vs sqlite-vec
    ``LargeBinary``). It is always named :data:`EMBEDDING_COLUMN_NAME` so
    shared query helpers can reference it uniformly.

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


class PGEmbeddingTable(ConceptEmbeddingMixin, EmbeddingTableBase):
    """Abstract base for pgvector embedding tables.

    Concrete per-model subclasses come from
    :func:`~omop_emb.backends.pgvector.pg_sql.pg_embedding_table_descriptor`,
    which adds the ``embedding`` column (type depends on dimensionality, so
    it can't be declared statically here -- access it via
    ``getattr(table, EMBEDDING_COLUMN_NAME)``).
    """

    __abstract__ = True

