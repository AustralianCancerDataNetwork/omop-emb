from __future__ import annotations

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


def _build_pg_embedding_cls(model_record: EmbeddingModelRecord) -> type:
    """Build the SQLAlchemy ORM class for a pgvector embedding table."""
    from omop_emb.utils.embedding_utils import VectorColumnType, vector_column_type_for_dimensions
    from pgvector.sqlalchemy import VECTOR, HALFVEC  # optional dependency

    tablename = model_record.storage_identifier
    dimensions = model_record.dimensions
    col_type = vector_column_type_for_dimensions(dimensions)
    emb_col = mapped_column(
        HALFVEC(dimensions) if col_type == VectorColumnType.HALFVEC else VECTOR(dimensions),
        nullable=False,
        index=False,
    )
    return type(
        f"PGEmbedding_{tablename}",
        (ConceptEmbeddingMixin, EmbeddingTableBase),
        {
            "__tablename__": tablename,
            "__table_args__": {"extend_existing": True},
            "__module__": __name__,
            "embedding": emb_col,
        },
    )


def load_pg_embedding_table(model_record: EmbeddingModelRecord) -> type:
    """Return an ORM class for an existing pgvector embedding table without DDL.

    Parameters
    ----------
    model_record : EmbeddingModelRecord

    Returns
    -------
    type
        SQLAlchemy ORM class ready for queries. No ``CREATE TABLE`` is issued.
    """
    return _build_pg_embedding_cls(model_record)


def create_pg_embedding_table(engine: Engine, model_record: EmbeddingModelRecord) -> type:
    """Create a pgvector embedding table and return its ORM class.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine for the pgvector database.
    model_record : EmbeddingModelRecord

    Returns
    -------
    type
        SQLAlchemy ORM class mapped to the newly created table.

    Notes
    -----
    Uses ``halfvec(N)`` for dimensions greater than 2 000 and ``vector(N)``
    otherwise. Caching is handled by ``_ensure_storage_table`` in the backend
    base class; this function always issues DDL.
    """
    table_cls = _build_pg_embedding_cls(model_record)
    EmbeddingTableBase.metadata.create_all(engine, tables=[table_cls.__table__])  # type: ignore[arg-type]
    return table_cls


