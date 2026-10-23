"""SQL helpers for the pgvector backend.

The embedding table carries filter columns alongside the vector:
  - ``domain_id``     TEXT    (OMOP domain)
  - ``vocabulary_id`` TEXT    (OMOP vocabulary)
  - ``is_standard``   BOOLEAN (standard_concept in ('S','C') → True)

These are populated at upsert time by the caller and enable efficient
pre-filtering during KNN without re-querying the OMOP CDM.
"""

from __future__ import annotations

import functools
import logging
import typing
from collections.abc import Sequence

import numpy as np
from numpy import ndarray
from oa_configurator import (
    Dialect,
    Role,
    guard_schema_provenance_for,
    physical_schema_of,
    qualified,
)
from sqlalchemy import (
    CursorResult,
    Engine,
    MetaData,
    Row,
    Select,
    delete,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.orm import Session, mapped_column, registry
from sqlalchemy.sql.elements import ColumnElement

from omop_emb.backends.base_backend import ConceptEmbeddingRecord
from omop_emb.backends.db_utils import apply_concept_filter_where, in_values, table_for
from omop_emb.backends.embedding_table import (
    EMBEDDING_COLUMN_NAME,
    ConceptEmbeddingMixin,
    EmbeddingTableBase,
    PGEmbeddingTable,
)
from omop_emb.backends.index_config import HNSWIndexConfig
from omop_emb.config import MetricType
from omop_emb.model_registry import EmbeddingModelRecord
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter, vector_column_type_for_dimensions

logger = logging.getLogger(__name__)


def table_exists(engine: Engine, table_name: str) -> bool:
    """Return ``True`` if *table_name* exists in the engine's configured schema."""
    return inspect(engine).has_table(table_name, schema=physical_schema_of(engine))


def create_pg_embedding_table(
    engine: Engine,
    model_record: EmbeddingModelRecord,
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
    - Uses ``halfvec(N)`` for dimensions greater than 2 000 and ``vector(N)``
    otherwise. Caching is handled by ``_ensure_storage_table`` in the backend
    base class; this function always issues DDL.

    - Requires provenance guard as EmbeddingBackend caches the engine and creates storage
        tables on it later (in register_model()), after construction-time enforcement has
        already run once.

    """
    table_cls = pg_embedding_table_descriptor(model_record)
    with (
        engine.begin() as connection,
        guard_schema_provenance_for(connection, schema_tag=Role.PRIMARY),
    ):
        EmbeddingTableBase.metadata.create_all(connection, tables=[table_for(table_cls)])
    return table_cls


def drop_pg_embedding_table(engine: Engine, model_record: EmbeddingModelRecord) -> None:
    """Drop the physical embedding table from the database.

    Parameters
    ----------
    engine : Engine
    model_record : EmbeddingModelRecord
    """
    tablename = model_record.storage_identifier
    with (
        engine.begin() as conn,
        guard_schema_provenance_for(conn, schema_tag=Role.PRIMARY),
    ):
        conn.execute(
            text(
                f"DROP TABLE IF EXISTS {qualified(conn, tablename, physical_schema=physical_schema_of(conn))}"
            )
        )
    logger.info(f"Dropped embedding table '{tablename}'.")


def hnsw_index_name(tablename: str, metric_type: MetricType) -> str:
    """Physical name of tablename's HNSW index for metric_type."""
    return f"idx_{tablename}_{metric_type.value}"


def hnsw_operator_class(metric_type: MetricType, dimensions: int) -> str:
    """pgvector operator class for metric_type on the column type dimensions selects."""
    return f"{vector_column_type_for_dimensions(dimensions).value}_{metric_type.value}_ops"


def hnsw_index_ddl(
    engine: Engine, model_record: EmbeddingModelRecord, index_config: HNSWIndexConfig
) -> str:
    """CREATE INDEX statement for model_record's table under index_config."""
    tablename = model_record.storage_identifier
    ops = hnsw_operator_class(index_config.metric_type, model_record.dimensions)
    table_ref = qualified(engine, tablename, physical_schema=physical_schema_of(engine))
    return (
        f"CREATE INDEX {hnsw_index_name(tablename, index_config.metric_type)} "
        f"ON {table_ref} "
        f"USING hnsw ({EMBEDDING_COLUMN_NAME} {ops}) "
        f"WITH (m = {index_config.num_neighbors}, ef_construction = {index_config.ef_construction})"
    )


# ---------------------------------------------------------------------------
# DML helpers
# ---------------------------------------------------------------------------


def upsert_embedding_rows(
    session: Session,
    records: Sequence[ConceptEmbeddingRecord],
    embeddings: ndarray,
    registered_table: type[PGEmbeddingTable],
) -> None:
    """Insert or update embedding rows with INSERT ... ON CONFLICT DO UPDATE.

    Rows are sent as executemany parameter sets, which SQLAlchemy splits into
    batches below PostgreSQL's bind-parameter limit, so any row count works.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session (must be in a transaction).
    records : Sequence[ConceptEmbeddingRecord]
        Concept metadata rows, one per embedding.
    embeddings : ndarray
        Float32 array of shape ``(N, D)``.
    registered_table : type[PGEmbeddingTable]
        ORM class for the target embedding table.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    stmt = pg_insert(registered_table)
    stmt = stmt.on_conflict_do_update(
        index_elements=["concept_id"],
        set_={
            "domain_id": stmt.excluded.domain_id,
            "vocabulary_id": stmt.excluded.vocabulary_id,
            "is_standard": stmt.excluded.is_standard,
            "is_valid": stmt.excluded.is_valid,
            EMBEDDING_COLUMN_NAME: getattr(stmt.excluded, EMBEDDING_COLUMN_NAME),
        },
    )
    session.execute(
        stmt,
        [
            {
                "concept_id": rec.concept_id,
                "domain_id": rec.domain_id,
                "vocabulary_id": rec.vocabulary_id,
                "is_standard": rec.is_standard,
                "is_valid": rec.is_valid,
                EMBEDDING_COLUMN_NAME: emb,
            }
            for rec, emb in zip(records, embeddings)
        ],
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


def q_embedding_count_by_vocabulary(embedding_table: type[PGEmbeddingTable]) -> Select:
    """Build a SELECT for stored embedding counts grouped by vocabulary_id.

    Parameters
    ----------
    embedding_table : type[PGEmbeddingTable]

    Returns
    -------
    Select
        Columns: ``vocabulary_id`` (str), ``count`` (int).
    """
    return select(embedding_table.vocabulary_id, func.count(embedding_table.concept_id)).group_by(
        embedding_table.vocabulary_id
    )


def _as_float32_array(value) -> ndarray:
    """Return a stored vector as a float32 array; halfvec values arrive as ``HalfVector``."""
    to_numpy = getattr(value, "to_numpy", None)
    return np.asarray(to_numpy() if to_numpy else value, dtype=np.float32)


def query_embeddings_by_ids(
    session: Session,
    embedding_table: type[PGEmbeddingTable],
    concept_ids: Sequence[int],
) -> dict[int, ndarray]:
    """Fetch float32 embedding vectors for concept_ids, keyed by concept ID; absent IDs are omitted."""
    column = inspect(embedding_table).columns
    rows = session.execute(
        select(column.concept_id, column[EMBEDDING_COLUMN_NAME]).where(
            in_values(column.concept_id, concept_ids, dialect=Dialect.POSTGRESQL)
        )
    ).all()
    return {int(row[0]): _as_float32_array(row[1]) for row in rows}


def delete_embedding_rows(
    session: Session,
    embedding_table: type[PGEmbeddingTable],
    concept_ids: Sequence[int],
) -> int:
    """Delete the rows of concept_ids and return how many existed.

    ``Session.execute`` is typed ``Result``; a DML statement returns a ``CursorResult``,
    which carries ``rowcount``.
    """
    column = inspect(embedding_table).columns
    result = session.execute(
        delete(embedding_table).where(
            in_values(column.concept_id, concept_ids, dialect=Dialect.POSTGRESQL)
        )
    )
    return int(typing.cast(CursorResult[typing.Any], result).rowcount)


# ---------------------------------------------------------------------------
# ANN query
# ---------------------------------------------------------------------------


def query_nearest_concept_ids(
    session: Session,
    embedding_table: type[PGEmbeddingTable],
    query_embedding: ndarray,
    metric_type: MetricType,
    k: int,
    concept_filter: EmbeddingConceptFilter | None = None,
) -> Sequence[Row]:
    """Run a pgvector ANN query for a single query vector.

    Notes
    -----
    Deliberately removed batching of queries for multiple vectors, as batching
    the vectors in one statement (a VALUES list joined through a lateral subquery)
    cast the query text to vector once per stored row and lost the parallel scan.
    Separate statements for each query vector are faster and scale linearly.


    Parameters
    ----------
    session : Session
    embedding_table : type[PGEmbeddingTable]
        ORM class for the embedding table.
    query_embedding : ndarray
        Query vector of shape ``(D,)``, bound as a typed vector.
    metric_type : MetricType
    k : int
        Maximum number of results.
    concept_filter : EmbeddingConceptFilter, optional

    Returns
    -------
    Sequence[Row]
        At most ``k`` rows ordered by distance with columns ``concept_id``,
        ``domain_id``, ``vocabulary_id``, ``is_standard``, ``is_valid`` and
        ``distance``.
    """
    distance = get_distance(embedding_table, query_embedding, metric_type)

    stmt = (
        select(
            embedding_table.concept_id,
            embedding_table.domain_id,
            embedding_table.vocabulary_id,
            embedding_table.is_standard,
            embedding_table.is_valid,
            distance.label("distance"),
        )
        .order_by(distance)
        .limit(k)
    )

    if concept_filter is not None:
        stmt = apply_concept_filter_where(
            stmt, inspect(embedding_table).columns, concept_filter, dialect=Dialect.POSTGRESQL
        )

    return session.execute(stmt).all()


def installed_pgvector_version(session: Session) -> tuple[int, ...]:
    """Return the installed ``vector`` extension version, e.g. ``(0, 8, 2)``."""
    version = session.execute(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    ).scalar_one()
    return tuple(int(part) for part in version.split("."))


def query_concept_ids_matching_filter(
    session: Session,
    embedding_table: type[PGEmbeddingTable],
    concept_filter: EmbeddingConceptFilter,
) -> set[int]:
    """Return every ``concept_id`` satisfying *concept_filter*."""
    stmt = select(embedding_table.concept_id)
    stmt = apply_concept_filter_where(
        stmt, inspect(embedding_table).columns, concept_filter, dialect=Dialect.POSTGRESQL
    )
    rows = session.execute(stmt).all()
    return {int(row[0]) for row in rows}


def query_concept_filter_metadata(
    session: Session,
    embedding_table: type[PGEmbeddingTable],
    concept_filter: EmbeddingConceptFilter,
) -> Sequence[Row]:
    """Return filter metadata columns (raw rows) for every concept ID
    satisfying concept_filter.

    Columns: ``concept_id``, ``domain_id``, ``vocabulary_id``, ``is_standard``,
    ``is_valid``. Row-to-domain-object conversion is the caller's job.
    """
    stmt = select(
        embedding_table.concept_id,
        embedding_table.domain_id,
        embedding_table.vocabulary_id,
        embedding_table.is_standard,
        embedding_table.is_valid,
    )
    stmt = apply_concept_filter_where(
        stmt, inspect(embedding_table).columns, concept_filter, dialect=Dialect.POSTGRESQL
    )
    return session.execute(stmt).all()


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------


def get_distance(
    embedding_table: type[PGEmbeddingTable],
    text_embedding: ndarray | ColumnElement,
    metric: MetricType,
) -> ColumnElement:
    """Return a SQLAlchemy distance expression for the given metric.

    Parameters
    ----------
    embedding_table : type[PGEmbeddingTable]
        ORM class whose ``embedding`` column is used. The column itself
        isn't declared on ``PGEmbeddingTable`` (its type depends on
        dimensionality), hence ``getattr``.
    text_embedding : ndarray or ColumnElement
        Query vector of shape ``(D,)``.
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
    """Return the SQLAlchemy ORM class descriptor for a pgvector embedding table.

    Cached by (tablename, dimensions): a repeated call for the same table
    reuses the one mapped class instead of building another, which
    ``extend_existing=True`` would otherwise only paper over (SQLAlchemy
    still warns about multiple mapped classes for one table).
    """
    return _cached_pg_embedding_table_descriptor(
        model_record.storage_identifier, model_record.dimensions
    )


@functools.cache
def _cached_pg_embedding_table_descriptor(
    tablename: str, dimensions: int
) -> type[PGEmbeddingTable]:
    """Build and map the embedding table class for one table name and dimension.

    The class is created with ``type()``, so the type checker cannot see its bases
    and the mapped result is cast to ``type[PGEmbeddingTable]``.
    """
    from pgvector.sqlalchemy import HALFVEC, VECTOR  # optional dependency

    from omop_emb.utils.embedding_utils import (
        VectorColumnType,
        vector_column_type_for_dimensions,
    )

    col_type = vector_column_type_for_dimensions(dimensions)
    descriptor_registry = registry(metadata=MetaData())
    emb_col = mapped_column(
        HALFVEC(dimensions) if col_type == VectorColumnType.HALFVEC else VECTOR(dimensions),
        nullable=False,
        index=False,
    )
    descriptor = type(
        f"PGEmbedding_{tablename}",
        (ConceptEmbeddingMixin,),
        {
            "__tablename__": tablename,
            "__table_args__": {"schema": Role.PRIMARY.value},
            "__module__": __name__,
            EMBEDDING_COLUMN_NAME: emb_col,
        },
    )
    return typing.cast(type[PGEmbeddingTable], descriptor_registry.mapped(descriptor))
