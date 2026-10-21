"""SQL helpers for the DuckDB VSS embedding backend.

The embedding column is a fixed-size ``FLOAT[N]`` array (DuckDB VSS's HNSW
index requires exactly that type, not a variable-length ``LIST``). Distance
is computed with ``array_distance`` (L2) / ``array_cosine_distance``
(cosine); DuckDB VSS has no L1 array function. Each query vector is sent as
its own statement: DuckDB has no lateral-join equivalent of pgvector's
multi-vector batching verified here, so batching is left for later if it
turns out to matter.
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
    REAL,
    Engine,
    MetaData,
    Row,
    Select,
    Table,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.orm import Session, mapped_column, registry
from sqlalchemy.schema import DropTable
from sqlalchemy.sql import cast
from sqlalchemy.sql.elements import ColumnElement

try:
    from duckdb_sqlalchemy.datatypes import FixedArray
except ImportError as _e:
    raise ImportError(
        "duckdb-sqlalchemy is not installed. Install it with: pip install omop-emb[duckdbvss]"
    ) from _e

from omop_emb.backends.base_backend import ConceptEmbeddingRecord
from omop_emb.backends.db_utils import apply_concept_filter_where
from omop_emb.backends.embedding_table import (
    EMBEDDING_COLUMN_NAME,
    ConceptEmbeddingMixin,
    DuckDBVSSEmbeddingTable,
    EmbeddingTableBase,
)
from omop_emb.backends.index_config import HNSWIndexConfig
from omop_emb.config import MetricType
from omop_emb.model_registry import EmbeddingModelRecord
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter

logger = logging.getLogger(__name__)

_HNSW_METRIC = {
    MetricType.L2: "l2sq",
    MetricType.COSINE: "cosine",
}
_DISTANCE_FUNC = {
    MetricType.L2: "array_distance",
    MetricType.COSINE: "array_cosine_distance",
}


def table_exists(engine: Engine, table_name: str) -> bool:
    """Return ``True`` if *table_name* exists in the engine's configured schema."""
    return inspect(engine).has_table(table_name, schema=physical_schema_of(engine))


def create_duckdbvss_embedding_table(
    engine: Engine,
    model_record: EmbeddingModelRecord,
) -> type[DuckDBVSSEmbeddingTable]:
    """Create a DuckDB VSS embedding table and return its ORM class."""
    table_cls = duckdbvss_embedding_table_descriptor(model_record)
    with engine.begin() as connection:
        with guard_schema_provenance_for(connection, schema_tag=Role.PRIMARY):
            EmbeddingTableBase.metadata.create_all(connection, tables=[table_cls.__table__])  # ty: ignore[invalid-argument-type]
    return table_cls


def drop_duckdbvss_embedding_table(engine: Engine, model_record: EmbeddingModelRecord) -> None:
    """Drop the physical embedding table from the database."""
    tablename = model_record.storage_identifier
    with engine.begin() as conn:
        with guard_schema_provenance_for(conn, schema_tag=Role.PRIMARY):
            conn.execute(
                DropTable(Table(tablename, MetaData(), schema=physical_schema_of(conn)), if_exists=True)
            )
    logger.info(f"Dropped embedding table '{tablename}'.")


def hnsw_index_name(tablename: str, metric_type: MetricType) -> str:
    """Physical name of tablename's HNSW index for metric_type."""
    return f"idx_{tablename}_{metric_type.value}"


def hnsw_index_ddl(engine: Engine, model_record: EmbeddingModelRecord, index_config: HNSWIndexConfig) -> str:
    """CREATE INDEX statement for model_record's table under index_config."""
    tablename = model_record.storage_identifier
    metric = _HNSW_METRIC[index_config.metric_type]
    table_ref = qualified(engine, tablename, physical_schema=physical_schema_of(engine))
    index_ref = qualified(engine, hnsw_index_name(tablename, index_config.metric_type), physical_schema=None)
    return (
        f"CREATE INDEX {index_ref} "
        f"ON {table_ref} USING HNSW ({EMBEDDING_COLUMN_NAME}) "
        f"WITH (metric = '{metric}', ef_construction = {index_config.ef_construction}, "
        f"M = {index_config.num_neighbors})"
    )


# ---------------------------------------------------------------------------
# DML helpers
# ---------------------------------------------------------------------------


def upsert_embedding_rows(
    session: Session,
    records: Sequence[ConceptEmbeddingRecord],
    embeddings: ndarray,
    registered_table: type[DuckDBVSSEmbeddingTable],
) -> None:
    """Insert or update embedding rows with INSERT ... ON CONFLICT DO UPDATE."""
    from duckdb_sqlalchemy import insert as duckdb_insert

    stmt = duckdb_insert(registered_table)
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
                EMBEDDING_COLUMN_NAME: emb.tolist(),
            }
            for rec, emb in zip(records, embeddings)
        ],
    )


def delete_embedding_rows(
    session: Session,
    embedding_table: type[DuckDBVSSEmbeddingTable],
    concept_ids: Sequence[int],
) -> int:
    """Delete rows for concept_ids; returns the number of rows deleted."""
    from sqlalchemy import delete

    from omop_emb.backends.db_utils import in_values

    columns = inspect(embedding_table).columns
    result = session.execute(
        delete(embedding_table).where(
            in_values(columns.concept_id, concept_ids, dialect=Dialect.DUCKDB)
        )
    )
    return int(result.rowcount)  # ty: ignore[unresolved-attribute]


def database_block_bytes(engine: Engine) -> int:
    """Blocks DuckDB has allocated on disk for engine's database, in bytes.

    ``0`` for an in-memory database, which has no on-disk footprint.
    """
    with engine.connect() as conn:
        row = conn.execute(text("PRAGMA database_size")).mappings().first()
    assert row is not None
    return int(row["total_blocks"]) * int(row["block_size"])


def q_all_concept_ids(embedding_table: type[DuckDBVSSEmbeddingTable]) -> Select:
    """Build a SELECT for all concept IDs in an embedding table."""
    return select(embedding_table.concept_id)


def q_embedding_count_by_vocabulary(embedding_table: type[DuckDBVSSEmbeddingTable]) -> Select:
    """Build a SELECT for stored embedding counts grouped by vocabulary_id."""
    return select(
        embedding_table.vocabulary_id, func.count(embedding_table.concept_id)
    ).group_by(embedding_table.vocabulary_id)


def query_embeddings_by_ids(
    session: Session,
    embedding_table: type[DuckDBVSSEmbeddingTable],
    concept_ids: Sequence[int],
) -> dict[int, ndarray]:
    """Fetch float32 embedding vectors for concept_ids, keyed by concept ID; absent IDs are omitted."""
    from omop_emb.backends.db_utils import in_values

    columns = inspect(embedding_table).columns
    rows = session.execute(
        select(columns.concept_id, columns[EMBEDDING_COLUMN_NAME]).where(
            in_values(columns.concept_id, concept_ids, dialect=Dialect.DUCKDB)
        )
    ).all()
    return {int(row[0]): np.asarray(row[1], dtype=np.float32) for row in rows}


# ---------------------------------------------------------------------------
# ANN query
# ---------------------------------------------------------------------------


def get_distance(
    embedding_col: ColumnElement,
    query_vector: list[float],
    metric: MetricType,
    dimensions: int,
) -> ColumnElement:
    """Return a SQLAlchemy distance expression for the given metric.

    Parameters
    ----------
    embedding_col : ColumnElement
        The embedding column to compare against, either the mapped table's
        own column or a CTE's projection of it.
    query_vector : list[float]
    metric : MetricType
    dimensions : int
        Embedding dimensionality; used to cast the bound query vector to the
        same ``FLOAT[N]`` array type as the stored column.

    Raises
    ------
    ValueError
        If ``metric`` is not supported by DuckDB VSS.
    """
    func_name = _DISTANCE_FUNC.get(metric)
    if func_name is None:
        raise ValueError(
            f"DuckDB VSS does not support metric '{metric.value}'. "
            f"Supported: {[m.value for m in _DISTANCE_FUNC]}"
        )
    query_param = cast(query_vector, FixedArray(REAL, dimensions))
    return getattr(func, func_name)(embedding_col, query_param)


def _build_filtered_exact_scan_stmt(
    embedding_table: type[DuckDBVSSEmbeddingTable],
    query_vector: list[float],
    metric_type: MetricType,
    k: int,
    dimensions: int,
    concept_filter: EmbeddingConceptFilter,
) -> Select:
    """Exact-scan KNN over only the rows satisfying concept_filter.

    Materializing the filtered rows into a CTE first stops DuckDB's optimizer
    from rewriting the query into an ``HNSW_INDEX_SCAN``. Without this, the filter
    is applied after the index already picked its candidates, resulting in less
    or no rows returned.

    Cost scales with the size of the filtered set.

    Notes
    -----
    Alternative: https://duckdb.org/community_extensions/extensions/hnsw_acorn

    """
    columns = inspect(embedding_table).columns
    base = select(
        embedding_table.concept_id,
        embedding_table.domain_id,
        embedding_table.vocabulary_id,
        embedding_table.is_standard,
        embedding_table.is_valid,
        getattr(embedding_table, EMBEDDING_COLUMN_NAME),
    )
    base = apply_concept_filter_where(base, columns, concept_filter, dialect=Dialect.DUCKDB)
    filtered = base.cte("filtered_embeddings").prefix_with("MATERIALIZED")
    distance = get_distance(
        filtered.c[EMBEDDING_COLUMN_NAME], query_vector, metric_type, dimensions
    ).label("distance")
    return (
        select(
            filtered.c.concept_id,
            distance,
            filtered.c.domain_id,
            filtered.c.vocabulary_id,
            filtered.c.is_standard,
            filtered.c.is_valid,
        )
        .order_by(distance)
        .limit(k)
    )


def _build_knn_stmt(
    embedding_table: type[DuckDBVSSEmbeddingTable],
    query_vector: list[float],
    metric_type: MetricType,
    k: int,
    dimensions: int,
    concept_filter: EmbeddingConceptFilter | None = None,
    exact_scan_if_filtered: bool = False,
) -> Select:
    """Build a single-vector KNN statement. Internal; see :func:`query_knn_batch`.

    Parameters
    ----------
    embedding_table : type[DuckDBVSSEmbeddingTable]
        The mapped table class to query.
    query_vector : list[float]
        The query vector to find nearest neighbors for.
    metric_type : MetricType
        The distance metric to use for the KNN query.
    k : int
        The number of nearest neighbors to return.
    dimensions : int
        The dimensionality of the embedding vectors.
    concept_filter : EmbeddingConceptFilter | None, optional
        If provided, only return neighbors whose concept IDs satisfy this filter.
    exact_scan_if_filtered : bool, optional
        If ``True`` and *concept_filter* is provided, route the query through
        :func:`_build_filtered_exact_scan_stmt` to avoid a DuckDB VSS bug
        that can return fewer than ``k`` rows on a filtered query with HNSW
        index.
    """
    if concept_filter is not None and exact_scan_if_filtered:
        return _build_filtered_exact_scan_stmt(
            embedding_table, query_vector, metric_type, k, dimensions, concept_filter
        )
    embedding_col = getattr(embedding_table, EMBEDDING_COLUMN_NAME)
    distance = get_distance(embedding_col, query_vector, metric_type, dimensions).label("distance")
    stmt = (
        select(
            embedding_table.concept_id,
            distance,
            embedding_table.domain_id,
            embedding_table.vocabulary_id,
            embedding_table.is_standard,
            embedding_table.is_valid,
        )
        .order_by(distance)
        .limit(k)
    )
    if concept_filter is not None:
        stmt = apply_concept_filter_where(
            stmt, inspect(embedding_table).columns, concept_filter, dialect=Dialect.DUCKDB
        )
    return stmt


def query_knn_batch(
    session: Session,
    embedding_table: type[DuckDBVSSEmbeddingTable],
    query_vectors: Sequence[ndarray],
    metric_type: MetricType,
    k: int,
    dimensions: int,
    concept_filter: EmbeddingConceptFilter | None = None,
    exact_scan_if_filtered: bool = False,
) -> list[Sequence[Row]]:
    """Run KNN queries against a DuckDB VSS table, one per vector in *query_vectors*.

    Returns
    -------
    list[Sequence[Row]]
        One row sequence per query vector, ordered by distance ascending.
        DuckDB has no multi-vector batched form (unlike pgvector's lateral
        join), so each vector is its own statement.
    """
    results: list[Sequence[Row]] = []
    for query_vector in query_vectors:
        stmt = _build_knn_stmt(
            embedding_table,
            query_vector.tolist(),
            metric_type,
            k,
            dimensions,
            concept_filter,
            exact_scan_if_filtered,
        )
        results.append(session.execute(stmt).all())
    return results


def query_concept_ids_matching_filter(
    session: Session,
    embedding_table: type[DuckDBVSSEmbeddingTable],
    concept_filter: EmbeddingConceptFilter,
) -> set[int]:
    """Return every ``concept_id`` satisfying *concept_filter*."""
    stmt = select(embedding_table.concept_id)
    stmt = apply_concept_filter_where(
        stmt, inspect(embedding_table).columns, concept_filter, dialect=Dialect.DUCKDB
    )
    rows = session.execute(stmt).all()
    return {int(row[0]) for row in rows}


def query_concept_filter_metadata(
    session: Session,
    embedding_table: type[DuckDBVSSEmbeddingTable],
    concept_filter: EmbeddingConceptFilter,
) -> Sequence[Row]:
    """Return filter metadata columns (raw rows) for every concept ID
    satisfying concept_filter."""
    stmt = select(
        embedding_table.concept_id,
        embedding_table.domain_id,
        embedding_table.vocabulary_id,
        embedding_table.is_standard,
        embedding_table.is_valid,
    )
    stmt = apply_concept_filter_where(
        stmt, inspect(embedding_table).columns, concept_filter, dialect=Dialect.DUCKDB
    )
    return session.execute(stmt).all()


def duckdbvss_embedding_table_descriptor(model_record: EmbeddingModelRecord) -> type[DuckDBVSSEmbeddingTable]:
    """Return the SQLAlchemy ORM class descriptor for a DuckDB VSS embedding table.

    Cached by (tablename, dimensions): a repeated call for the same table
    reuses the one mapped class instead of building another.
    """
    return _cached_duckdbvss_embedding_table_descriptor(model_record.storage_identifier, model_record.dimensions)


@functools.cache
def _cached_duckdbvss_embedding_table_descriptor(
    tablename: str, dimensions: int
) -> type[DuckDBVSSEmbeddingTable]:
    descriptor_registry = registry(metadata=MetaData())
    emb_col = mapped_column(FixedArray(REAL, dimensions), nullable=False)
    descriptor = type(
        f"DuckDBVSSEmbedding_{tablename}",
        (ConceptEmbeddingMixin,),
        {
            "__tablename__": tablename,
            "__table_args__": {"schema": Role.PRIMARY.value},
            "__module__": __name__,
            EMBEDDING_COLUMN_NAME: emb_col,
        },
    )
    return typing.cast(type[DuckDBVSSEmbeddingTable], descriptor_registry.mapped(descriptor))
