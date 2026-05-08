"""pgvector index managers.

Each manager handles the DDL lifecycle (CREATE / DROP / REBUILD) for SQL
indices on one model's embedding table.  Postgres auto-maintains index
content on INSERT, so there is no vector-addition step.

Halfvec support
---------------
``halfvec`` columns require different operator classes than ``vector`` columns.
The correct operator class is selected automatically based on
:class:`~omop_emb.config.VectorColumnType` stored in ``EmbeddingModelRecord``.
"""
from __future__ import annotations

import abc
import logging
from typing import Generic, TypeVar

from sqlalchemy import Engine, inspect, text

from omop_emb.config import IndexType, MetricType, VectorColumnType
from omop_emb.backends.index_config import IndexConfig, FlatIndexConfig, HNSWIndexConfig
from omop_emb.backends.base_index_manager import BaseIndexManager
from omop_emb.utils.embedding_utils import vector_column_type_for_dimensions

logger = logging.getLogger(__name__)

C = TypeVar("C", bound=IndexConfig)


class PGVectorBaseIndexManager(BaseIndexManager[C], Generic[C]):
    """pgvector index manager base class.

    Parameters
    ----------
    emb_engine : Engine
        Engine for the pgvector instance.
    tablename : str
        Name of the embedding table.
    embedding_column : str
        Name of the vector column.
    index_config : C
        Configuration for this index type.
    dimensions : int
        Vector dimensionality. Used to select the correct operator class
        (``vector_*_ops`` vs ``halfvec_*_ops``).
    """

    def __init__(
        self,
        emb_engine: Engine,
        tablename: str,
        embedding_column: str,
        index_config: C,
        dimensions: int,
    ):
        if index_config.index_type != self.supported_index_type:
            raise ValueError(
                f"index_config has index_type {index_config.index_type!r} but "
                f"this manager supports {self.supported_index_type!r}."
            )
        self._engine = emb_engine
        self._tablename = tablename
        self._embedding_column = embedding_column
        self._index_config = index_config
        self._vector_col_type = vector_column_type_for_dimensions(dimensions)

    @property
    def index_config(self) -> C:
        return self._index_config

    # ------------------------------------------------------------------
    # BaseIndexManager lifecycle
    # ------------------------------------------------------------------

    def has_index(self, metric_type: MetricType) -> bool:
        with self._engine.connect() as conn:
            existing = {idx["name"] for idx in inspect(conn).get_indexes(self._tablename)}
        return self._index_name(metric_type) in existing

    def create_index(self, metric_type: MetricType, **_kwargs) -> None:
        if self.has_index(metric_type):
            logger.debug(f"Index '{self._index_name(metric_type)}' already exists, skipping.")
            return
        sql = self._create_index_ddl(metric_type)
        if sql is None:
            return
        with self._engine.begin() as conn:
            conn.execute(text(sql))
        logger.info(
            f"Created pgvector index '{self._index_name(metric_type)}' "
            f"on '{self._tablename}' (metric={metric_type.value})."
        )

    def drop_index(self, metric_type: MetricType) -> None:
        name = self._index_name(metric_type)
        existed = self.has_index(metric_type)
        with self._engine.begin() as conn:
            conn.execute(text(f"DROP INDEX IF EXISTS {name}"))
        if existed:
            logger.info(f"Dropped pgvector index '{name}'.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _index_name(self, metric_type: MetricType) -> str:
        return f"idx_{self._tablename}_{metric_type.value}"

    def _ops_for_metric(self, metric_type: MetricType) -> str:
        """Return the operator class for *metric_type* and the column's vector type."""
        return _OPERATOR_CLASSES[self._vector_col_type][metric_type]

    @abc.abstractmethod
    def _create_index_ddl(self, metric_type: MetricType) -> str | None: ...


# ---------------------------------------------------------------------------
# Operator class lookup table
# ---------------------------------------------------------------------------

_VECTOR_OPS: dict[MetricType, str] = {
    MetricType.COSINE: "vector_cosine_ops",
    MetricType.L2: "vector_l2_ops",
    MetricType.L1: "vector_l1_ops",
    MetricType.HAMMING: "bit_hamming_ops",
    MetricType.JACCARD: "bit_jaccard_ops",
}

_HALFVEC_OPS: dict[MetricType, str] = {
    MetricType.COSINE: "halfvec_cosine_ops",
    MetricType.L2: "halfvec_l2_ops",
    MetricType.L1: "halfvec_l1_ops",
    # HAMMING / JACCARD are bit-type operations -> halfvec doesn't apply.
    MetricType.HAMMING: "bit_hamming_ops",
    MetricType.JACCARD: "bit_jaccard_ops",
}

_OPERATOR_CLASSES: dict[VectorColumnType, dict[MetricType, str]] = {
    VectorColumnType.VECTOR: _VECTOR_OPS,
    VectorColumnType.HALFVEC: _HALFVEC_OPS,
}


# ---------------------------------------------------------------------------
# Concrete managers
# ---------------------------------------------------------------------------

class PGVectorFlatIndexManager(PGVectorBaseIndexManager[FlatIndexConfig]):
    """No-op manager for FLAT search (sequential scan; no SQL index needed)."""

    @property
    def supported_index_type(self) -> IndexType:
        return IndexType.FLAT

    def has_index(self, metric_type: MetricType) -> bool:
        return True  # sequential scan is always available

    def create_index(self, metric_type: MetricType, **_kwargs) -> None:
        pass

    def drop_index(self, metric_type: MetricType) -> None:
        pass

    def _create_index_ddl(self, metric_type: MetricType) -> str | None:
        return None


class PGVectorHNSWIndexManager(PGVectorBaseIndexManager[HNSWIndexConfig]):
    """HNSW index manager for pgvector.

    Creates one ``USING hnsw`` SQL index per metric type.  ``ef_search``
    is applied per-session via ``SET hnsw.ef_search = <value>`` at query
    time rather than at index-creation time.

    Works with both ``vector`` and ``halfvec`` column types. The correct
    operator class is selected automatically.
    """

    @property
    def supported_index_type(self) -> IndexType:
        return IndexType.HNSW

    def _create_index_ddl(self, metric_type: MetricType) -> str:
        ops = self._ops_for_metric(metric_type)
        cfg = self.index_config
        return (
            f"CREATE INDEX {self._index_name(metric_type)} "
            f"ON {self._tablename} "
            f"USING hnsw ({self._embedding_column} {ops}) "
            f"WITH (m = {cfg.num_neighbors}, ef_construction = {cfg.ef_construction})"
        )
