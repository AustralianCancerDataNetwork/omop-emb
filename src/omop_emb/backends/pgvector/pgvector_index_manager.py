"""pgvector index manager.

Each ``PGVectorBaseIndexManager`` subclass manages SQL indices for one
(model, index_type) pair.  Unlike FAISS, Postgres maintains index content
automatically on every INSERT, so there is no ``add()`` method.  The manager
only handles the DDL lifecycle: create, drop, and rebuild.
"""

from __future__ import annotations

import abc
import logging
from typing import Generic, TypeVar

from sqlalchemy import Engine, inspect, text

from omop_emb.config import IndexType, MetricType
from omop_emb.backends.index_config import IndexConfig, FlatIndexConfig, HNSWIndexConfig
from omop_emb.backends.index_utils import BaseIndexManager

logger = logging.getLogger(__name__)

C = TypeVar("C", bound=IndexConfig)

class PGVectorBaseIndexManager(BaseIndexManager[C], Generic[C]):
    """pgvector index manager base class.

    Manages SQL index DDL (``CREATE INDEX``, ``DROP INDEX``) for one model's
    embedding table.  Postgres updates the index automatically on INSERT, so
    no explicit vector-addition step is needed.

    Parameters
    ----------
    engine:
        SQLAlchemy engine connected to the OMOP Postgres database.
    tablename:
        Name of the embedding table this manager operates on.
    embedding_column:
        Name of the vector column inside *tablename*.
    index_config:
        Configuration object for this index type.
    """

    def __init__(
        self,
        engine: Engine,
        tablename: str,
        embedding_column: str,
        index_config: C,
    ):
        if index_config.index_type != self.supported_index_type:
            raise ValueError(
                f"index_config has index_type {index_config.index_type!r} but this manager "
                f"supports {self.supported_index_type!r}."
            )
        self._engine = engine
        self._tablename = tablename
        self._embedding_column = embedding_column
        self._index_config = index_config

    # ------------------------------------------------------------------
    # BaseIndexManager identity
    # ------------------------------------------------------------------

    @property
    def index_config(self) -> C:
        return self._index_config

    # ------------------------------------------------------------------
    # BaseIndexManager lifecycle
    # ------------------------------------------------------------------

    def has_index(self, metric_type: MetricType) -> bool:
        """Return True if the SQL index for *metric_type* exists in the database."""
        with self._engine.connect() as conn:
            existing = {
                idx["name"]
                for idx in inspect(conn).get_indexes(self._tablename)
            }
        return self._index_name(metric_type) in existing

    def create_index(self, metric_type: MetricType, **_kwargs) -> None:
        """Create the SQL index for *metric_type* if it does not already exist."""
        if self.has_index(metric_type):
            logger.debug(
                f"pgvector index '{self._index_name(metric_type)}' already exists, skipping creation."
            )
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
        """Drop the SQL index for *metric_type* if it exists."""
        name = self._index_name(metric_type)
        existed = self.has_index(metric_type)
        with self._engine.begin() as conn:
            conn.execute(text(f"DROP INDEX IF EXISTS {name}"))
        if existed:
            logger.info(f"Dropped pgvector index '{name}'.")
        else:
            logger.debug(f"drop_index called for '{name}' but it did not exist; no-op.")

    # rebuild_index uses the base default: drop_index + create_index
    # Postgres rebuilds index content from the table automatically after CREATE INDEX.

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _index_name(self, metric_type: MetricType) -> str:
        return f"idx_{self._tablename}_{metric_type.value}"

    @staticmethod
    def _ops_for_metric(metric_type: MetricType) -> str:
        """Return the pgvector operator class for *metric_type*.
        Taken from here: https://github.com/pgvector/pgvector#hnsw
        """
        if metric_type == MetricType.COSINE:
            return "vector_cosine_ops"
        elif metric_type == MetricType.L2:
            return "vector_l2_ops"
        elif metric_type == MetricType.L1:
            return "vector_l1_ops"
        elif metric_type == MetricType.JACCARD:
            return "bit_jaccard_ops"
        elif metric_type == MetricType.HAMMING:
            return "bit_hamming_ops"
        else:
            raise ValueError(f"MetricType '{metric_type.value}' is not supported for pgvector vector indices.")

    @abc.abstractmethod
    def _create_index_ddl(self, metric_type: MetricType) -> str | None:
        """Return the ``CREATE INDEX`` SQL string, or ``None`` for no-op index types."""
        ...

# ---------------------------------------------------------------------------
# Concrete pgvector index managers
# ---------------------------------------------------------------------------

class PGVectorFlatIndexManager(PGVectorBaseIndexManager[FlatIndexConfig]):
    """No-op manager for FLAT index type.

    pgvector FLAT search is a sequential scan — no SQL index is created.
    This class exists for structural symmetry so the backend can always
    hold an index manager regardless of index type.
    """

    @property
    def supported_index_type(self) -> IndexType:
        return IndexType.FLAT

    def has_index(self, metric_type: MetricType) -> bool:
        return True  # sequential scan is always available

    def create_index(self, metric_type: MetricType, **_kwargs) -> None:
        pass  # no SQL index needed

    def drop_index(self, metric_type: MetricType) -> None:
        pass  # nothing to drop

    def _create_index_ddl(self, metric_type: MetricType) -> str | None:
        return None


class PGVectorHNSWIndexManager(PGVectorBaseIndexManager[HNSWIndexConfig]):
    """HNSW index manager for pgvector.

    Creates one ``USING hnsw`` SQL index per metric type, each bound to the
    appropriate pgvector operator class.  Parameters (``m``, ``ef_construction``)
    are read from ``HNSWIndexConfig``.

    Notes
    -----
    ``ef_search`` (query-time parameter) is set per-session via
    ``SET hnsw.ef_search = <value>`` in the backend's ``get_nearest_concepts``
    rather than at index-creation time.
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
