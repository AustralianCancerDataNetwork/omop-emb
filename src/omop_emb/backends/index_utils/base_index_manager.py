from __future__ import annotations

import abc
from typing import Generic, TypeVar

from omop_emb.config import IndexType, MetricType
from omop_emb.backends.index_config import IndexConfig

C = TypeVar("C", bound=IndexConfig)


class BaseIndexManager(abc.ABC, Generic[C]):
    """Shared lifecycle interface for backend index managers.

    Covers creation, existence checks, removal, and rebuilding of indices.
    Backend-specific operations (vector insertion for FAISS, SQL DDL for
    pgvector) live in the respective subclasses.

    One manager instance represents a single (model, index_type) pair and may
    manage multiple metric-specific indices under that pair.

    Notes
    -----
    ``train()`` is a no-op by default.  IVF managers on both sides override it
    to run the k-means training step before the index can accept vectors.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def index_config(self) -> C:
        """The configuration object that parameterises this manager."""

    @property
    @abc.abstractmethod
    def supported_index_type(self) -> IndexType:
        """The single ``IndexType`` this manager handles."""

    # ------------------------------------------------------------------
    # Index lifecycle (must be implemented by each backend)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def has_index(self, metric_type: MetricType) -> bool:
        """Return ``True`` if a usable index exists for *metric_type*.

        For FAISS this means a ``.faiss`` file is on disk (or loaded in memory).
        For pgvector this means the SQL index row is present in the database.
        """

    @abc.abstractmethod
    def create_index(self, metric_type: MetricType, **kwargs) -> None:
        """Create the index for *metric_type* if it does not already exist.

        Implementations must be idempotent: calling this when the index already
        exists should be a no-op (not an error).
        """

    @abc.abstractmethod
    def drop_index(self, metric_type: MetricType) -> None:
        """Remove the index for *metric_type*.

        Implementations must be idempotent: calling this when no index exists
        should be a no-op (not an error).
        """

    # ------------------------------------------------------------------
    # Convenience methods with sensible defaults
    # ------------------------------------------------------------------

    def load_or_create(self, metric_type: MetricType, **kwargs) -> None:
        """Create the index for *metric_type* only if it does not yet exist.

        FAISS overrides this to also handle the disk-load path and the
        data-stream population path.  pgvector uses the default (``create_index``
        is already idempotent, so this is just a readable alias).
        """
        if not self.has_index(metric_type):
            self.create_index(metric_type, **kwargs)

    def rebuild_index(self, metric_type: MetricType, **kwargs) -> None:
        """Drop and recreate the index for *metric_type*.

        FAISS overrides this to stream vectors from HDF5 after creation.
        pgvector uses the default: ``DROP INDEX`` then ``CREATE INDEX``.
        Postgres rebuilds the index content from the table automatically.
        """
        self.drop_index(metric_type)
        self.create_index(metric_type, **kwargs)

    def train(self, **kwargs) -> None:
        """Optional training step required by IVF-style indices.

        No-op by default.  IVF managers override this to run k-means
        centroid training before any vectors can be added to the index.
        """
