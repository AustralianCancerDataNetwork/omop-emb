from __future__ import annotations

import abc
from typing import Generic, TypeVar

from omop_emb.config import IndexType, MetricType
from omop_emb.backends.index_config import IndexConfig

C = TypeVar("C", bound=IndexConfig)


class BaseIndexManager(abc.ABC, Generic[C]):
    """Shared lifecycle interface for backend index managers.

    One manager instance represents a single (model, index_type) pair and
    manages metric-specific indices under that pair.
    """

    @property
    @abc.abstractmethod
    def index_config(self) -> C:
        """The configuration object that parameterises this manager."""

    @property
    @abc.abstractmethod
    def supported_index_type(self) -> IndexType:
        """The single ``IndexType`` this manager handles."""

    @abc.abstractmethod
    def has_index(self, metric_type: MetricType) -> bool:
        """Return ``True`` if a usable index exists for *metric_type*."""

    @abc.abstractmethod
    def create_index(self, metric_type: MetricType, **kwargs) -> None:
        """Create the index for *metric_type* if it does not already exist (idempotent)."""

    @abc.abstractmethod
    def drop_index(self, metric_type: MetricType) -> None:
        """Remove the index for *metric_type* (idempotent)."""

    def load_or_create(self, metric_type: MetricType, **kwargs) -> None:
        """Create only if absent.  pgvector uses this as a readable alias for ``create_index``."""
        if not self.has_index(metric_type):
            self.create_index(metric_type, **kwargs)

    def rebuild_index(self, metric_type: MetricType, **kwargs) -> None:
        """Drop then recreate the index for *metric_type*."""
        self.drop_index(metric_type)
        self.create_index(metric_type, **kwargs)

    def train(self, **kwargs) -> None:
        """Optional training step for IVF-style indices.  No-op by default."""
