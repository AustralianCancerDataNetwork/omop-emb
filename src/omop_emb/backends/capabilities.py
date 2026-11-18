from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from omop_emb.backends.index_config import HNSWIndexConfig, IndexConfig
from omop_emb.config import BackendType, IndexType, MetricType
from omop_emb.utils.errors import UnsupportedIndexConfigError


@dataclass(frozen=True, kw_only=True)
class BackendCapabilities:
    """Index and storage options one backend supports.

    Attributes
    ----------
    backend_type : BackendType
        Backend the capabilities describe. Named in validation errors.
    indices : Mapping[IndexType, tuple[MetricType, ...]]
        Supported index types, each with the metrics it supports.
    max_scan_tuples : bool
        Whether ``HNSWIndexConfig.max_scan_tuples`` has an effect.
    """

    backend_type: BackendType
    indices: Mapping[IndexType, tuple[MetricType, ...]]
    max_scan_tuples: bool = False

    @property
    def index_types(self) -> tuple[IndexType, ...]:
        """Supported index types."""
        return tuple(self.indices)

    def supports_index_type(self, index_type: IndexType) -> bool:
        """Return whether ``index_type`` is supported."""
        return index_type in self.indices

    def supports_metric(self, index_type: IndexType, metric_type: MetricType) -> bool:
        """Return whether ``metric_type`` is supported with ``index_type``."""
        return metric_type in self.indices.get(index_type, ())

    def require_max_scan_tuples(self) -> None:
        """Raise ``UnsupportedIndexConfigError`` unless ``max_scan_tuples`` is supported."""
        if not self.max_scan_tuples:
            raise UnsupportedIndexConfigError(
                f"max_scan_tuples is not supported by backend '{self.backend_type.value}'."
            )

    def validate_index_config(self, index_config: IndexConfig) -> None:
        """Raise ``UnsupportedIndexConfigError`` if the backend cannot honour ``index_config``.

        Parameters
        ----------
        index_config : IndexConfig
            Configuration about to be built or saved.

        Raises
        ------
        UnsupportedIndexConfigError
            If the index type, its metric, or a set HNSW option is unsupported.
        """
        backend = self.backend_type.value
        if not self.supports_index_type(index_config.index_type):
            raise UnsupportedIndexConfigError(
                f"Index type '{index_config.index_type.value}' is not supported by "
                f"backend '{backend}'. "
                f"Supported: {[index.value for index in self.index_types]}."
            )
        if index_config.metric_type is not None and not self.supports_metric(
            index_config.index_type, index_config.metric_type
        ):
            raise UnsupportedIndexConfigError(
                f"Metric '{index_config.metric_type.value}' is not supported by backend "
                f"'{backend}' with index type '{index_config.index_type.value}'."
            )
        if isinstance(index_config, HNSWIndexConfig) and index_config.max_scan_tuples is not None:
            self.require_max_scan_tuples()
