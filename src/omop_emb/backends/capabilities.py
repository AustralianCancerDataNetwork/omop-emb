from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from omop_emb.backends.index_config import HNSWIndexConfig, IndexConfig
from omop_emb.config import BackendType, IndexType, MetricType, VectorPrecision
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
    storage_precisions : tuple[VectorPrecision, ...]
        Precisions the stored embedding column can use.
    index_precisions : Mapping[IndexType, tuple[VectorPrecision, ...]]
        Explicit index precisions each index type accepts. An index type
        without an entry accepts none, so it always follows the stored column.
    max_dimensions : int or None
        Largest dimensionality a model can be registered with. ``None`` for no limit.
    max_indexed_dimensions : Mapping[VectorPrecision, int]
        Largest dimensionality an index supports per index precision. FLAT
        has no index and is not limited. A precision without an entry has no limit.
    """

    backend_type: BackendType
    indices: Mapping[IndexType, tuple[MetricType, ...]]
    max_scan_tuples: bool = False
    storage_precisions: tuple[VectorPrecision, ...] = (VectorPrecision.FLOAT32,)
    index_precisions: Mapping[IndexType, tuple[VectorPrecision, ...]] = field(default_factory=dict)
    max_dimensions: int | None = None
    max_indexed_dimensions: Mapping[VectorPrecision, int] = field(default_factory=dict)

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

    def validate_registration(self, *, dimensions: int, storage_precision: VectorPrecision) -> None:
        """Raise ``UnsupportedIndexConfigError`` if a model cannot be stored as described.

        Parameters
        ----------
        dimensions : int
            Embedding dimensionality.
        storage_precision : VectorPrecision
            Precision of the stored column.

        Raises
        ------
        UnsupportedIndexConfigError
            If the precision or the dimensionality is unsupported.
        """
        backend = self.backend_type.value
        if storage_precision not in self.storage_precisions:
            raise UnsupportedIndexConfigError(
                f"Storage precision '{storage_precision.value}' is not supported by backend "
                f"'{backend}'. Supported: {[p.value for p in self.storage_precisions]}."
            )
        if self.max_dimensions is not None and dimensions > self.max_dimensions:
            raise UnsupportedIndexConfigError(
                f"Backend '{backend}' supports at most {self.max_dimensions:,} dimensions, "
                f"but the model has {dimensions:,}."
            )

    def validate_index_config(
        self,
        index_config: IndexConfig,
        *,
        dimensions: int,
        storage_precision: VectorPrecision,
    ) -> None:
        """Raise ``UnsupportedIndexConfigError`` if the backend cannot honour ``index_config``.

        Parameters
        ----------
        index_config : IndexConfig
            Configuration about to be built or saved.
        dimensions : int
            Dimensionality of the model's embeddings.
        storage_precision : VectorPrecision
            Precision of the model's stored column.

        Raises
        ------
        UnsupportedIndexConfigError
            If the index type, its metric, or a set HNSW option is unsupported,
            or the index cannot be built over this precision and dimensionality.
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
        self._validate_index_precision(
            index_config, dimensions=dimensions, storage_precision=storage_precision
        )

    def _validate_index_precision(
        self,
        index_config: IndexConfig,
        *,
        dimensions: int,
        storage_precision: VectorPrecision,
    ) -> None:
        backend = self.backend_type.value
        index_type = index_config.index_type
        requested = index_config.index_precision
        if requested is not None:
            accepted = self.index_precisions.get(index_type, ())
            if requested not in accepted:
                raise UnsupportedIndexConfigError(
                    f"Index precision '{requested.value}' is not supported for '{index_type.value}' "
                    f"indexes by backend '{backend}'. "
                    f"Supported: {[precision.value for precision in accepted]}."
                )
            if (
                requested == VectorPrecision.FLOAT32
                and storage_precision == VectorPrecision.FLOAT16
            ):
                raise UnsupportedIndexConfigError(
                    "Index precision 'float32' needs a float32 column, but the model "
                    "is stored as 'float16'."
                )
        if index_type == IndexType.FLAT:
            return
        effective = index_config.effective_precision(storage_precision)
        limit = self.max_indexed_dimensions.get(effective)
        if limit is None or dimensions <= limit:
            return
        half = VectorPrecision.FLOAT16
        half_fits = (
            effective != half
            and half in self.index_precisions.get(index_type, ())
            and dimensions <= self.max_indexed_dimensions.get(half, dimensions)
        )
        hint = " Use index precision 'float16' or a FLAT index." if half_fits else ""
        raise UnsupportedIndexConfigError(
            f"Backend '{backend}' indexes at most {limit:,} dimensions at precision "
            f"'{effective.value}', but the model has {dimensions:,}.{hint}"
        )
