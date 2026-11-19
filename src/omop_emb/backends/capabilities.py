from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from omop_emb.backends.index_config import HNSWIndexConfig, IndexConfig
from omop_emb.config import BackendType, IndexType, MetricType, VectorPrecision
from omop_emb.utils.errors import UnsupportedIndexConfigError


@dataclass(frozen=True, kw_only=True)
class StorageCapabilities:
    """What a backend can store.

    Attributes
    ----------
    precisions : tuple[VectorPrecision, ...]
        Precisions the stored embedding column can use.
    max_dimensions : int or None
        Largest dimensionality a model can be registered with. ``None`` for no limit.
    """

    precisions: tuple[VectorPrecision, ...] = (VectorPrecision.FLOAT32,)
    max_dimensions: int | None = None

    def validate(
        self, *, backend: BackendType, dimensions: int, storage_precision: VectorPrecision
    ) -> None:
        """Raise ``UnsupportedIndexConfigError`` if a model cannot be stored as described.

        Parameters
        ----------
        backend : BackendType
            Backend the capabilities belong to. Named in the error.
        dimensions : int
            Embedding dimensionality.
        storage_precision : VectorPrecision
            Precision of the stored column.

        Raises
        ------
        UnsupportedIndexConfigError
            If the precision or the dimensionality is unsupported.
        """
        if storage_precision not in self.precisions:
            raise UnsupportedIndexConfigError(
                f"Storage precision '{storage_precision.value}' is not supported by backend "
                f"'{backend.value}'. Supported: {[p.value for p in self.precisions]}."
            )
        if self.max_dimensions is not None and dimensions > self.max_dimensions:
            raise UnsupportedIndexConfigError(
                f"Backend '{backend.value}' supports at most {self.max_dimensions:,} dimensions, "
                f"but the model has {dimensions:,}."
            )


@dataclass(frozen=True, kw_only=True)
class IndexCapabilities:
    """What a backend supports for one index type.

    Attributes
    ----------
    metrics : tuple[MetricType, ...]
        Metrics the index supports.
    precisions : tuple[VectorPrecision, ...]
        Explicit index precisions the index accepts. Empty when the index always
        follows the stored column, as FLAT does.
    max_dimensions : Mapping[VectorPrecision, int]
        Largest dimensionality the index supports per index precision. A
        precision without an entry has no limit.
    max_scan_tuples : bool
        Whether ``HNSWIndexConfig.max_scan_tuples`` has an effect.
    """

    metrics: tuple[MetricType, ...]
    precisions: tuple[VectorPrecision, ...] = ()
    max_dimensions: Mapping[VectorPrecision, int] = field(default_factory=dict)
    max_scan_tuples: bool = False

    def validate(
        self,
        index_config: IndexConfig,
        *,
        backend: BackendType,
        dimensions: int,
        storage_precision: VectorPrecision,
    ) -> None:
        """Raise ``UnsupportedIndexConfigError`` if the index cannot honour ``index_config``.

        Parameters
        ----------
        index_config : IndexConfig
            Configuration about to be built or saved.
        backend : BackendType
            Backend the capabilities belong to. Named in the error.
        dimensions : int
            Dimensionality of the model's embeddings.
        storage_precision : VectorPrecision
            Precision of the model's stored column.

        Raises
        ------
        UnsupportedIndexConfigError
            If the metric, a set HNSW option or the precision is unsupported, or
            the index cannot be built over this precision and dimensionality.
        """
        metric_type = index_config.metric_type
        if metric_type is not None and metric_type not in self.metrics:
            raise UnsupportedIndexConfigError(
                f"Metric '{metric_type.value}' is not supported by backend "
                f"'{backend.value}' with index type '{index_config.index_type.value}'."
            )
        if (
            isinstance(index_config, HNSWIndexConfig)
            and index_config.max_scan_tuples is not None
            and not self.max_scan_tuples
        ):
            raise UnsupportedIndexConfigError(
                f"max_scan_tuples is not supported by backend '{backend.value}'."
            )
        self._validate_precision(
            index_config,
            backend=backend,
            dimensions=dimensions,
            storage_precision=storage_precision,
        )

    def _validate_precision(
        self,
        index_config: IndexConfig,
        *,
        backend: BackendType,
        dimensions: int,
        storage_precision: VectorPrecision,
    ) -> None:
        requested = index_config.index_precision
        if requested is not None:
            if requested not in self.precisions:
                raise UnsupportedIndexConfigError(
                    f"Index precision '{requested.value}' is not supported for "
                    f"'{index_config.index_type.value}' indexes by backend '{backend.value}'. "
                    f"Supported: {[precision.value for precision in self.precisions]}."
                )
            if (
                requested == VectorPrecision.FLOAT32
                and storage_precision == VectorPrecision.FLOAT16
            ):
                raise UnsupportedIndexConfigError(
                    "Index precision 'float32' needs a float32 column, but the model "
                    "is stored as 'float16'."
                )
        effective = index_config.effective_precision(storage_precision)
        limit = self.max_dimensions.get(effective)
        if limit is None or dimensions <= limit:
            return
        half = VectorPrecision.FLOAT16
        half_fits = (
            effective != half
            and half in self.precisions
            and dimensions <= self.max_dimensions.get(half, dimensions)
        )
        hint = " Use index precision 'float16' or a FLAT index." if half_fits else ""
        raise UnsupportedIndexConfigError(
            f"Backend '{backend.value}' indexes at most {limit:,} dimensions at precision "
            f"'{effective.value}', but the model has {dimensions:,}.{hint}"
        )


@dataclass(frozen=True, kw_only=True)
class BackendCapabilities:
    """Storage and index options one backend supports.

    Attributes
    ----------
    backend_type : BackendType
        Backend the capabilities describe. Named in validation errors.
    indices : Mapping[IndexType, IndexCapabilities]
        Supported index types, each with what it supports.
    storage : StorageCapabilities
        What the backend can store.
    """

    backend_type: BackendType
    indices: Mapping[IndexType, IndexCapabilities]
    storage: StorageCapabilities = field(default_factory=StorageCapabilities)

    @property
    def index_types(self) -> tuple[IndexType, ...]:
        """Supported index types."""
        return tuple(self.indices)

    def supports_index_type(self, index_type: IndexType) -> bool:
        """Return whether ``index_type`` is supported."""
        return index_type in self.indices

    def supports_metric(self, index_type: IndexType, metric_type: MetricType) -> bool:
        """Return whether ``metric_type`` is supported with ``index_type``."""
        index = self.indices.get(index_type)
        return index is not None and metric_type in index.metrics

    def index(self, index_type: IndexType) -> IndexCapabilities:
        """The capabilities of ``index_type``.

        Raises
        ------
        UnsupportedIndexConfigError
            If the backend does not support ``index_type``.
        """
        try:
            return self.indices[index_type]
        except KeyError:
            raise UnsupportedIndexConfigError(
                f"Index type '{index_type.value}' is not supported by "
                f"backend '{self.backend_type.value}'. "
                f"Supported: {[index.value for index in self.index_types]}."
            ) from None

    def require_max_scan_tuples(self, index_type: IndexType = IndexType.HNSW) -> None:
        """Raise ``UnsupportedIndexConfigError`` unless ``max_scan_tuples`` is supported."""
        if not self.index(index_type).max_scan_tuples:
            raise UnsupportedIndexConfigError(
                f"max_scan_tuples is not supported by backend '{self.backend_type.value}'."
            )

    def validate_registration(self, *, dimensions: int, storage_precision: VectorPrecision) -> None:
        """Raise ``UnsupportedIndexConfigError`` if a model cannot be stored as described."""
        self.storage.validate(
            backend=self.backend_type, dimensions=dimensions, storage_precision=storage_precision
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
        """
        self.index(index_config.index_type).validate(
            index_config,
            backend=self.backend_type,
            dimensions=dimensions,
            storage_precision=storage_precision,
        )
