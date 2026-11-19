from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from enum import Enum
from types import UnionType
from typing import Any, Self, get_args, get_origin, get_type_hints

from omop_emb.config import IndexType, MetricType, VectorPrecision, parse_index_type

# ----------------------------------------------------------------------------
# Resevered metadata keys
# ----------------------------------------------------------------------------
# Populate if there are keys that should be reserved in the metadata dict
RESERVED_METADATA_KEYS: frozenset[str] = frozenset({})

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


def _enum_type_of(field_type: Any) -> type[Enum] | None:
    """Return the Enum class a field is annotated with, looking through ``X | None``."""
    candidates = get_args(field_type) if get_origin(field_type) is UnionType else (field_type,)
    for candidate in candidates:
        if isinstance(candidate, type) and issubclass(candidate, Enum):
            return candidate
    return None


@dataclass(kw_only=True, frozen=True)
class IndexConfig(ABC):
    """Abstract base class for index configurations.

    Attributes
    ----------
    index_type : IndexType
        Index structure (FLAT or HNSW).
    metric_type : MetricType or None
        Distance metric the model is locked to. ``None`` for FLAT without a
        metric, which leaves the metric to the query.
    index_precision : VectorPrecision or None
        Precision the index holds its vectors in. ``None`` follows the stored
        column. ``FLOAT16`` over a float32 column builds a half-size index
        over a cast and keeps the full vectors in the table. Which index types
        accept an explicit value depends on the backend, see
        ``BackendCapabilities.index_precisions``.
    """

    index_type: IndexType
    metric_type: MetricType | None
    index_precision: VectorPrecision | None = None

    def __post_init__(self) -> None:
        self._validate_metric_type_after_init()

    @abstractmethod
    def _validate_metric_type_after_init(self) -> None:
        """Raise ``ValueError`` if ``metric_type`` is invalid for this index."""

    def effective_precision(self, storage_precision: VectorPrecision) -> VectorPrecision:
        """Precision the index holds its vectors in.

        Parameters
        ----------
        storage_precision : VectorPrecision
            Precision of the stored column.

        Returns
        -------
        VectorPrecision
            ``index_precision``, or ``storage_precision`` when it is ``None``.
        """
        return self.index_precision or storage_precision

    def casts_storage(self, storage_precision: VectorPrecision) -> bool:
        """Whether the index holds the stored vectors in a different precision than the column.

        True for a float16 index over a float32 column. Queries must then
        compute their distance on the vectors cast to the index precision,
        or the database cannot use the index.

        Parameters
        ----------
        storage_precision : VectorPrecision
            Precision of the stored column.
        """
        return self.effective_precision(storage_precision) != storage_precision

    # TODO: Validate index_config parameters against backend capabilities before
    #       saving to the registry. Would require passing the backend into
    #       IndexConfig, which overlaps with EmbeddingModelRecord.

    def to_dict(self) -> dict[str, Any]:
        """Serialise this config to a plain dict suitable for JSON storage.

        Returns
        -------
        dict[str, Any]
            ``dataclasses.asdict`` output. Enum values are stored as their
            Python objects; the ORM JSON encoder converts them to strings.

        Raises
        ------
        TypeError
            If called on a non-dataclass instance.
        """
        if not is_dataclass(self):
            raise TypeError(
                f"to_dict() must be called on a dataclass instance, not {type(self).__name__}."
            )
        return asdict(self)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> Self:
        """Instantiate from keyword arguments, silently ignoring unknown keys.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments matching the dataclass fields. Unknown keys and
            explicit ``None`` values are discarded, so unset fields keep the
            dataclass's own default instead of being overridden with ``None``.

        Returns
        -------
        Self
            A new instance of the concrete subclass.

        Raises
        ------
        TypeError
            If called on the abstract base class or a non-dataclass.
        """
        if not is_dataclass(cls):
            raise TypeError(f"Must be called on a dataclass, not {cls.__name__}.")
        if inspect.isabstract(cls):
            raise TypeError(
                f"Cannot instantiate abstract class {cls.__name__} directly. Call on a subclass."
            )
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in kwargs.items() if k in known and v is not None})

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Self:
        """Reconstruct a config from the raw dict stored in the ORM column.

        Parameters
        ----------
        config_dict : dict[str, Any]
            The dict previously produced by :meth:`to_dict`. Enum fields may
            be stored as string values and are coerced back to their types.
            Fields with a default may be absent, as in records written before
            the field existed.

        Returns
        -------
        Self
            A new instance of the concrete subclass.

        Raises
        ------
        TypeError
            If called on the abstract base class or a non-dataclass.
        ValueError
            If a field without a default is missing or an enum value is
            unrecognised.

        Notes
        -----
        Use this method to reconstruct an ``IndexConfig`` from the ORM
        ``index_config`` JSON column.
        """
        if not is_dataclass(cls):
            raise TypeError(f"Must be called on a dataclass, not {cls.__name__}.")
        if inspect.isabstract(cls):
            raise TypeError(
                f"Cannot instantiate abstract class {cls.__name__} directly. Call on a subclass."
            )
        type_hints = get_type_hints(cls)
        init_params: dict[str, Any] = {}
        for field in fields(cls):
            if field.name not in config_dict:
                if field.default is MISSING:
                    raise ValueError(f"Strict check failed: '{field.name}' missing in config dict.")
                continue
            value = config_dict[field.name]
            field_type = _enum_type_of(type_hints.get(field.name))
            if value is not None and field_type is not None:
                try:
                    value = field_type(value)
                except ValueError:
                    raise ValueError(f"Invalid enum value '{value}' for {field_type.__name__}.")
            init_params[field.name] = value
        return cls(**init_params)


# ---------------------------------------------------------------------------
# Concrete configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class FlatIndexConfig(IndexConfig):
    """Configuration for a flat (exact scan) embedding table.

    Attributes
    ----------
    index_type : IndexType
        Always ``IndexType.FLAT``.
    metric_type : MetricType or None
        Backends that fix the metric on the table (sqlite-vec) lock the model
        to it and default to cosine. Backends that compute distances from the
        raw vectors (pgvector) drop it, so queries choose the metric.
    """

    index_type: IndexType = IndexType.FLAT
    metric_type: MetricType | None = None

    def _validate_metric_type_after_init(self) -> None:
        """FLAT accepts any metric or none."""


@dataclass(frozen=True, kw_only=True)
class HNSWIndexConfig(IndexConfig):
    """Configuration for an HNSW approximate nearest-neighbour index.

    Attributes
    ----------
    index_type : IndexType
        Always ``IndexType.HNSW``.
    metric_type : MetricType
        Distance metric locked to this index. Required; cannot be ``None``.
    num_neighbors : int
        Number of bi-directional links per node (``M`` in the HNSW paper).
        Default ``32``.
    ef_search : int
        Size of the dynamic candidate list during search. Default ``200``.
    ef_construction : int
        Size of the dynamic candidate list during index construction.
        Default ``200``.
    max_scan_tuples : int or None
        Upper bound on the rows one filtered query may read from the index
        (pgvector ``hnsw.max_scan_tuples``). ``None`` keeps the server default
        of 20,000. Only pgvector supports it.
    Notes
    -----
    Supported by pgvector, DuckDB VSS, and FAISS backends. sqlite-vec does
    not support HNSW.
    """

    metric_type: MetricType
    num_neighbors: int = 32
    ef_search: int = 200
    ef_construction: int = 200
    max_scan_tuples: int | None = None
    index_type: IndexType = IndexType.HNSW

    def _validate_metric_type_after_init(self) -> None:
        if self.metric_type is None:
            raise ValueError("HNSW index requires a metric_type (e.g. MetricType.COSINE).")
        for name in ("num_neighbors", "ef_search", "ef_construction", "max_scan_tuples"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"HNSW '{name}' must be a positive integer, got {value}.")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def index_config_from_index_type(index_type: IndexType, **kwargs: Any) -> IndexConfig:
    """Build an ``IndexConfig`` for a given index type from raw kwargs.

    Parameters
    ----------
    index_type : IndexType
        Target index type.
    **kwargs : Any
        Parameters forwarded to the concrete config class. Unknown keys are
        silently ignored.

    Returns
    -------
    IndexConfig
        A concrete ``FlatIndexConfig`` or ``HNSWIndexConfig``.

    Raises
    ------
    ValueError
        If ``index_type`` has no registered ``IndexConfig`` subclass.
    """
    index_type = parse_index_type(index_type)
    if index_type == IndexType.FLAT:
        return FlatIndexConfig.from_kwargs(
            metric_type=kwargs.get("metric_type"), index_precision=kwargs.get("index_precision")
        )
    if index_type == IndexType.HNSW:
        return HNSWIndexConfig.from_kwargs(**kwargs)
    raise ValueError(f"No IndexConfig defined for index type {index_type!r}.")


def index_config_from_dict(
    index_type: IndexType | str, config_dict: dict[str, Any] | None
) -> IndexConfig:
    """Reconstruct an ``IndexConfig`` from its serialised ``to_dict()`` form.

    Used for the registry's ORM columns, bundle attributes and FAISS sidecars.
    Enum fields are coerced from their string values.

    Parameters
    ----------
    index_type : IndexType or str
        The serialised index type.
    config_dict : dict[str, Any] or None
        The ``to_dict()`` output. May be ``None`` or empty for FLAT.

    Returns
    -------
    IndexConfig
        A concrete ``FlatIndexConfig`` or ``HNSWIndexConfig``.

    Raises
    ------
    ValueError
        If ``index_type`` has no registered config class, or if HNSW
        reconstruction fails.
    """
    index_type = parse_index_type(index_type)
    if index_type == IndexType.FLAT:
        metric = (config_dict or {}).get("metric_type")
        return FlatIndexConfig(metric_type=MetricType(metric) if metric else None)
    if index_type == IndexType.HNSW:
        if not config_dict:
            raise ValueError(
                "HNSW index_config column is empty; cannot reconstruct HNSWIndexConfig."
            )
        return HNSWIndexConfig.from_dict(config_dict)
    raise ValueError(f"No IndexConfig defined for index type {index_type!r}.")
