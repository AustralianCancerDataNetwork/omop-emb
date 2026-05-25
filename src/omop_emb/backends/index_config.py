from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Self, cast, get_type_hints

from omop_emb.config import IndexType, MetricType, parse_index_type

# ----------------------------------------------------------------------------
# Resevered metadata keys
# ----------------------------------------------------------------------------
# Populate if there are keys that should be reserved in the metadata dict
RESERVED_METADATA_KEYS: frozenset[str] = frozenset({})

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

@dataclass(kw_only=True, frozen=True)
class IndexConfig(ABC):
    """Abstract base class for index configurations.

    Attributes
    ----------
    index_type : IndexType
        Index structure (FLAT or HNSW).
    metric_type : MetricType or None
        Distance metric locked to this index. ``None`` for FLAT, which
        supports any metric at query time.
    """

    index_type: IndexType
    metric_type: Optional[MetricType]

    def __post_init__(self) -> None:
        self._validate_metric_type_after_init()

    @abstractmethod
    def _validate_metric_type_after_init(self) -> None:
        """Raise ``ValueError`` if ``metric_type`` is invalid for this index."""

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
                f"to_dict() must be called on a dataclass instance, "
                f"not {type(self).__name__}."
            )
        return asdict(self)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> Self:
        """Instantiate from keyword arguments, silently ignoring unknown keys.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments matching the dataclass fields. Unknown keys are
            discarded.

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
                f"Cannot instantiate abstract class {cls.__name__} directly. "
                "Call on a subclass."
            )
        known = {f.name for f in fields(cls)}
        factory = cast(Callable[..., Self], cls)
        return factory(**{k: v for k, v in kwargs.items() if k in known})

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Self:
        """Reconstruct a config from the raw dict stored in the ORM column.

        Parameters
        ----------
        config_dict : dict[str, Any]
            The dict previously produced by :meth:`to_dict`. Enum fields may
            be stored as string values and are coerced back to their types.

        Returns
        -------
        Self
            A new instance of the concrete subclass.

        Raises
        ------
        TypeError
            If called on the abstract base class or a non-dataclass.
        ValueError
            If a required field is missing or an enum value is unrecognised.

        Notes
        -----
        Use this method to reconstruct an ``IndexConfig`` from the ORM
        ``index_config`` JSON column. It is distinct from
        :meth:`from_metadata`, which reads from a metadata dict that wraps the
        config under ``"index_config"`` key.
        """
        if not is_dataclass(cls):
            raise TypeError(f"Must be called on a dataclass, not {cls.__name__}.")
        if inspect.isabstract(cls):
            raise TypeError(
                f"Cannot instantiate abstract class {cls.__name__} directly. "
                "Call on a subclass."
            )
        type_hints = get_type_hints(cls)
        init_params: dict[str, Any] = {}
        for field in fields(cls):
            if field.name not in config_dict:
                raise ValueError(
                    f"Strict check failed: '{field.name}' missing in config dict."
                )
            value = config_dict[field.name]
            field_type = type_hints.get(field.name)
            if value is not None and isinstance(field_type, type) and issubclass(field_type, Enum):
                try:
                    value = field_type(value)
                except ValueError:
                    raise ValueError(
                        f"Invalid enum value '{value}' for {field_type.__name__}."
                    )
            init_params[field.name] = value
        factory = cast(Callable[..., Self], cls)
        return factory(**init_params)


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
    metric_type : None
        FLAT supports all metrics; the caller supplies the metric at query
        time. Setting this to a non-``None`` value raises ``ValueError``.
    """

    index_type: IndexType = IndexType.FLAT
    metric_type: Optional[MetricType] = None

    def _validate_metric_type_after_init(self) -> None:
        if self.metric_type is not None:
            raise ValueError(
                "FLAT index is an exact scan and does not take a metric_type. "
                f"Got: {self.metric_type}."
            )


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
        Size of the dynamic candidate list during search. Default ``16``.
    ef_construction : int
        Size of the dynamic candidate list during index construction.
        Default ``64``.

    Notes
    -----
    Supported by pgvector and FAISS backends only. sqlite-vec does not
    support HNSW.
    """

    metric_type: MetricType
    num_neighbors: int = 32
    ef_search: int = 16
    ef_construction: int = 64
    index_type: IndexType = IndexType.HNSW

    def _validate_metric_type_after_init(self) -> None:
        if self.metric_type is None:
            raise ValueError(
                "HNSW index requires a metric_type (e.g. MetricType.COSINE)."
            )


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
        return FlatIndexConfig()
    if index_type == IndexType.HNSW:
        return HNSWIndexConfig.from_kwargs(**kwargs)
    raise ValueError(f"No IndexConfig defined for index type {index_type!r}.")


def index_config_from_orm_row(index_type: IndexType, config_dict: Optional[dict[str, Any]]) -> IndexConfig:
    """Reconstruct an ``IndexConfig`` from ORM column values.

    Parameters
    ----------
    index_type : IndexType
        Value of the ORM ``index_type`` column.
    config_dict : dict[str, Any] or None
        Value of the ORM ``index_config`` JSON column. May be ``None`` or
        empty for legacy FLAT rows.

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
    if index_type == IndexType.FLAT:
        return FlatIndexConfig()
    if index_type == IndexType.HNSW:
        if not config_dict:
            raise ValueError(
                "HNSW index_config column is empty; cannot reconstruct HNSWIndexConfig."
            )
        return HNSWIndexConfig.from_dict(config_dict)
    raise ValueError(f"No IndexConfig defined for index type {index_type!r}.")
