from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import DateTime, Engine, Integer, JSON, String, Enum, func
from sqlalchemy.orm import DeclarativeBase, mapped_column, validates, Mapped

from omop_emb.config import (
    IndexType,
    MetricType,
    ProviderType,
)
from omop_emb.backends.index_config import IndexConfig


class ModelRegistryBase(DeclarativeBase):
    """Dedicated declarative base for local model registry metadata."""
    pass


class ModelRegistry(ModelRegistryBase):
    """ORM row for one registered embedding model.

    The primary key is ``model_name``.  ``provider_type`` is stored as a plain
    nullable column for diagnostics.

    Each model has exactly one active index at any time.
    ``index_type`` and ``metric_type`` are regular (non-key) columns that are
    automatically synced when ``index_config`` is assigned.

    Attributes
    ----------
    model_name : str
        Canonical model name including tag.
    provider_type : ProviderType
        Provider that served the model (informational only).
    storage_identifier : str
        Physical table name where the model's embeddings are stored.
        Must be unique across the registry. Format: ``<backend>_<safe_model>``.
    dimensions : int
        Embedding vector dimensionality.
    index_type : IndexType
        Active index type. Synced from ``index_config``. Default ``FLAT``.
    metric_type : MetricType or None
        Distance metric locked by the index. ``None`` for FLAT (any metric
        accepted at query time).
    index_config : dict or None
        JSON serialisation of the active ``IndexConfig``. Written by the
        ``@validates`` hook whenever an ``IndexConfig`` object is assigned.
    details : dict or None
        Free-form operational data (e.g. user extras).
        Exposed as ``metadata`` on ``EmbeddingModelRecord``.
    created_at : datetime
        Row creation timestamp (UTC).
    updated_at : datetime
        Row last-updated timestamp (UTC).

    Notes
    -----
    Assign an ``IndexConfig`` instance to ``row.index_config``: The
    ``@validates`` hook unpacks it into ``index_type`` / ``metric_type``
    columns and stores the ``to_dict()`` result in the JSON column. Do not
    assign the raw dict directly.
    """

    __tablename__ = "model_registry"

    model_name = mapped_column(String, primary_key=True)

    provider_type = mapped_column(Enum(ProviderType, native_enum=False))
    storage_identifier = mapped_column(String, nullable=False, unique=True)
    dimensions = mapped_column(Integer, nullable=False)

    index_type: Mapped[IndexType] = mapped_column(
        Enum(IndexType, native_enum=False), nullable=False, default=IndexType.FLAT
    )
    metric_type: Mapped[Optional[MetricType]] = mapped_column(
        Enum(MetricType, native_enum=False), nullable=True
    )
    index_config: Mapped[Any] = mapped_column(JSON, nullable=True, default=dict)
    details: Mapped[Any] = mapped_column(JSON, nullable=True, default=dict)

    created_at = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    @validates("provider_type")
    def _validate_provider_type(self, _key: str, value: str) -> str:
        """Reject unknown provider types on assignment."""
        if value not in ProviderType:
            raise ValueError(f"Unsupported provider type: {value!r}. Supported: {list(ProviderType)}")
        return value

    @validates("index_config")
    def _validate_and_sync_index_config(self, _key: str, index_config: IndexConfig) -> Optional[dict[str, Any]]:
        """Unpack an ``IndexConfig`` into the row's index columns.

        Parameters
        ----------
        _key : str
            SQLAlchemy attribute name (always ``'index_config'``).
        index_config : IndexConfig
            The config object being assigned. Must be an ``IndexConfig``
            subclass instance, not a raw dict.

        Returns
        -------
        dict or None
            ``index_config.to_dict()`` stored in the JSON column.

        Raises
        ------
        TypeError
            If ``config_obj`` is not an ``IndexConfig`` subclass.
        ValueError
            If ``config_obj`` is ``None``, has a ``None`` ``index_type``, or
            has an incompatible ``metric_type`` for its ``index_type``.

        Notes
        -----
        Backend-level index support validation (e.g. sqlite-vec does not
        support HNSW) is performed by the backend before calling the registry,
        not here. The ORM only enforces structural constraints.
        """
        if index_config is None:
            raise ValueError(
                "index_config cannot be None. Provide a valid IndexConfig instance."
            )
        if not isinstance(index_config, IndexConfig):
            raise TypeError("Must assign an IndexConfig subclass instance.")
        if index_config.index_type is None:
            raise ValueError("index_config must have a non-null index_type.")

        if index_config.index_type == IndexType.FLAT:
            if index_config.metric_type is not None:
                raise ValueError(
                    "FLAT index does not take a metric_type. "
                    "Set metric_type to None for FLAT indices."
                )
        else:
            if index_config.metric_type is None:
                raise ValueError(
                    f"{index_config.index_type} index requires a metric_type "
                    "(e.g. MetricType.COSINE)."
                )

        self.index_type = index_config.index_type
        self.metric_type = index_config.metric_type
        return index_config.to_dict()


def ensure_registry_schema(engine: Engine) -> None:
    """Create the model registry table if it does not exist.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine connected to the registry database.
    """
    ModelRegistryBase.metadata.create_all(engine, tables=[ModelRegistry.__table__])  # type: ignore[arg-type]
