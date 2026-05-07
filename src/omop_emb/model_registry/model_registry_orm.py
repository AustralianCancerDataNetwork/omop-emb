from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import DateTime, Engine, Integer, JSON, String, Enum, func
from sqlalchemy.orm import DeclarativeBase, mapped_column, validates, Mapped

from omop_emb.config import (
    BackendType,
    IndexType,
    MetricType,
    ProviderType,
    get_supported_index_types_for_backend,
    is_index_type_supported_for_backend,
)
from omop_emb.backends.index_config import IndexConfig


class ModelRegistryBase(DeclarativeBase):
    """Dedicated declarative base for local model registry metadata."""
    pass


class ModelRegistry(ModelRegistryBase):
    """ORM row for one registered embedding model.

    The primary key is ``(model_name, backend_type)``.  ``provider_type`` is
    stored as a plain nullable column for diagnostics; it is not part of the
    key because the same model name + backend always maps to the same physical
    table regardless of which compatible endpoint generated the vectors.

    Each model has exactly one active index at any time.
    ``index_type`` and ``metric_type`` are regular (non-key) columns that are automatically
    synced when ``index_config`` is assigned.

    Attributes
    ----------
    model_name : str
        Canonical model name including tag.
    provider_type : ProviderType
        Provider that served the model (informational only).
    backend_type : BackendType
        Embedding storage backend.
    storage_identifier : str
        Physical table name (or file name for FAISS) where the model's embeddings are stored. 
        Must be unique across the registry.
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
        Free-form operational data (e.g. FAISS cache info, user extras).
        Exposed as ``metadata`` on ``EmbeddingModelRecord``. Never contains
        ``index_config`` data; that key is reserved.
    created_at : datetime
        Row creation timestamp (UTC).
    updated_at : datetime
        Row last-updated timestamp (UTC).

    Notes
    -----
    Assign an ``IndexConfig`` instance to ``row.index_config`` — the
    ``@validates`` hook unpacks it into ``index_type`` / ``metric_type``
    columns, validates backend support, and stores the ``to_dict()`` result
    in the JSON column. Do not assign the raw dict directly.
    """

    __tablename__ = "model_registry"

    model_name = mapped_column(String, primary_key=True)
    backend_type = mapped_column(Enum(BackendType, native_enum=False), primary_key=True)
    
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

    @validates("backend_type")
    def _validate_backend_type(self, _key: str, value: str) -> str:
        """Reject unknown backend types on assignment."""
        if value not in BackendType:
            raise ValueError(f"Unsupported backend type: {value}: Supported: {list(BackendType)}")
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
        ValueError
            If ``config_obj`` is ``None``, has a ``None`` ``index_type``, has
            an incompatible ``metric_type`` for its ``index_type``, or names
            an index type not supported by this row's ``backend_type``.
        TypeError
            If ``config_obj`` is not an ``IndexConfig`` subclass.

        Notes
        -----
        After this hook runs, ``self.index_type`` and ``self.metric_type`` are
        always consistent with the stored ``index_config`` JSON.
        """
        # TODO: MAybe allow dict type? Then we can piece together an IndexConfig from the dict and validate it properly, instead of just trusting the dict keys/values.
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

        if not is_index_type_supported_for_backend(
            index=index_config.index_type, backend=self.backend_type
        ):
            supported = get_supported_index_types_for_backend(self.backend_type)
            raise ValueError(
                f"Index type '{index_config.index_type.value}' is not supported for "
                f"backend '{self.backend_type.value}'. "
                f"Supported: {[idx.value for idx in supported]}."
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
