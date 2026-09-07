from __future__ import annotations

import warnings
from typing import Any, Optional

from oa_configurator import (
    SCHEMA_TRANSLATE_MAP_KEY,
    Dialect,
    ResolvedDatabase,
    Role,
    ensure_schema,
    guard_schema_provenance,
    qualified,
    schema_inspect,
    supports_schemas,
)
from sqlalchemy import (
    DateTime,
    Engine,
    Enum,
    Integer,
    JSON,
    String,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, mapped_column, validates, Mapped

from omop_llm import supported_providers

from omop_emb.config import (
    MODEL_REGISTRY_SCHEMA,
    IndexType,
    MetricType,
)
from omop_emb.backends.index_config import IndexConfig

# Schema name for the model registry table. Dialects with schema support
# store the registry table in a dedicated schema to allow schema-independent 
# access to the registry table from any schema in the same database.
REGISTRY_SCHEMA_KEY = "registry"


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
    provider_type : str
        Provider that served the model (required; not part of any lookup
        key, but every registered model has one: the omop-llm provider
        key, e.g. ``'ollama'``).
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
    __table_args__ = {"schema": REGISTRY_SCHEMA_KEY}

    model_name = mapped_column(String, primary_key=True)

    provider_type = mapped_column(String, nullable=False)
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
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    @validates("provider_type")
    def _validate_provider_type(self, _key: str, value: str) -> str:
        """Reject a missing or unrecognized provider key."""
        if value is None:
            raise ValueError("provider_type is required.")
        if value not in supported_providers():
            raise ValueError(
                f"Unsupported provider type: {value!r}. Supported: {sorted(supported_providers())}"
            )
        return value

    @validates("index_config")
    def _validate_and_sync_index_config(
        self, _key: str, index_config: IndexConfig
    ) -> Optional[dict[str, Any]]:
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


def _registry_schema(bindable) -> str | None:
    """MODEL_REGISTRY_SCHEMA on a dialect with real schema support, else None."""
    return MODEL_REGISTRY_SCHEMA if supports_schemas(bindable) else None


def ensure_registry_table(engine: Engine, *, resolved: ResolvedDatabase | None = None) -> None:
    """Create or upgrade the model registry table, in its own reserved schema.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine connected to the registry database. Not required
        to already carry a REGISTRY_SCHEMA_KEY entry in its schema_translate_map.
    resolved : ResolvedDatabase, optional
        Enables the schema-provenance guard around the ``create_all()``
        call. Omitted by callers with no resolved config behind their
        engine, in which case the guard no-ops.
    """
    with engine.begin() as connection:
        connection = connection.execution_options(
            schema_translate_map={
                **(connection.get_execution_options().get(SCHEMA_TRANSLATE_MAP_KEY) or {}),
                REGISTRY_SCHEMA_KEY: _registry_schema(connection),
            }
        )
        ensure_schema(connection, MODEL_REGISTRY_SCHEMA)
        with guard_schema_provenance(connection, resolved, role=Role.PRIMARY):
            ModelRegistryBase.metadata.create_all(connection, tables=[ModelRegistry.__table__])  # ty: ignore[invalid-argument-type]
    _migrate_legacy_provider_type_column(engine)


def _migrate_legacy_provider_type_column(engine: Engine) -> None:
    """Upgrade the pre-omop-llm provider column without rebuilding embeddings.

    Older registries used ``Enum(ProviderType, native_enum=False)``, which
    stored enum member names such as ``OLLAMA`` in a ``VARCHAR(6)`` column.
    SQLite does not enforce that length, but PostgreSQL does, preventing newer
    provider keys such as ``anthropic`` from being inserted. Widen the
    PostgreSQL column and normalize legacy names in place on both backends.

    The migration is deliberately idempotent so normal backend construction
    can safely run it for both existing and newly-created registries.
    """
    columns = schema_inspect(engine, schema=_registry_schema(engine)).get_columns(ModelRegistry.__tablename__)
    provider_column = next(
        (column for column in columns if column["name"] == "provider_type"),
        None,
    )
    if provider_column is None:
        return

    legacy_length = getattr(provider_column["type"], "length", None)
    with engine.begin() as connection:
        registry_schema = _registry_schema(connection)
        if engine.dialect.name == Dialect.POSTGRESQL and legacy_length is not None:
            warnings.warn(
                "Widening a legacy fixed-length provider_type column. This "
                "migration path is deprecated and will be removed once no "
                "pre-omop-llm registry remains.",
                DeprecationWarning,
                stacklevel=2,
            )
            connection.execute(
                text(
                    f"ALTER TABLE {qualified(connection, ModelRegistry.__tablename__, schema=registry_schema)} "
                    "ALTER COLUMN provider_type TYPE VARCHAR "
                    "USING provider_type::text"
                )
            )
        # Raw text(), not update(): update() against the full mapped table
        # would pull in updated_at's onupdate=func.now() default, which the
        # legacy partial table (provider_type only) doesn't have.
        connection.execute(
            text(
                f"UPDATE {qualified(connection, ModelRegistry.__tablename__, schema=registry_schema)} "
                "SET provider_type = lower(provider_type) "
                "WHERE provider_type IS NOT NULL "
                "AND provider_type <> lower(provider_type)"
            )
        )
