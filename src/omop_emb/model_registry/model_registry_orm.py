from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any, Optional

from oa_configurator import (
    Dialect,
    ResolvedDatabase,
    SchemaClaim,
    find_table_in_other_schemas,
    physical_schema_of,
    qualified,
)
from sqlalchemy import (
    Connection,
    DateTime,
    Engine,
    Enum,
    Integer,
    JSON,
    String,
    func,
    inspect,
    text,
)
from sqlalchemy.orm import DeclarativeBase, mapped_column, validates, Mapped

from omop_llm import supported_providers

from omop_emb.config import (
    MODEL_REGISTRY_SCHEMA,
    REGISTRY_SCHEMA_KEY,
    IndexType,
    MetricType,
)
from omop_emb.backends.index_config import IndexConfig
from omop_emb.utils.errors import MisplacedRegistryError


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


def resolve_registry_physical_schema(bindable) -> str | None:
    """REGISTRY_SCHEMA_KEY's physical schema resolved off bindable's own
    schema_translate_map.

    Raises
    ------
    oa_configurator.UnregisteredSchemaTagError
        If bindable wasn't built with the registry claim (registry_reader_engine()/
        registry_writer_engine()), on a dialect with real schema support.
    """
    return physical_schema_of(bindable, schema_tag=REGISTRY_SCHEMA_KEY)


def _registry_engine(
    database: ResolvedDatabase,
    *,
    extensions: Sequence[Callable[[Any, Any], None]],
    register_claims: bool,
) -> Engine:
    """Engine carrying the registry's reserved schema claim, registered or only checked.
    Parameters
    ----------
    database : ResolvedDatabase
        The resolved database to build the engine against.
    extensions : Sequence[Callable[[Any, Any], None]], optional
        Connect-event callables forwarded to ``database.create_engine()`` for
        any database extension the backend needs on every physical connection
        (see ``ResolvedDatabase.create_engine``).
    register_claims : bool, optional
        Forwarded to ``database.create_engine()``. False checks the registry
        claim without writing it.
    """
    return database.create_engine(
        schema_claims=[
            SchemaClaim(
                schema_tag=REGISTRY_SCHEMA_KEY,
                physical_schema=MODEL_REGISTRY_SCHEMA,
                reserved=True,
                owner="omop_emb",
            )
        ],
        extensions=extensions,
        register_claims=register_claims,
    )


def registry_reader_engine(
    database: ResolvedDatabase,
    *,
    extensions: Sequence[Callable[[Any, Any], None]] = (),
) -> Engine:
    """Map the registry schema without registering its claim or creating
    anything. If the schema or table doesn't exist yet,
    RegistryManager.registry_available reports that honestly.

    Parameters
    ----------
    database : ResolvedDatabase
        The resolved database to build the engine against.
    extensions : Sequence[Callable[[Any, Any], None]], optional
        Connect-event callables forwarded to ``database.create_engine()`` for
        any database extension the backend needs on every physical connection
        (see ``ResolvedDatabase.create_engine``).
    """
    return _registry_engine(database, extensions=extensions, register_claims=False)


def registry_writer_engine(
    database: ResolvedDatabase,
    *,
    extensions: Sequence[Callable[[Any, Any], None]] = (),
) -> Engine:
    """Claim the registry schema and ensure its table exists. The writable path.

    Call before constructing a RegistryManager/EmbeddingBackend that needs to
    be able to write. Both classes themselves never claim schemas or run DDL.
    """
    engine = _registry_engine(database, extensions=extensions, register_claims=True)
    ensure_registry_table(engine)
    return engine


def ensure_registry_table(engine: Engine) -> None:
    """Create or upgrade the model registry table, in its own reserved schema.

    The registry's schema is claimed and created by registry_writer_engine()
    at create_engine() time. This function only reads the
    already-resolved REGISTRY_SCHEMA_KEY off *engine*'s own
    schema_translate_map. 
    
    Notes
    -----
    - No provenance guard needed as engine was just built for this one call by
    registry_writer_engine(), so create_engine()'s own construction-time drift
    enforcement already covers it.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine connected to the registry database, already
        carrying a REGISTRY_SCHEMA_KEY entry in its schema_translate_map.

    Raises
    ------
    MisplacedRegistryError
        If the registry schema has no registry table but another schema does.
    """
    with engine.begin() as connection:
        registry_schema = resolve_registry_physical_schema(connection)
        if registry_schema is not None:
            _reject_misplaced_registry(connection, registry_schema=registry_schema)
        ModelRegistryBase.metadata.create_all(connection, tables=[ModelRegistry.__table__])  # ty: ignore[invalid-argument-type]
    _migrate_legacy_provider_type_column(engine)


def _reject_misplaced_registry(connection: Connection, *, registry_schema: str) -> None:
    """Refuse to create an empty registry while a registry table exists in another schema.

    Parameters
    ----------
    connection : sqlalchemy.Connection
        Connection the registry is created on.
    registry_schema : str
        Physical schema the registry belongs in.

    Raises
    ------
    MisplacedRegistryError
        If *registry_schema* has no registry table but another schema does.
    """
    table_name = ModelRegistry.__tablename__
    if inspect(connection).has_table(table_name, schema=registry_schema):
        return
    found = find_table_in_other_schemas(connection, table_name, physical_schema=registry_schema)
    if not found:
        return
    quoted_registry_schema = connection.dialect.identifier_preparer.quote_schema(registry_schema)
    raise MisplacedRegistryError(
        f"Found the model registry in schema(s) {list(found)}, but omop-emb keeps it in "
        f"{registry_schema!r}. Move it before continuing:\n"
        f"  CREATE SCHEMA IF NOT EXISTS {quoted_registry_schema};\n"
        f"  ALTER TABLE {qualified(connection, table_name, physical_schema=found[0])} "
        f"SET SCHEMA {quoted_registry_schema};"
    )


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
    columns = inspect(engine).get_columns(
        ModelRegistry.__tablename__, schema=resolve_registry_physical_schema(engine)
    )
    provider_column = next(
        (column for column in columns if column["name"] == "provider_type"),
        None,
    )
    if provider_column is None:
        return

    legacy_length = getattr(provider_column["type"], "length", None)
    with engine.begin() as connection:
        registry_schema = resolve_registry_physical_schema(connection)
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
                    f"ALTER TABLE {qualified(connection, ModelRegistry.__tablename__, physical_schema=registry_schema)} "
                    "ALTER COLUMN provider_type TYPE VARCHAR "
                    "USING provider_type::text"
                )
            )
        # Raw text(), not update(): update() against the full mapped table
        # would pull in updated_at's onupdate=func.now() default, which the
        # legacy partial table (provider_type only) doesn't have.
        connection.execute(
            text(
                f"UPDATE {qualified(connection, ModelRegistry.__tablename__, physical_schema=registry_schema)} "
                "SET provider_type = lower(provider_type) "
                "WHERE provider_type IS NOT NULL "
                "AND provider_type <> lower(provider_type)"
            )
        )
