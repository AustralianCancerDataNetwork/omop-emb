from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime

from oa_configurator import database_config_name_of
from sqlalchemy import Engine, inspect, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from omop_emb.backends.index_config import (
    RESERVED_METADATA_KEYS,
    IndexConfig,
    index_config_from_dict,
)
from omop_emb.config import MetricType
from omop_emb.model_registry.model_registry_orm import (
    ModelRegistry,
    resolve_registry_physical_schema,
)
from omop_emb.model_registry.model_registry_types import EmbeddingModelRecord
from omop_emb.utils.errors import ModelRegistrationConflictError

logger = logging.getLogger(__name__)
_MAX_POSTGRES_IDENTIFIER_BYTES = 63
_STORAGE_IDENTIFIER_HASH_LENGTH = 8
_INDEX_NAME_FIXED_BYTES = len("idx_") + len("emb_") + 2 + _STORAGE_IDENTIFIER_HASH_LENGTH
STORAGE_IDENTIFIER_READABLE_PREFIX_LENGTH = (
    _MAX_POSTGRES_IDENTIFIER_BYTES
    - _INDEX_NAME_FIXED_BYTES
    - max(len(metric.value.encode("utf-8")) for metric in MetricType)
)


def _register_model_atomically(
    manager: RegistryManager,
    *,
    model_name: str,
    provider_type: str,
    dimensions: int,
    index_config: IndexConfig,
    metadata: Mapping[str, object] | None,
    registered_at: datetime | None,
) -> EmbeddingModelRecord:
    try:
        with manager.emb_session_factory(expire_on_commit=False) as session, session.begin():
            if manager.embedding_engine.dialect.name == "sqlite":
                # SQLite's deferred transactions can both read before either
                # inserts. Acquire the write reservation before the pre-check
                # so the second registration observes the first committed row.
                session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            existing = manager._fetch_row(session, model_name)
            if existing is not None:
                return _existing_registration_result(
                    manager, existing, model_name, dimensions, metadata
                )
            other_store = _other_store_for_model(session, manager, model_name)
            if other_store is not None:
                raise _store_conflict(model_name, other_store)
            row = ModelRegistry(
                database_config_name=manager._database_config_name,
                model_name=model_name,
                provider_type=provider_type,
                dimensions=dimensions,
                storage_identifier=manager.storage_name(manager._database_config_name, model_name),
                details=dict(metadata) if metadata else {},
                index_config=index_config,
            )
            if registered_at is not None:
                row.created_at = registered_at
                row.updated_at = registered_at
            session.add(row)
            session.flush()
    except IntegrityError:
        with manager.emb_session_factory() as session:
            existing = manager._fetch_row(session, model_name)
            if existing is not None:
                return _existing_registration_result(
                    manager, existing, model_name, dimensions, metadata
                )
            other_store = _other_store_for_model(session, manager, model_name)
            if other_store is not None:
                raise _store_conflict(model_name, other_store)
        raise
    return manager._row_to_record(row)


def _other_store_for_model(
    session: Session, manager: RegistryManager, model_name: str
) -> str | None:
    return session.scalar(
        select(ModelRegistry.database_config_name).where(
            ModelRegistry.model_name == model_name,
            ModelRegistry.database_config_name != manager._database_config_name,
        )
    )


def _store_conflict(model_name: str, store_name: str) -> ModelRegistrationConflictError:
    return ModelRegistrationConflictError(
        f"Model '{model_name}' is already registered by vector store '{store_name}' "
        "in this database. Use that store, or delete the model there first.",
        conflict_field="model_name",
    )


def _existing_registration_result(
    manager: RegistryManager,
    existing: ModelRegistry,
    model_name: str,
    dimensions: int,
    metadata: Mapping[str, object] | None,
) -> EmbeddingModelRecord:
    if existing.dimensions != dimensions:
        raise ModelRegistrationConflictError(
            f"Model '{model_name}' already registered with "
            f"dimensions={existing.dimensions}, got {dimensions}.",
            conflict_field="dimensions",
        )
    if existing.details != (metadata or {}):
        raise ModelRegistrationConflictError(
            f"Model '{model_name}' is already registered with different metadata. "
            "Reuse the existing model name or choose a new one.",
            conflict_field="metadata",
        )
    return manager._row_to_record(existing)


class RegistryManager:
    """Registry of embedding models co-located with the embedding store.

    For sqlite-vec the engine points at the same .db file as the vec0 tables.
    For pgvector it points at the same Postgres database.

    Parameters
    ----------
    embedding_engine : Engine
        SQLAlchemy embedding engine connected to the embedding store. Must
        have been built via registry_reader_engine()/registry_writer_engine(),
        which is checked here to ensure the registry schema claim is present. 

    Notes
    -----
    The registry table is created under the ``registry`` schema tag (a
    ``schema_translate_map`` key, resolved to the physical schema named by
    ``MODEL_REGISTRY_SCHEMA``), independent of whichever schema the embedding
    store itself resolves to.

    Raises
    ------
    oa_configurator.UnregisteredSchemaTagError
        If embedding_engine wasn't built with the registry schema claim.
    """

    def __init__(self, embedding_engine: Engine) -> None:
        self._embedding_engine = embedding_engine
        self._embedding_sessionmaker = sessionmaker(self._embedding_engine)
        resolve_registry_physical_schema(embedding_engine)
        self._database_config_name = database_config_name_of(embedding_engine)

    @property
    def registry_available(self) -> bool:
        """Whether the registry table currently exists on this engine.

        True for any writable-bootstrapped engine (``ensure_registry_table``
        already ran). May be ``False`` for a peek-constructed engine on a
        store that's never been configured -- callers must not assume this
        is always True.
        """
        return inspect(self._embedding_engine).has_table(
            ModelRegistry.__tablename__, schema=resolve_registry_physical_schema(self._embedding_engine)
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def embedding_engine(self) -> Engine:
        """SQLAlchemy engine connected to the embedding store."""
        return self._embedding_engine

    @property
    def emb_session_factory(self) -> sessionmaker:
        """Session factory bound to ``embedding_engine``."""
        return self._embedding_sessionmaker

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_registered_models(
        self,
        *,
        model_name: str | None = None,
        provider_type: str | None = None,
    ) -> tuple[EmbeddingModelRecord, ...]:
        """Return all registered models matching the given filters.

        Parameters
        ----------
        model_name : str, optional
            Filter by canonical model name.
        provider_type : str, optional
            omop-llm provider key that serves the model.
        Returns
        -------
        tuple[EmbeddingModelRecord, ...]
        """
        if not self.registry_available:
            return ()
        stmt = select(ModelRegistry).where(ModelRegistry.database_config_name == self._database_config_name)
        if model_name is not None:
            stmt = stmt.where(ModelRegistry.model_name == model_name)
        if provider_type is not None:
            stmt = stmt.where(ModelRegistry.provider_type == provider_type)

        with self.emb_session_factory(expire_on_commit=False) as session:
            rows = session.scalars(stmt).all()
        return tuple(self._row_to_record(r) for r in rows)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def register_model(
        self,
        *,
        model_name: str,
        index_config: IndexConfig,
        dimensions: int,
        provider_type: str,
        metadata: Mapping[str, object] | None = None,
        registered_at: datetime | None = None,
    ) -> EmbeddingModelRecord:
        """Register a model or return the existing record if already registered.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        index_config : IndexConfig
            Initial index configuration.
        dimensions : int
            Embedding vector dimensionality.
        provider_type : str
            omop-llm provider key that serves the model.
        metadata : Mapping[str, object], optional
            Free-form operational metadata.
        registered_at : datetime, optional
            Backdate ``created_at``/``updated_at`` to this timestamp instead
            of "now" (e.g. to match the source data's true snapshot time
            when importing an export bundle on a different machine). Only
            applies to a brand-new registration; ignored when the model is
            already registered.

        Returns
        -------
        EmbeddingModelRecord
            The newly created or already-existing record.

        Raises
        ------
        ModelRegistrationConflictError
            If the model is already registered with a different configuration.
        ValueError
            If ``metadata`` contains a reserved key.
        """
        _validate_metadata_keys(metadata)
        return _register_model_atomically(
            self,
            model_name=model_name,
            provider_type=provider_type,
            dimensions=dimensions,
            index_config=index_config,
            metadata=metadata,
            registered_at=registered_at,
        )

    def delete_model(self, *, model_name: str) -> None:
        """Delete a registry row. No-op if the row does not exist.

        Parameters
        ----------
        model_name : str
        """
        with self.emb_session_factory() as session:
            row = self._fetch_row(session, model_name)
            if row is not None:
                session.delete(row)
                session.commit()

    def update_index_config(
        self,
        *,
        model_name: str,
        index_config: IndexConfig,
    ) -> EmbeddingModelRecord:
        """Replace the index configuration of an existing registry row.

        Parameters
        ----------
        model_name : str
        index_config : IndexConfig
            New index configuration. The ``@validates`` hook syncs
            ``index_type`` and ``metric_type`` columns automatically.

        Returns
        -------
        EmbeddingModelRecord
            Updated record.

        Raises
        ------
        ValueError
            If the model is not registered.
        """
        with self.emb_session_factory(expire_on_commit=False) as session:
            row = self._fetch_row(session, model_name)
            if row is None:
                raise ValueError(f"Model '{model_name}' is not registered.")
            row.index_config = index_config
            session.commit()
            return self._row_to_record(row)

    def update_metadata(
        self,
        *,
        model_name: str,
        metadata: Mapping[str, object],
    ) -> EmbeddingModelRecord:
        """Replace the free-form metadata of an existing registry row.

        Parameters
        ----------
        model_name : str
        metadata : Mapping[str, object]
            New metadata dict. Replaces the existing value entirely.

        Returns
        -------
        EmbeddingModelRecord
            Updated record.

        Raises
        ------
        ValueError
            If the model is not registered.
        """
        _validate_metadata_keys(metadata)
        with self.emb_session_factory(expire_on_commit=False) as session:
            row = self._fetch_row(session, model_name)
            if row is None:
                raise ValueError(f"Model '{model_name}' is not registered.")
            row.details = dict(metadata)
            session.commit()
            return self._row_to_record(row)

    def refresh_model_updated_at_timestamp(self, *, model_name: str) -> None:
        """Bump a registry row's ``updated_at`` to now, with no other change.

        Notes
        -----
        Uses a Python-side timestamp rather than ``func.now()`` so the value
        carries microsecond precision on every backend. SQLite's
        ``CURRENT_TIMESTAMP``only has whole-second resolution, 
        which made staleness checks racy when an upsert and its preceding 
        FAISS export land in the same second.

        Parameters
        ----------
        model_name : str
        """
        with self.emb_session_factory.begin() as session:
            session.execute(
                update(ModelRegistry)
                .where(
                    ModelRegistry.database_config_name == self._database_config_name,
                    ModelRegistry.model_name == model_name,
                )
                .values(updated_at=datetime.now(UTC))
            )

    # ------------------------------------------------------------------
    # Naming helpers
    # ------------------------------------------------------------------

    @staticmethod
    def safe_model_name(model_name: str) -> str:
        """Normalise a model name for use in table identifiers.

        Lowercases and replaces any run of non-word characters with a single
        underscore, then strips leading/trailing underscores.

        Parameters
        ----------
        model_name : str
            Raw model name (e.g. ``'nomic-embed-text:v1.5'``).

        Returns
        -------
        str
            Normalised name (e.g. ``'nomic_embed_text_v1_5'``).
        """
        name = model_name.lower().strip()
        sanitized = re.sub(r"[^\w]+", "_", name)
        return re.sub(r"_+", "_", sanitized).strip("_")

    @staticmethod
    def storage_name(database_config_name: str | None, model_name: str) -> str:
        """Build the deterministic physical table name for a new model registration.

        Parameters
        ----------
        database_config_name : str or None
            The owning ``[databases.*]`` entry name, if available.
        model_name : str
            Canonical provider model ID.

        Returns
        -------
        str
            Store-scoped name ``emb_<readable>_<8 hex SHA-256 characters>``.
        """
        readable = re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]", "_", model_name.lower()))
        readable = readable.strip("_")[:STORAGE_IDENTIFIER_READABLE_PREFIX_LENGTH].rstrip("_")
        readable = readable or "model"
        identity = f"{database_config_name or ''}|{model_name}".encode()
        digest = hashlib.sha256(identity).hexdigest()[:_STORAGE_IDENTIFIER_HASH_LENGTH]
        return f"emb_{readable}_{digest}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_row(self, session: Session, model_name: str) -> ModelRegistry | None:
        return session.scalar(
            select(ModelRegistry).where(
                ModelRegistry.database_config_name == self._database_config_name,
                ModelRegistry.model_name == model_name,
            )
        )

    @staticmethod
    def _row_to_record(row: ModelRegistry) -> EmbeddingModelRecord:
        """Convert a SQLAlchemy ORM row to a dataclass record.

        Notes
        -----
        Rows written with v1.X ProviderType Enum column stored the
        Python enum *member name* ("OLLAMA"), not its value ("ollama"). This function normalises
        legacy rows to the lowercase omop-llm provider key every current/future row already uses.
        The only two deprecated omop-emb providers are the following:
        - OLLAMA -> ollama
        - OPENAI -> openai
        """

        index_config = index_config_from_dict(row.index_type, row.index_config)
        provider_type = row.provider_type.lower() if row.provider_type else row.provider_type
        return EmbeddingModelRecord(
            model_name=row.model_name,
            provider_type=provider_type,
            index_config=index_config,
            dimensions=row.dimensions,
            storage_identifier=row.storage_identifier,
            metadata=dict(row.details or {}),
            created_at=_as_utc(row.created_at),
            updated_at=_as_utc(row.updated_at),
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _as_utc(value: datetime | None) -> datetime | None:
    """Attach UTC tzinfo to a naive datetime, leaving aware ones untouched.

    sqlite has no native timezone-aware storage, so SQLAlchemy round-trips
    ``DateTime(timezone=True)`` columns as naive on that backend (pgvector
    preserves tzinfo). Every timestamp this registry writes is UTC, so a
    naive value read back is always UTC too: normalize here, once, so
    every caller can assume tz-aware and compare against other UTC-aware
    datetimes (e.g. a bundle's ``exported_at``) without crashing.
    """
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _validate_metadata_keys(metadata: Mapping[str, object] | None) -> None:
    """Raise ``ValueError`` if ``metadata`` contains a reserved key.

    Parameters
    ----------
    metadata : Mapping[str, object] or None
        Caller-supplied metadata dict to validate.

    Raises
    ------
    ValueError
        If any key in ``metadata`` is in ``RESERVED_METADATA_KEYS``.
    """
    if not metadata:
        return
    protected_keys_in_metadata = set(metadata.keys()) & RESERVED_METADATA_KEYS
    if protected_keys_in_metadata:
        raise ValueError(
            f"Metadata must not contain reserved keys: {sorted(protected_keys_in_metadata)}. "
            "These are managed internally by the registry."
        )
