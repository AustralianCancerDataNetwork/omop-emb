"""Postgres-backed model registry.

Replaces the old SQLite ``ModelRegistryManager``.  The registry table lives in
the same Postgres instance as the embedding tables, so the caller manages one
engine for both concerns.  The engine is always injected — there is no
env-var fallback or auto-created SQLite file.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Mapping, Optional, Any

from sqlalchemy import (
    DateTime,
    Engine,
    Integer,
    JSON,
    String,
    and_,
    func,
    select,
    Enum,
)
from sqlalchemy.orm import DeclarativeBase, Session, mapped_column, validates

from omop_emb.config import BackendType, IndexType, ProviderType
from omop_emb.storage.index_config import (
    IndexConfig,
    index_config_from_index_type_and_metadata,
)
from omop_emb.utils.errors import ModelRegistrationConflictError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORM model
# ---------------------------------------------------------------------------

class _RegistryBase(DeclarativeBase):
    """Dedicated declarative base for the embedding registry."""


class _EmbeddingRegistry(_RegistryBase):
    """Persistent record of a registered embedding model.

    Stored in the same Postgres instance as the embedding tables so that one
    engine handles everything.

    Column notes
    ------------
    ``storage_identifier``
        The name of the corresponding embedding table in the same database.
    ``details``
        JSON blob containing at minimum ``{"index_config": {...}}``.
        May also hold the ``"faiss_cache"`` key once a FAISS export has been
        performed — see :const:`~omop_emb.storage.index_config.FAISS_CACHE_METADATA_KEY`.
    """

    __tablename__ = "embedding_registry"

    model_name = mapped_column(String, primary_key=True)
    provider_type = mapped_column(Enum(ProviderType, native_enum=False), nullable=False, primary_key=True)
    backend_type = mapped_column(Enum(BackendType, native_enum=False), nullable=False, primary_key=True)
    index_type = mapped_column(Enum(IndexType, native_enum=False), nullable=False, primary_key=True)
    dimensions = mapped_column(Integer, nullable=False)
    storage_identifier = mapped_column(String, unique=True, nullable=False)
    details = mapped_column(JSON, nullable=True, default=dict)
    created_at = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    @validates("provider_type")
    def _validate_provider_type(self, _key, value):
        if value not in ProviderType:
            raise ValueError(f"Unsupported provider type: {value}")
        return value

    @validates("backend_type")
    def _validate_backend_type(self, _key, value):
        if value not in BackendType:
            raise ValueError(f"Unsupported backend type: {value}")
        return value


def _ensure_registry_schema(engine: Engine) -> None:
    _RegistryBase.metadata.create_all(engine, tables=[_EmbeddingRegistry.__table__])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Dataclass returned to callers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingModelRecord:
    """Canonical description of a registered embedding model.

    ``storage_identifier`` is the name of the embedding table in the pgvector
    Postgres instance.
    """

    model_name: str
    provider_type: ProviderType
    dimensions: int
    backend_type: BackendType
    index_type: IndexType
    storage_identifier: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def index_config(self) -> IndexConfig:
        return index_config_from_index_type_and_metadata(self.index_type, self.metadata)


# ---------------------------------------------------------------------------
# Registry manager
# ---------------------------------------------------------------------------

class PostgresRegistryManager:
    """Manages the embedding model registry stored in the pgvector Postgres instance.

    Parameters
    ----------
    emb_engine : Engine
        Injected SQLAlchemy engine for the dedicated pgvector instance.
        The registry table (``embedding_registry``) is created here on first
        use via ``create_all``.
    """

    def __init__(self, emb_engine: Engine) -> None:
        self._engine = emb_engine
        _ensure_registry_schema(emb_engine)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_registered_models_from_db(
        self,
        backend_type: Optional[BackendType] = None,
        model_name: Optional[str] = None,
        provider_type: Optional[ProviderType] = None,
        index_type: Optional[IndexType] = None,
    ) -> tuple[EmbeddingModelRecord, ...]:
        stmt = select(_EmbeddingRegistry)
        if backend_type is not None:
            stmt = stmt.where(_EmbeddingRegistry.backend_type == backend_type)
        if model_name is not None:
            stmt = stmt.where(_EmbeddingRegistry.model_name == model_name)
        if provider_type is not None:
            stmt = stmt.where(_EmbeddingRegistry.provider_type == provider_type)
        if index_type is not None:
            stmt = stmt.where(_EmbeddingRegistry.index_type == index_type)

        with Session(self._engine, expire_on_commit=False) as session:
            rows = session.scalars(stmt).all()

        return tuple(self._row_to_record(r) for r in rows)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def register_model(
        self,
        model_name: str,
        dimensions: int,
        *,
        provider_type: ProviderType,
        backend_type: BackendType,
        index_type: IndexType,
        metadata: Optional[Mapping[str, object]] = None,
        storage_identifier: Optional[str] = None,
    ) -> EmbeddingModelRecord:
        """Register a model, returning the (possibly pre-existing) record.

        Raises ``ModelRegistrationConflictError`` if the model is already
        registered with different ``dimensions`` or ``metadata``.
        """
        with Session(self._engine, expire_on_commit=False) as session:
            existing = session.scalar(
                select(_EmbeddingRegistry).where(
                    and_(
                        _EmbeddingRegistry.model_name == model_name,
                        _EmbeddingRegistry.provider_type == provider_type,
                        _EmbeddingRegistry.backend_type == backend_type,
                        _EmbeddingRegistry.index_type == index_type,
                    )
                )
            )
            if existing is not None:
                if existing.dimensions != dimensions:
                    raise ModelRegistrationConflictError(
                        f"Model '{model_name}' already registered with "
                        f"dimensions={existing.dimensions}, requested {dimensions}.",
                        conflict_field="dimensions",
                    )
                if existing.details != (metadata or {}):
                    raise ModelRegistrationConflictError(
                        f"Model '{model_name}' already registered with different metadata.",
                        conflict_field="metadata",
                    )
                return self._row_to_record(existing)

        metadata = metadata or {}
        safe = self.safe_model_name(model_name)
        storage_name = storage_identifier or self.storage_name(
            safe_model_name=safe,
            index_type=index_type,
        )

        new_row = _EmbeddingRegistry(
            model_name=model_name,
            provider_type=provider_type,
            backend_type=backend_type,
            index_type=index_type,
            dimensions=dimensions,
            storage_identifier=storage_name,
            details=metadata,
        )
        with Session(self._engine, expire_on_commit=False) as session:
            session.add(new_row)
            session.commit()
        return self._row_to_record(new_row)

    def delete_model(
        self,
        model_name: str,
        *,
        provider_type: ProviderType,
        backend_type: BackendType,
        index_type: IndexType,
    ) -> None:
        with Session(self._engine) as session:
            row = session.scalar(
                select(_EmbeddingRegistry).where(
                    and_(
                        _EmbeddingRegistry.model_name == model_name,
                        _EmbeddingRegistry.provider_type == provider_type,
                        _EmbeddingRegistry.backend_type == backend_type,
                        _EmbeddingRegistry.index_type == index_type,
                    )
                )
            )
            if row is not None:
                session.delete(row)
                session.commit()

    def update_model_metadata(
        self,
        model_name: str,
        *,
        provider_type: ProviderType,
        backend_type: BackendType,
        index_type: IndexType,
        metadata: Mapping[str, object],
    ) -> EmbeddingModelRecord:
        """Replace the ``details`` JSON for an existing model registration."""
        with Session(self._engine, expire_on_commit=False) as session:
            row = session.scalar(
                select(_EmbeddingRegistry).where(
                    and_(
                        _EmbeddingRegistry.model_name == model_name,
                        _EmbeddingRegistry.provider_type == provider_type,
                        _EmbeddingRegistry.backend_type == backend_type,
                        _EmbeddingRegistry.index_type == index_type,
                    )
                )
            )
            if row is None:
                raise ValueError(
                    f"Model '{model_name}' "
                    f"(provider='{provider_type}', index='{index_type}') not found."
                )
            row.details = dict(metadata)
            session.commit()
            return self._row_to_record(row)

    def patch_model_metadata_key(
        self,
        model_name: str,
        *,
        provider_type: ProviderType,
        backend_type: BackendType,
        index_type: IndexType,
        key: str,
        value: Any,
    ) -> EmbeddingModelRecord:
        """Merge a single key into the ``details`` JSON without overwriting other keys.

        Used by the FAISS cache layer to persist ``"faiss_cache"`` metadata
        without disturbing the ``"index_config"`` key.
        """
        with Session(self._engine, expire_on_commit=False) as session:
            row = session.scalar(
                select(_EmbeddingRegistry).where(
                    and_(
                        _EmbeddingRegistry.model_name == model_name,
                        _EmbeddingRegistry.provider_type == provider_type,
                        _EmbeddingRegistry.backend_type == backend_type,
                        _EmbeddingRegistry.index_type == index_type,
                    )
                )
            )
            if row is None:
                raise ValueError(f"Model '{model_name}' not found in registry.")
            updated = dict(row.details or {})
            updated[key] = value
            row.details = updated
            session.commit()
            return self._row_to_record(row)

    # ------------------------------------------------------------------
    # Naming helpers
    # ------------------------------------------------------------------

    @staticmethod
    def safe_model_name(model_name: str) -> str:
        name = model_name.lower().strip()
        sanitized = re.sub(r"[^\w]+", "_", name)
        return re.sub(r"_+", "_", sanitized).strip("_")

    @staticmethod
    def storage_name(safe_model_name: str, index_type: IndexType) -> str:
        """Table name for the embedding vectors in the pgvector instance."""
        return f"pgvector_{safe_model_name}_{index_type.value}"

    @staticmethod
    def storage_name_to_index_and_model(storage_name: str) -> tuple[IndexType, str]:
        """Reverse a storage name back to ``(IndexType, safe_model_name)``."""
        parts = storage_name.split("_")
        if len(parts) < 3:
            raise ValueError(f"Invalid storage name: '{storage_name}'.")
        index_part = parts[-1]
        safe_model = "_".join(parts[1:-1])  # skip leading "pgvector" prefix
        return IndexType(index_part), safe_model

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: _EmbeddingRegistry) -> EmbeddingModelRecord:
        return EmbeddingModelRecord(
            model_name=row.model_name,
            provider_type=row.provider_type,
            dimensions=row.dimensions,
            backend_type=row.backend_type,
            index_type=row.index_type,
            storage_identifier=row.storage_identifier,
            metadata=row.details or {},
        )
