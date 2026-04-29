from __future__ import annotations

from sqlalchemy import Engine, create_engine, and_, select
from sqlalchemy.orm import Session
import logging
from typing import Optional, Mapping
import pathlib
import re

from omop_emb.config import (
    IndexType,
    BackendType,
    ProviderType
)
from omop_emb.embeddings import EmbeddingProvider, OllamaProvider, OpenAIProvider
from .model_registry_cdm import ModelRegistry, ensure_model_registry_schema
from .model_registry_types import EmbeddingModelRecord
from omop_emb.utils.errors import ModelRegistrationConflictError

logger = logging.getLogger(__name__)



class ModelRegistryManager:
    """Manages model registry (metadata) for embedding models locally in a separate SQLite database."""
    DB_FILENAME = "metadata.db"
    def __init__(
        self,
        base_dir: str | pathlib.Path,
        db_file: Optional[str] = None,
    ):
        db_file = db_file or self.DB_FILENAME
        self._db_path = pathlib.Path(base_dir) / db_file
        if self._db_path.suffix != ".db":
            raise ValueError(f"Database file must have .db extension, got '{self._db_path.suffix}'.")
        self._engine = create_engine(f"sqlite:///{self._db_path}")

        ensure_model_registry_schema(self._engine)

    def get_registered_models_from_db(
        self,
        backend_type: Optional[BackendType] = None,
        model_name: Optional[str] = None,
        provider_type: Optional[ProviderType] = None,
        index_type: Optional[IndexType] = None,
    ) -> tuple[EmbeddingModelRecord, ...]:
        """
        Get the registered embedding models from the database, optionally filtered.

        Parameters
        ----------
        backend_type : BackendType, optional
            Filter by storage backend type
        model_name : str, optional
            Filter by model name
        provider_type : ProviderType, optional
            Filter by provider type
        index_type : IndexType, optional
            Filter by index type

        Returns
        -------
        tuple[EmbeddingModelRecord, ...]
            Matching records, or empty tuple if no matches
        """
        stmt = select(ModelRegistry)
        if backend_type is not None:
            stmt = stmt.where(ModelRegistry.backend_type == backend_type)
        if model_name is not None:
            stmt = stmt.where(ModelRegistry.model_name == model_name)
        if provider_type is not None:
            stmt = stmt.where(ModelRegistry.provider_type == provider_type)
        if index_type is not None:
            stmt = stmt.where(ModelRegistry.index_type == index_type)

        with Session(self._engine, expire_on_commit=False) as session:
            existing_models = session.scalars(stmt).all()

        return tuple(self._registry_entry_to_model_record(match) for match in existing_models)
       

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
        """
        Shared template method for model registration.

        Parameters
        ----------
        model_name : str
            Canonical model name (already validated by provider)
        provider_type : str
            Provider type that validated the model name (e.g. "OllamaProvider")
        dimensions : int
            Embedding vector dimension
        backend_type : BackendType
            Storage backend type
        index_type : IndexType
            Vector index type
        metadata : Mapping[str, object], optional
            Arbitrary metadata
        storage_identifier : str, optional
            Backend-specific storage identifier
        """
        with Session(self._engine, expire_on_commit=False) as session:
            existing_row = session.scalar(
                select(ModelRegistry).where(
                    and_(
                        ModelRegistry.model_name == model_name,
                        ModelRegistry.provider_type == provider_type,
                        ModelRegistry.backend_type == backend_type,
                        ModelRegistry.index_type == index_type,
                    )
                )
            )
            if existing_row is not None:
                # Model + provider + backend + index combo already registered
                # Check only the fields that can actually differ (provider/backend/index are filtered by WHERE)
                if existing_row.dimensions != dimensions:
                    raise ModelRegistrationConflictError(
                        f"Model '{model_name}' is already registered with dimensions "
                        f"{existing_row.dimensions}, not {dimensions}.",
                        conflict_field="dimensions"
                    )
                if existing_row.details != (metadata or {}):
                    raise ModelRegistrationConflictError(
                        f"Model '{model_name}' is already registered with different "
                        f"metadata. Reuse the existing model name or choose a new one.",
                        conflict_field="metadata"
                    )
                return self._registry_entry_to_model_record(existing_row)

            metadata = metadata or {}

        safe_name = self.safe_model_name(model_name)
        storage_name = storage_identifier or self.storage_name(
            safe_model_name=safe_name,
            index_type=index_type,
            backend_type=backend_type
        )
        
        new_entry = ModelRegistry(
            model_name=model_name,
            provider_type=provider_type,
            backend_type=backend_type,
            index_type=index_type,
            dimensions=dimensions,
            storage_identifier=storage_name,
            details=metadata
        )

        with Session(self._engine, expire_on_commit=False) as session:
            session.add(new_entry)
            session.commit()
        return self._registry_entry_to_model_record(new_entry)

    @staticmethod
    def safe_model_name(model_name: str) -> str:
        name = model_name.lower().strip()
        sanitized = re.sub(r"[^\w]+", "_", name)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        return sanitized
    
    @staticmethod
    def storage_name(
        safe_model_name: str,
        index_type: IndexType,
        backend_type: BackendType
    ) -> str:
        return f"{backend_type.value.lower()}_{safe_model_name}_{index_type.value}"
    
    def delete_model(
        self,
        model_name: str,
        *,
        provider_type: ProviderType,
        backend_type: BackendType,
        index_type: IndexType,
    ) -> None:
        """Remove a model registration from the registry database."""
        with Session(self._engine) as session:
            row = session.scalar(
                select(ModelRegistry).where(
                    and_(
                        ModelRegistry.model_name == model_name,
                        ModelRegistry.provider_type == provider_type,
                        ModelRegistry.backend_type == backend_type,
                        ModelRegistry.index_type == index_type,
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
        """Replace the metadata (details) for an existing model registration."""
        with Session(self._engine, expire_on_commit=False) as session:
            row = session.scalar(
                select(ModelRegistry).where(
                    and_(
                        ModelRegistry.model_name == model_name,
                        ModelRegistry.provider_type == provider_type,
                        ModelRegistry.backend_type == backend_type,
                        ModelRegistry.index_type == index_type,
                    )
                )
            )
            if row is None:
                raise ValueError(
                    f"Model '{model_name}' with provider='{provider_type}', "
                    f"backend='{backend_type}', index='{index_type}' not found in registry."
                )
            row.details = dict(metadata)
            session.commit()
            return self._registry_entry_to_model_record(row)

    @staticmethod
    def _coerce_registry_metadata(value: object) -> Mapping[str, object]:
        if isinstance(value, Mapping):
            return dict(value)
        return {}
    
    @staticmethod
    def _registry_entry_to_model_record(entry: ModelRegistry) -> EmbeddingModelRecord:
        return EmbeddingModelRecord(
            model_name=entry.model_name,
            provider_type=entry.provider_type,
            dimensions=entry.dimensions,
            backend_type=entry.backend_type,
            index_type=entry.index_type,
            storage_identifier=entry.storage_identifier,
            metadata=entry.details,
        )
