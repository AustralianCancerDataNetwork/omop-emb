from __future__ import annotations

from sqlalchemy import Engine, create_engine, and_, select
from sqlalchemy.orm import Session
import logging
from typing import Optional, Mapping
import pathlib
import re

from omop_emb.config import (
    IndexType, 
    BackendType
)
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
        self.db_path = pathlib.Path(base_dir) / db_file
        self.engine = create_engine(f"sqlite:///{self.db_path}")

        ensure_model_registry_schema(self.engine)

    def get_registered_models_from_db(
        self, 
        backend_type: Optional[BackendType] = None,
        model_name: Optional[str] = None,
        index_type: Optional[IndexType] = None,
    ) -> Optional[tuple[EmbeddingModelRecord, ...]]:
        """
        Get the registered embedding models of the database, optionally filtered by backend type, model name, or index type.
        """
        stmt = select(ModelRegistry)
        if backend_type is not None:
            stmt = stmt.where(ModelRegistry.backend_type == backend_type)
        if model_name is not None:
            stmt = stmt.where(ModelRegistry.model_name == model_name)
        if index_type is not None:
            stmt = stmt.where(ModelRegistry.index_type == index_type)
    
        with Session(self.engine, expire_on_commit=False) as session:
            existing_models = session.scalars(stmt).all()

        if not existing_models:
            return None
        
        return tuple(self._registry_entry_to_model_record(match) for match in existing_models)
       

    def register_model(
        self,
        model_name: str,
        dimensions: int,
        *,
        backend_type: BackendType,
        index_type: IndexType,
        metadata: Optional[Mapping[str, object]] = None,
        storage_identifier: Optional[str] = None,
    ) -> EmbeddingModelRecord:
        """
        Shared template method for model registration.
        """
        with Session(self.engine, expire_on_commit=False) as session:
            existing_row = session.scalar(
                select(ModelRegistry).where(
                    and_(
                        ModelRegistry.model_name == model_name,
                        ModelRegistry.backend_type == backend_type,
                        ModelRegistry.index_type == index_type,
                    )
                )
            )
            if existing_row is not None:
                if existing_row.backend_type != backend_type:
                    raise ModelRegistrationConflictError(
                        f"Model '{model_name}' is already registered with backend "
                        f"'{existing_row.backend_type}', not '{backend_type}'. "
                        "Reuse the existing model name or choose a new one.",
                        conflict_field="backend_type"
                    )

                if existing_row.dimensions != dimensions:
                    raise ModelRegistrationConflictError(
                        f"Model '{model_name}' is already registered with dimensions "
                        f"{existing_row.dimensions}, not {dimensions}.",
                        conflict_field="dimensions"
                    )
                if existing_row.index_type != index_type:
                    raise ModelRegistrationConflictError(
                        f"Model '{model_name}' is already registered with "
                        f"index_method='{existing_row.index_type}', not "
                        f"'{index_type}'. Reuse the existing model "
                        "configuration or register a new model name.",
                        conflict_field="index_type"
                    )
                if existing_row.details != metadata:
                    raise ModelRegistrationConflictError(
                        f"Model '{model_name}' is already registered with different "
                        f"metadata. Reuse the existing model name or choose a new one.",
                        conflict_field="metadata"
                    )
                if storage_identifier is not None and existing_row.storage_identifier != storage_identifier:
                    raise ModelRegistrationConflictError(
                        f"Model '{model_name}' is already registered with storage identifier "
                        f"'{existing_row.storage_identifier}', not '{storage_identifier}'.",
                        conflict_field="storage_identifier"
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
            backend_type=backend_type,
            index_type=index_type,
            dimensions=dimensions,
            storage_identifier=storage_name,
            details=metadata
        )

        with Session(self.engine, expire_on_commit=False) as session:
            session.add(new_entry)
            session.commit()
        return self._registry_entry_to_model_record(new_entry)

    def delete_model(
        self,
        *,
        backend_type: BackendType,
        model_name: str,
        index_type: IndexType,
    ) -> bool:
        with Session(self.engine, expire_on_commit=False) as session:
            row = session.scalar(
                select(ModelRegistry).where(
                    and_(
                        ModelRegistry.model_name == model_name,
                        ModelRegistry.backend_type == backend_type,
                        ModelRegistry.index_type == index_type,
                    )
                )
            )
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    
    @staticmethod
    def safe_model_name(model_name: str) -> str:
        name = model_name.lower()
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
    
    @staticmethod
    def _coerce_registry_metadata(value: object) -> Mapping[str, object]:
        if isinstance(value, Mapping):
            return dict(value)
        return {}
    
    @staticmethod
    def _registry_entry_to_model_record(entry: ModelRegistry) -> EmbeddingModelRecord:
        return EmbeddingModelRecord(
            model_name=entry.model_name,
            dimensions=entry.dimensions,
            backend_type=entry.backend_type,
            index_type=entry.index_type,
            storage_identifier=entry.storage_identifier,
            metadata=entry.details,
        )
