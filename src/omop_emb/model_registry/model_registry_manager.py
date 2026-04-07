from __future__ import annotations

from sqlalchemy import Engine, create_engine, and_, select
from sqlalchemy.orm import Session
import logging
from typing import Optional, Mapping
import pathlib
import re
from dataclasses import dataclass, field

from orm_loader.helpers import Base

from omop_emb.config import (
    IndexType, 
    BackendType
)
from .model_registry_cdm import ModelRegistry
from omop_emb.utils.errors import ModelRegistrationConflictError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingModelRecord:
    """
    Canonical description of a registered embedding model.

    ``storage_identifier`` is intentionally backend-specific. For example:
    - PostgreSQL backend: dynamic embedding table name
    - FAISS backend: on-disk index path or logical collection name
    """

    model_name: str
    dimensions: int
    backend_type: BackendType
    index_type: IndexType
    storage_identifier: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    
    @classmethod
    def from_modelregistry(cls, model_registry: ModelRegistry) -> EmbeddingModelRecord:
        return cls(
            model_name=model_registry.model_name,
            dimensions=model_registry.dimensions,
            backend_type=model_registry.backend_type,
            storage_identifier=model_registry.storage_identifier,
            index_type=model_registry.index_type,
            metadata=model_registry.details,
        )


class ModelRegistryManager:
    """Manages model registry (metadata) for embedding models locally in a separate SQLite database."""
    DB_FILENAME = "metadata.db"
    REGISTRY_BASE_DIR = ".omop_emb"
    def __init__(
        self, 
        base_dir: str = ".omop_emb",
        db_file: Optional[str] = None,
    ):
        db_file = db_file or self.DB_FILENAME
        self.db_path = pathlib.Path(base_dir) / db_file
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}")

        Base.metadata.create_all(self.engine, tables=[ModelRegistry.__table__])  # type: ignore[arg-type]

    def get_registered_models(
        self, 
        backend_type: Optional[BackendType] = None,
        model_name: Optional[str] = None,
        index_type: Optional[IndexType] = None,
    ) -> Optional[tuple[EmbeddingModelRecord, ...]]:
        """
        Prepare any required storage structures.

        Examples:
        - create registry tables
        - warm caches from a registry
        - create directories or sidecar files
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
        return tuple(EmbeddingModelRecord.from_modelregistry(row) for row in existing_models)
       

    def register_model(
        self,
        model_name: str,
        dimensions: int,
        *,
        backend_type: BackendType,
        index_type: IndexType,
        metadata: Mapping[str, object] = {},
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
                return EmbeddingModelRecord.from_modelregistry(existing_row)

        safe_name = self.safe_model_name(model_name)
        storage_name = self.storage_name(
            safe_model_name=safe_name, 
            index_type=index_type,
            backend_type=backend_type
        )
        
        new_entry = ModelRegistry(
            model_name=model_name,
            dimensions=dimensions,
            storage_identifier=storage_name,
            index_type=index_type,
            backend_type=backend_type,
            details=metadata
        )

        with Session(self.engine, expire_on_commit=False) as session:
            # TODO: Add logic here to check for existing records before adding
            session.add(new_entry)
            session.commit()
        return EmbeddingModelRecord.from_modelregistry(new_entry)

    
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