from sqlalchemy import DateTime, Engine, Integer, JSON, String, func, create_engine, Enum
from sqlalchemy.orm import mapped_column, validates

from orm_loader.helpers import Base

from ..config import BackendType, IndexType, is_index_type_supported_for_backend, get_supported_index_types_for_backend

class ModelRegistry(Base):
    """
    Shared database-backed registry for embedding models across backends.

    Notes
    -----
    The underlying database column storing the backend-specific location is
    named ``table_name`` in the schema, while the ORM attribute exposed here
    is ``storage_identifier`` because it is not necessarily always a SQL table
    name.
    """

    __tablename__ = "model_registry"

    model_name = mapped_column(String, primary_key=True)
    backend_type = mapped_column(Enum(BackendType, native_enum=False), nullable=False, primary_key=True)
    index_type = mapped_column(Enum(IndexType, native_enum=False), nullable=False, primary_key=True)
    dimensions = mapped_column(Integer, nullable=False)
    storage_identifier = mapped_column("table_name", String, unique=True, nullable=False)
    details = mapped_column(JSON, nullable=True, default=dict)
    created_at = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    @validates("index_type")
    def validate_index_for_backend(self, key, index_type):
        if not is_index_type_supported_for_backend(self.backend_type, index_type):
            raise ValueError(
                f"Backend {self.backend_type} does not support {index_type}. "
                f"Supported: {get_supported_index_types_for_backend(self.backend_type)}"
            )
        return index_type
    
    @validates("backend_type")
    def validate_backend_type(self, key, backend_type):
        if backend_type not in BackendType:
            raise ValueError(f"Unsupported backend type: {backend_type}. Supported backends: {list(BackendType)}")
        return backend_type