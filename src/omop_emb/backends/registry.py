from __future__ import annotations

from sqlalchemy import DateTime, Engine, Integer, JSON, String, func, inspect, text, Enum
from sqlalchemy.orm import mapped_column, validates

from orm_loader.helpers import Base

from .config import BACKEND_SUPPORTED_INDICES, IndexType, BackendType


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

    # TODO: Think about having multiple models per index and backend. Would require to 
    # have the storage_identifier (i.e. the table_name) to be unique as well, which is 
    # currently not enforced explicitly but implicitily as it only depends on model_name that is unique
    model_name = mapped_column(String, primary_key=True)
    dimensions = mapped_column(Integer, nullable=False)
    storage_identifier = mapped_column("table_name", String, unique=True, nullable=False)
    index_type = mapped_column(Enum(IndexType, native_enum=False), nullable=False)
    backend_type = mapped_column(Enum(BackendType, native_enum=False), nullable=False)
    details = mapped_column("metadata", JSON, nullable=False, default=dict)
    created_at = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    @validates("index_type")
    def validate_index_for_backend(self, key, index_type):
        supported = BACKEND_SUPPORTED_INDICES.get(self.backend_type, [])
        if index_type not in supported:
            raise ValueError(
                f"Backend {self.backend_type} does not support {index_type}. "
                f"Supported: {supported}"
            )
        return index_type


def ensure_model_registry_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine, tables=[ModelRegistry.__table__])  # type: ignore[arg-type]

    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns(ModelRegistry.__tablename__)}

    with engine.begin() as conn:
        if "index_type" not in columns:
            conn.execute(text("ALTER TABLE model_registry ADD COLUMN index_type VARCHAR"))
            conn.execute(
                text("UPDATE model_registry SET index_type = 'hnsw' WHERE index_type IS NULL")
            )
            conn.execute(text("ALTER TABLE model_registry ALTER COLUMN index_type SET NOT NULL"))
            conn.execute(text("ALTER TABLE model_registry ALTER COLUMN index_type SET DEFAULT 'hnsw'"))

        if "backend_type" not in columns:
            conn.execute(text("ALTER TABLE model_registry ADD COLUMN backend_type VARCHAR"))
            conn.execute(
                text("UPDATE model_registry SET backend_type = 'pgvector' WHERE backend_type IS NULL")
            )
            conn.execute(text("ALTER TABLE model_registry ALTER COLUMN backend_type SET NOT NULL"))
            conn.execute(text("ALTER TABLE model_registry ALTER COLUMN backend_type SET DEFAULT 'pgvector'"))

        if "metadata" not in columns:
            conn.execute(text("ALTER TABLE model_registry ADD COLUMN metadata JSON"))
            conn.execute(text("UPDATE model_registry SET metadata = '{}' WHERE metadata IS NULL"))
            conn.execute(text("ALTER TABLE model_registry ALTER COLUMN metadata SET NOT NULL"))

        if "created_at" not in columns:
            conn.execute(text("ALTER TABLE model_registry ADD COLUMN created_at TIMESTAMP"))
            conn.execute(text("UPDATE model_registry SET created_at = NOW() WHERE created_at IS NULL"))
            conn.execute(text("ALTER TABLE model_registry ALTER COLUMN created_at SET NOT NULL"))
            conn.execute(text("ALTER TABLE model_registry ALTER COLUMN created_at SET DEFAULT NOW()"))

        if "updated_at" not in columns:
            conn.execute(text("ALTER TABLE model_registry ADD COLUMN updated_at TIMESTAMP"))
            conn.execute(text("UPDATE model_registry SET updated_at = NOW() WHERE updated_at IS NULL"))
            conn.execute(text("ALTER TABLE model_registry ALTER COLUMN updated_at SET NOT NULL"))
            conn.execute(text("ALTER TABLE model_registry ALTER COLUMN updated_at SET DEFAULT NOW()"))
