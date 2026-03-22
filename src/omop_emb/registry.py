from __future__ import annotations

from sqlalchemy import Engine, Integer, String, inspect, text
from sqlalchemy.orm import mapped_column

from orm_loader.helpers import Base


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

    name = mapped_column(String, primary_key=True)
    dimensions = mapped_column(Integer, nullable=False)
    storage_identifier = mapped_column("table_name", String, unique=True, nullable=False)
    index_method = mapped_column(String, nullable=False, default="hnsw")
    backend_name = mapped_column(String, nullable=False, default="postgres")


def ensure_model_registry_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine, tables=[ModelRegistry.__table__])  # type: ignore[arg-type]

    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns(ModelRegistry.__tablename__)}

    with engine.begin() as conn:
        if "index_method" not in columns:
            conn.execute(text("ALTER TABLE model_registry ADD COLUMN index_method VARCHAR"))
            conn.execute(
                text("UPDATE model_registry SET index_method = 'hnsw' WHERE index_method IS NULL")
            )
            conn.execute(text("ALTER TABLE model_registry ALTER COLUMN index_method SET NOT NULL"))
            conn.execute(text("ALTER TABLE model_registry ALTER COLUMN index_method SET DEFAULT 'hnsw'"))

        if "backend_name" not in columns:
            conn.execute(text("ALTER TABLE model_registry ADD COLUMN backend_name VARCHAR"))
            conn.execute(
                text("UPDATE model_registry SET backend_name = 'postgres' WHERE backend_name IS NULL")
            )
            conn.execute(text("ALTER TABLE model_registry ALTER COLUMN backend_name SET NOT NULL"))
            conn.execute(text("ALTER TABLE model_registry ALTER COLUMN backend_name SET DEFAULT 'postgres'"))
