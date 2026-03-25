from __future__ import annotations

from typing import Type, Optional
import logging
from sqlalchemy import Integer, Engine, inspect, text, select
from sqlalchemy.orm import Mapped, mapped_column, Session

from orm_loader.helpers import Base

from ..base import ConceptIDEmbeddingBase
from ..config import BackendType, IndexType
from ..registry import ModelRegistry

logger = logging.getLogger(__name__)


class FAISSConceptIDEmbeddingMapping(ConceptIDEmbeddingBase, Base):
    """Mapping table between concept_ids and FAISS index positions."""
    __abstract__ = True
    index_position: Mapped[int]

def create_faiss_mapping_table(
    engine: Engine,
    model_registry_entry: ModelRegistry, 
) -> Type[FAISSConceptIDEmbeddingMapping]:
    """
    Create or retrieve a SQLAlchemy ORM class for the mapping table of concept_ids to FAISS index positions.
    The table name is derived from the model name to ensure uniqueness and traceability. 
    """

    table_name = model_registry_entry.storage_identifier

    mapping_table = type(
        f"{BackendType.FAISS.value.capitalize()}_{table_name}",
        (FAISSConceptIDEmbeddingMapping, ),
        {
            "__tablename__": table_name,
            "__table_args__": {"extend_existing": True},
            "index_position": mapped_column(Integer, nullable=False, unique=True, index=True),
        },
    )
    Base.metadata.create_all(mapping_table.metadata, tables=[mapping_table.__table__])  # type: ignore[arg-type]

    with engine.connect() as conn:
        inspector = inspect(conn)
        existing_indexes = [idx['name'] for idx in inspector.get_indexes(model_registry_entry.storage_identifier)]
        

        create_index_sql = _create_index_sql(
            model_registry_entry.storage_identifier,
            IndexType(model_registry_entry.index_type),
        )
        if (
            _index_from_storage_identifier(model_registry_entry.storage_identifier) not in existing_indexes and
            create_index_sql is not None
        ):
            conn.execute(text(create_index_sql))
            conn.commit()

    logger.debug(
        f"Initialized {FAISSConceptIDEmbeddingMapping.__name__} table for model '{model_registry_entry.model_name}' with dimensions {model_registry_entry.dimensions} using index_method={model_registry_entry.index_type}",
    )

    return mapping_table

def initialise_faiss_mapping_tables(
    engine: Engine,
    model_cache: dict[str, Type[FAISSConceptIDEmbeddingMapping]],
) -> None:
    with Session(engine, expire_on_commit=False) as session:
        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector CASCADE;"))
        existing_models = session.scalars(select(ModelRegistry).where(ModelRegistry.backend_type == BackendType.FAISS.value)).all()
        #for model_entry in existing_models:
        #    _heal_legacy_index_method(session, model_entry)
        session.commit()

    for model_entry in existing_models:
        if model_entry.model_name not in model_cache:
            # Create the class and cache it
            dynamic_table = create_faiss_mapping_table(engine=engine, model_registry_entry=model_entry)
            model_cache[model_entry.model_name] = dynamic_table

def _index_from_storage_identifier(storage_identifier: str) -> str:
    return "idx_" + storage_identifier

def _create_index_sql(table_name: str, index_type: IndexType) -> Optional[str]:
    index_name = _index_from_storage_identifier(table_name)
    if index_type == IndexType.HNSW:
        return (
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON {table_name} USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = 16, ef_construction = 64);"
        )
    if index_type == IndexType.IVF_FLAT:
        return (
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON {table_name} USING ivfflat (embedding vector_cosine_ops) "
            f"WITH (lists = 100);"
        )
    raise ValueError(f"Unsupported resolved index method: {index_type.value}")

