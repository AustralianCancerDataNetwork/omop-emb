from sqlalchemy import Index, Integer, String, select, ForeignKey, text, Engine, inspect
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, mapped_column, Mapped
from pgvector.sqlalchemy import Vector
from orm_loader.helpers import Base
from typing import Type
import re
from numpy import ndarray

from omop_alchemy.cdm.model.vocabulary.concept import Concept

import logging
logger = logging.getLogger(__name__)

class ModelRegistry(Base):
    __tablename__ = "model_registry"
    
    name = mapped_column(String, primary_key=True)  # e.g., "nomic-embed-text-v1.5"
    dimensions = mapped_column(Integer, nullable=False) # e.g., 768
    table_name = mapped_column(String, unique=True, nullable=False) # e.g., "emb_nomic_v1_5"

class EmbeddingBase(Base):
    __abstract__ = True
    
    concept_id = mapped_column(
        Integer, 
        ForeignKey(Concept.concept_id, ondelete="CASCADE"), 
        primary_key=True
    )

    # Type hint for ORM
    embedding: Mapped[list[float]]

_MODEL_CACHE: dict[str, Type[EmbeddingBase]] = {}

def initialize_embedding_tables(engine: Engine):
    """
    STARTUP HOOK: Reads the DB Registry and warms up the cache.
    Call this once when your app/script starts!
    """
    print("Initializing embedding models from registry...")

    query_add_embedding_column = text("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;")
    with Session(engine) as session:
        session.execute(query_add_embedding_column)
        session.commit()
        existing_models = session.scalars(select(ModelRegistry)).all()

    for model_entry in existing_models:
        if model_entry.name not in _MODEL_CACHE:
            # Create the class and cache it
            _create_embedding_class_dynamic(engine, model_entry)

def _create_embedding_class_dynamic(
    engine: Engine,
    registry_entry: ModelRegistry
) -> Type[EmbeddingBase]:
    """
    Internal factory: Turns a Registry row into a Python Class.
    """
    metadata = Base.metadata
    class_type =type(
        f"Embedding_{registry_entry.table_name}",
        (EmbeddingBase,),
        {
            "__tablename__": registry_entry.table_name,
            "embedding": mapped_column(Vector(registry_entry.dimensions)),
            "__table_args__": (
                # Define the DiskANN index for pgvectorscale
                Index(
                    f"idx_{registry_entry.table_name}",
                    "embedding",
                    postgresql_using="diskann",
                    postgresql_ops={"embedding": "vector_cosine_ops"} # or vector_l2_ops
                ),
                {"extend_existing": True}
            )
        }
    )
    # Add to cache
    _MODEL_CACHE[registry_entry.name] = class_type

    # Create (if not exists)
    metadata.create_all(engine, tables=[class_type.__table__])  # type: ignore

    # Make sure the index exists (sometimes create_all doesn't handle custom indexes well)
    with engine.connect() as conn:
        inspector = inspect(conn)
        existing_indexes = [idx['name'] for idx in inspector.get_indexes(registry_entry.table_name)]
        
        if f"idx_{registry_entry.table_name}" not in existing_indexes:
            # Manually trigger the index creation if it's missing
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{registry_entry.table_name} 
                ON {registry_entry.table_name} 
                USING diskann (embedding vector_cosine_ops);
            """))
            conn.commit()

    logger.debug(f"Initialized embedding table for model '{registry_entry.name}' with dimensions {registry_entry.dimensions}")
    return class_type

def register_new_model(engine: Engine, model_name: str, dimensions: int) -> Type[EmbeddingBase]:
    """
    RUNTIME HOOK: Registers a BRAND NEW model (DB + Cache).
    Use this when you onboard a new embedding model.
    """
    # 1. Check if already exists
    try:
        existing = get_embedding_model(model_name)
        return existing
    except KeyError:
        pass  # Not found, proceed to create
    
    # 2. Add to Database Registry
    safe_name = get_save_model_name(model_name)
    table_name = f"emb_{safe_name}"
    
    new_entry = ModelRegistry(
        name=model_name,
        dimensions=dimensions,
        table_name=table_name
    )

    with Session(engine, expire_on_commit=False) as session:
        session.add(new_entry)
        session.commit()
    
    DynamicClass = _create_embedding_class_dynamic(engine, new_entry)
    logger.info(f"Registered and created table for: {model_name}")
    return DynamicClass

def get_embedding_model(model_name: str) -> Type[EmbeddingBase]:
    """
    ACCESSOR: Just gets the class (assumes it's loaded).
    """
    if model_name not in _MODEL_CACHE:
        raise KeyError(f"Model '{model_name}' not found! Did you run initialize_models()?")
    return _MODEL_CACHE[model_name]


def get_save_model_name(model_name: str) -> str:
    name = model_name.lower()
    sanitized = re.sub(r'[^\w]+', '_', name)
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    return sanitized

def add_embeddings_to_registered_table(
    concept_ids: tuple[int, ...],
    embeddings: ndarray,
    session: Session,
    model: str | type[EmbeddingBase],
):
    
    assert session.bind is not None, "No engine assigned to session. Unexpected"
    assert session.bind.dialect.name == "postgresql", "Only postgres dialect supported for now."
    # TODO: Support other dialects

    if isinstance(model, str):
        model = get_embedding_model(model)

    insert_values = [
        {
            model.concept_id.key: cid,
            model.embedding.key: emb.tolist(),
        }
        for cid, emb in zip(concept_ids, embeddings)
    ]

    stmt = insert(model).values(insert_values)
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=[model.concept_id.key], 
        set_={model.embedding.key: stmt.excluded.embedding}
    )

    session.execute(upsert_stmt)
    session.commit()