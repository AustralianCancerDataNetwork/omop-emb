import os
import re
from typing import Literal, Optional, Type

from numpy import ndarray
from sqlalchemy import Engine, ForeignKey, Index, Integer, String, inspect, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, mapped_column, Mapped
from pgvector.sqlalchemy import Vector
from orm_loader.helpers import Base

from omop_alchemy.cdm.model.vocabulary.concept import Concept

import logging
logger = logging.getLogger(__name__)


IndexMethod = Literal["diskann", "hnsw", "ivfflat", "none"]
IndexMethodOrAuto = Literal["auto", "diskann", "hnsw", "ivfflat", "none"]
SUPPORTED_INDEX_METHODS = {"auto", "diskann", "hnsw", "ivfflat", "none"}
DEFAULT_INDEX_METHOD = "auto"
PGVECTOR_INDEX_MAX_DIMENSIONS = 2000

class ModelRegistry(Base):
    __tablename__ = "model_registry"
    
    name = mapped_column(String, primary_key=True)  # e.g., "nomic-embed-text-v1.5"
    dimensions = mapped_column(Integer, nullable=False) # e.g., 768
    table_name = mapped_column(String, unique=True, nullable=False) # e.g., "emb_nomic_v1_5"
    index_method = mapped_column(String, nullable=False, default="hnsw")

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


def _normalize_index_method(index_method: Optional[str]) -> IndexMethodOrAuto:
    method = (index_method or os.getenv("OMOP_EMB_INDEX_METHOD") or DEFAULT_INDEX_METHOD).strip().lower()
    if method not in SUPPORTED_INDEX_METHODS:
        raise ValueError(
            f"Unsupported index_method={index_method!r}. "
            f"Expected one of {sorted(SUPPORTED_INDEX_METHODS)}."
        )
    return method  # type: ignore[return-value]


def _has_extension(session: Session, extension_name: str) -> bool:
    stmt = text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = :extension_name)")
    return bool(
        session.execute(stmt, {"extension_name": extension_name}).scalar_one()
    )


def _has_extension_connection(connection, extension_name: str) -> bool:
    stmt = text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = :extension_name)")
    return bool(
        connection.execute(stmt, {"extension_name": extension_name}).scalar_one()
    )


def _resolve_index_method(
    session: Session,
    dimensions: Optional[int] = None,
    index_method: Optional[str] = None,
) -> IndexMethod:
    requested = _normalize_index_method(index_method)
    session.execute(text("CREATE EXTENSION IF NOT EXISTS vector CASCADE;"))

    if requested == "auto":
        if _has_extension(session, "vectorscale"):
            return "diskann"
        if dimensions is not None and dimensions > PGVECTOR_INDEX_MAX_DIMENSIONS:
            logger.warning(
                "Embedding dimension %s exceeds pgvector's %s-dimension ANN index limit for the "
                "'vector' type. Falling back to no index.",
                dimensions,
                PGVECTOR_INDEX_MAX_DIMENSIONS,
            )
            return "none"
        return "hnsw"

    if requested == "diskann":
        try:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;"))
        except Exception as exc:
            raise RuntimeError(
                "index_method='diskann' requires the PostgreSQL 'vectorscale' extension. "
                "Use OMOP_EMB_INDEX_METHOD=hnsw (or ivfflat/none) for plain pgvector installs."
            ) from exc

    if (
        requested in {"hnsw", "ivfflat"}
        and dimensions is not None
        and dimensions > PGVECTOR_INDEX_MAX_DIMENSIONS
    ):
        raise RuntimeError(
            f"index_method='{requested}' is not supported for embeddings with "
            f"{dimensions} dimensions when stored as pgvector 'vector'. "
            f"pgvector ANN indexes support up to {PGVECTOR_INDEX_MAX_DIMENSIONS} dimensions for this type. "
            "Use index_method='none', install vectorscale and use 'diskann', or redesign storage to use halfvec."
        )

    return requested


def _ensure_model_registry_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine, tables=[ModelRegistry.__table__])  # type: ignore[arg-type]

    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns(ModelRegistry.__tablename__)}
    if "index_method" in columns:
        return

    with engine.begin() as conn:
        legacy_default = "diskann" if _has_extension_connection(conn, "vectorscale") else "hnsw"
        # Existing rows were historically created with DiskANN only.
        conn.execute(text("ALTER TABLE model_registry ADD COLUMN index_method VARCHAR"))
        conn.execute(
            text("UPDATE model_registry SET index_method = :legacy_default WHERE index_method IS NULL"),
            {"legacy_default": legacy_default},
        )
        conn.execute(text("ALTER TABLE model_registry ALTER COLUMN index_method SET NOT NULL"))
        conn.execute(text("ALTER TABLE model_registry ALTER COLUMN index_method SET DEFAULT 'hnsw'"))


def _heal_legacy_index_method(session: Session, model_entry: ModelRegistry) -> None:
    if model_entry.index_method != "diskann":
        if (
            model_entry.index_method in {"hnsw", "ivfflat"}
            and model_entry.dimensions > PGVECTOR_INDEX_MAX_DIMENSIONS
        ):
            logger.warning(
                "Model '%s' is registered with index_method='%s' but has %s dimensions, "
                "which exceeds pgvector's %s-dimension ANN index limit for the 'vector' type. "
                "Falling back to 'none'.",
                model_entry.name,
                model_entry.index_method,
                model_entry.dimensions,
                PGVECTOR_INDEX_MAX_DIMENSIONS,
            )
            model_entry.index_method = "none"
        return
    if _has_extension(session, "vectorscale"):
        return

    fallback = "none" if model_entry.dimensions > PGVECTOR_INDEX_MAX_DIMENSIONS else "hnsw"
    logger.warning(
        "Model '%s' is registered with index_method='diskann' but vectorscale is not installed. "
        "Falling back to '%s' for compatibility.",
        model_entry.name,
        fallback,
    )
    model_entry.index_method = fallback


def _get_index(
    table_name: str,
    index_method: IndexMethod,
) -> Optional[Index]:
    index_name = f"idx_{table_name}"
    kwargs = {
        "postgresql_ops": {"embedding": "vector_cosine_ops"},
    }

    if index_method == "none":
        return None
    if index_method == "diskann":
        kwargs["postgresql_using"] = "diskann"
    elif index_method == "hnsw":
        kwargs["postgresql_using"] = "hnsw"
        kwargs["postgresql_with"] = {"m": 16, "ef_construction": 64}
    elif index_method == "ivfflat":
        kwargs["postgresql_using"] = "ivfflat"
        kwargs["postgresql_with"] = {"lists": 100}
    else:
        raise ValueError(f"Unsupported resolved index method: {index_method}")

    return Index(index_name, "embedding", **kwargs)


def _create_index_sql(table_name: str, index_method: IndexMethod) -> Optional[str]:
    index_name = f"idx_{table_name}"

    if index_method == "none":
        return None
    if index_method == "diskann":
        return (
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON {table_name} USING diskann (embedding vector_cosine_ops);"
        )
    if index_method == "hnsw":
        return (
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON {table_name} USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = 16, ef_construction = 64);"
        )
    if index_method == "ivfflat":
        return (
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON {table_name} USING ivfflat (embedding vector_cosine_ops) "
            f"WITH (lists = 100);"
        )
    raise ValueError(f"Unsupported resolved index method: {index_method}")

def initialize_embedding_tables(engine: Engine):
    """
    STARTUP HOOK: Reads the DB Registry and warms up the cache.
    Call this once when your app/script starts!
    """
    print("Initializing embedding models from registry...")
    _ensure_model_registry_schema(engine)

    with Session(engine, expire_on_commit=False) as session:
        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector CASCADE;"))
        existing_models = session.scalars(select(ModelRegistry)).all()
        for model_entry in existing_models:
            _heal_legacy_index_method(session, model_entry)
        session.commit()

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
    index_method = _normalize_index_method(registry_entry.index_method)
    index = _get_index(registry_entry.table_name, index_method if index_method != "auto" else "hnsw")
    table_args: list[object] = []
    if index is not None:
        table_args.append(index)
    table_args.append({"extend_existing": True})

    class_type =type(
        f"Embedding_{registry_entry.table_name}",
        (EmbeddingBase,),
        {
            "__tablename__": registry_entry.table_name,
            "embedding": mapped_column(Vector(registry_entry.dimensions)),
            "__table_args__": tuple(table_args)
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
        
        create_index_sql = _create_index_sql(registry_entry.table_name, index_method if index_method != "auto" else "hnsw")
        if f"idx_{registry_entry.table_name}" not in existing_indexes and create_index_sql is not None:
            conn.execute(text(create_index_sql))
            conn.commit()

    logger.debug(
        "Initialized embedding table for model '%s' with dimensions %s using index_method=%s",
        registry_entry.name,
        registry_entry.dimensions,
        index_method,
    )
    return class_type

def register_new_model(
    engine: Engine,
    model_name: str,
    dimensions: int,
    index_method: Optional[str] = None,
) -> Type[EmbeddingBase]:
    """
    RUNTIME HOOK: Registers a BRAND NEW model (DB + Cache).
    Use this when you onboard a new embedding model.
    """
    _ensure_model_registry_schema(engine)

    # 1. Check if already exists
    try:
        existing = get_embedding_model(model_name)
        return existing
    except KeyError:
        pass  # Not found, proceed to create

    with Session(engine) as session:
        resolved_index_method = _resolve_index_method(
            session,
            dimensions=dimensions,
            index_method=index_method,
        )
        session.commit()
    
    # 2. Add to Database Registry
    safe_name = get_save_model_name(model_name)
    table_name = f"emb_{safe_name}"
    
    new_entry = ModelRegistry(
        name=model_name,
        dimensions=dimensions,
        table_name=table_name,
        index_method=resolved_index_method,
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
