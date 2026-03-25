from sqlalchemy import select, case, literal, Select, text, Engine, inspect
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import mapped_column, Mapped, Session
from pgvector.sqlalchemy import Vector
from orm_loader.helpers import Base
from omop_alchemy.cdm.model.vocabulary import Concept

from typing import Type, Tuple, Optional
import logging
from numpy import ndarray

from ..config import BackendType, IndexType
from ..registry import ModelRegistry
from ..base import ConceptIDEmbeddingBase

logger = logging.getLogger(__name__)


class PGVectorConceptIDEmbeddingTable(ConceptIDEmbeddingBase, Base):
    """Abstract base for Postgres linter support."""
    __abstract__ = True
    embedding: Mapped[list[float]]

def create_pg_embedding_table(
    engine: Engine,
    model_registry_entry: ModelRegistry, 
) -> Type[PGVectorConceptIDEmbeddingTable]: # Note the specific Type hint here

    table_name = model_registry_entry.storage_identifier
    dimensions = model_registry_entry.dimensions
    
    table = type(
        f"{BackendType.PGVECTOR.value.capitalize}_{table_name}",
        (PGVectorConceptIDEmbeddingTable,), # It already inherits from Base and ConceptIDBase
        {
            "__tablename__": table_name,
            # We override the attribute with the specific dimension-aware column
            "embedding": mapped_column(Vector(dimensions), nullable=False, index=False),
            "__table_args__": {"extend_existing": True},
        }
    )
    Base.metadata.create_all(table.metadata, tables=[table.__table__])  # type: ignore[arg-type]

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
        f"Initialized {PGVectorConceptIDEmbeddingTable.__name__} table for model '{model_registry_entry.model_name}' with dimensions {model_registry_entry.dimensions} using index_method={model_registry_entry.index_type}",
    )

    return table

def initialise_pg_embedding_tables(
    engine: Engine,
    model_cache: dict[str, Type[PGVectorConceptIDEmbeddingTable]],
) -> None:
    with Session(engine, expire_on_commit=False) as session:
        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector CASCADE;"))
        existing_models = session.scalars(select(ModelRegistry).where(ModelRegistry.backend_type == BackendType.PGVECTOR.value)).all()
        #for model_entry in existing_models:
        #    _heal_legacy_index_method(session, model_entry)
        session.commit()

    for model_entry in existing_models:
        if model_entry.model_name not in model_cache:
            # Create the class and cache it
            dynamic_table = create_pg_embedding_table(engine=engine, model_registry_entry=model_entry)
            model_cache[model_entry.model_name] = dynamic_table

def add_embeddings_to_registered_table(
    concept_ids: tuple[int, ...],
    embeddings: ndarray,
    session: Session,
    model: type[PGVectorConceptIDEmbeddingTable],
):
    assert session.bind is not None, "No engine assigned to session. Unexpected"
    assert session.bind.dialect.name == "postgresql", "Only postgres dialect supported for now."

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


def _create_index_sql(table_name: str, index_type: IndexType) -> Optional[str]:
    index_name = _index_from_storage_identifier(table_name)
    if index_type == IndexType.DISKANN:
        return (
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON {table_name} USING diskann (embedding vector_cosine_ops);"
        )
    if index_type == IndexType.HNSW:
        return (
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON {table_name} USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = 16, ef_construction = 64);"
        )
    # TODO: Not sure how to implement exactly here
    #if index_type == IndexType.FLAT:
    #    return (
    #        f"CREATE INDEX IF NOT EXISTS {index_name} "
    #        f"ON {table_name} USING ivfflat (embedding vector_cosine_ops) "
    #        f"WITH (lists = 100);"
    #    )
    raise ValueError(f"Unsupported resolved index method: {index_type.value}")

def _index_from_storage_identifier(storage_identifier: str) -> str:
    return "idx_" + storage_identifier

def q_embedding_cosine_similarity(
    embedding_table: Type[PGVectorConceptIDEmbeddingTable],
    text_embedding: list[float],
    concept_ids: Optional[Tuple[int, ...]] = None,
    limit: Optional[int] = None,
):
    
    distance = embedding_table.embedding.cosine_distance(text_embedding)
    stmt = (
        select(
            Concept.concept_id,
            (1 - distance).label("similarity")
        )
        .join(embedding_table, Concept.concept_id == embedding_table.concept_id)  # type: ignore
        .order_by(distance)
    )

    # 4. Add your optional where clause
    if concept_ids:
        stmt = stmt.where(Concept.concept_id.in_(concept_ids))
        limit = len(concept_ids)

    if limit is not None:
        stmt = stmt.limit(limit)
    return stmt


def q_embedding_nearest_concepts(
    embedding_table: Type[PGVectorConceptIDEmbeddingTable],
    text_embedding: list[float],
    concept_ids: Optional[Tuple[int, ...]] = None,
    domains: Optional[Tuple[str, ...]] = None,
    vocabularies: Optional[Tuple[str, ...]] = None,
    require_standard: bool = False,
    limit: int = 10,
) -> Select:
    distance = embedding_table.embedding.cosine_distance(text_embedding)
    stmt = (
        select(
            Concept.concept_id,
            Concept.concept_name,
            case(
                (Concept.standard_concept.in_(["S", "C"]), literal(True)),
                else_=literal(False),
            ).label("is_standard"),
            case(
                (Concept.invalid_reason.in_(["D", "U"]), literal(False)),
                else_=literal(True),
            ).label("is_active"),
            (1 - distance).label("similarity"),
        )
        .join(embedding_table, Concept.concept_id == embedding_table.concept_id)  # type: ignore
        .order_by(distance)
    )

    if concept_ids:
        stmt = stmt.where(Concept.concept_id.in_(concept_ids))

    if domains is not None:
        stmt = stmt.where(Concept.domain_id.in_(domains))

    if vocabularies is not None:
        stmt = stmt.where(Concept.vocabulary_id.in_(vocabularies))

    if require_standard:
        stmt = stmt.where(Concept.standard_concept.in_(["S", "C"]))

    return stmt.limit(limit)


def q_embedding_vectors_by_concept_ids(
    embedding_table: Type[PGVectorConceptIDEmbeddingTable],
    concept_ids: Tuple[int, ...],
) -> Select:
    return (
        select(
            embedding_table.concept_id,
            embedding_table.embedding,
        )
        .where(embedding_table.concept_id.in_(concept_ids))
    )