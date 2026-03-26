from __future__ import annotations

from typing import Type, Optional, TYPE_CHECKING
import logging
from sqlalchemy import  Engine, text, select, Select, case, literal
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from orm_loader.helpers import Base
from omop_alchemy.cdm.model.vocabulary import Concept

from ..base import ConceptIDEmbeddingBase
from ..config import BackendType
from ..registry import ModelRegistry

if TYPE_CHECKING:
    # Circular Import - might require some better solution in the future
    # Separate sql_utils.py?
    from omop_emb.backends.base import EmbeddingConceptFilter

logger = logging.getLogger(__name__)


class FAISSConceptIDEmbeddingRegistry(ConceptIDEmbeddingBase, Base):
    """Registry table to track which concept_ids are present in the FAISS/H5 storage for a specific model."""
    __abstract__ = True

def create_faiss_embedding_registry_table(
    engine: Engine,
    model_registry_entry: ModelRegistry, 
) -> Type[FAISSConceptIDEmbeddingRegistry]:
    """
    Creates a dynamic SQLAlchemy ORM class that tracks which concept_ids 
    are present in the FAISS/H5 storage for a specific model.
    """
    table_name = model_registry_entry.storage_identifier

    mapping_table = type(
        f"{BackendType.FAISS.value.capitalize()}_{table_name}",
        (FAISSConceptIDEmbeddingRegistry, ),
        {
            "__tablename__": table_name,
            "__table_args__": {"extend_existing": True},
        },
    )
    
    Base.metadata.create_all(engine, tables=[mapping_table.__table__])  # type: ignore[arg-type]
    
    logger.debug(
        f"Initialized {FAISSConceptIDEmbeddingRegistry.__name__} table for model '{model_registry_entry.model_name}'",
    )

    return mapping_table

def initialise_faiss_mapping_tables(
    engine: Engine,
    model_cache: dict[str, Type[FAISSConceptIDEmbeddingRegistry]],
) -> None:
    with Session(engine, expire_on_commit=False) as session:
        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector CASCADE;"))
        existing_models = session.scalars(select(ModelRegistry).where(ModelRegistry.backend_type == BackendType.FAISS.value)).all()
        session.commit()

    for model_entry in existing_models:
        if model_entry.model_name not in model_cache:
            # Create the class and cache it
            dynamic_table = create_faiss_embedding_registry_table(engine=engine, model_registry_entry=model_entry)
            model_cache[model_entry.model_name] = dynamic_table

def add_concept_ids_to_faiss_registry(
    concept_ids: tuple[int, ...],
    session: Session,
    registered_table: type[FAISSConceptIDEmbeddingRegistry],
):
    r"""Adds the given concept_ids to the FAISS registry table for the specified model. This is used to keep track of which concept_ids are present in the FAISS/H5 storage, as we don't have direct access to the contents of the index like we do with a SQL database.
    
    Raises
    ------
    sqlalchemy.exc.IntegrityError
        If there is an attempt to add a concept_id that already exists in the registry, which indicates a mismatch between the registry and the actual contents of the FAISS index. This is a safeguard as partial updates and overwrites are not yet supported.
    """

    assert session.bind is not None, "Session must be bound to an engine"
    assert session.bind.dialect.name == "postgresql", "This function is only implemented for PostgreSQL databases"

    stmt = insert(registered_table).values(list(concept_ids))
    session.execute(stmt)
    session.commit()

def q_concept_ids_with_embeddings(
    embedding_table: Type[FAISSConceptIDEmbeddingRegistry],
    concept_filter: Optional[EmbeddingConceptFilter] = None,
    limit: Optional[int] = None,
) -> Select:
    
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
        )
        .join(embedding_table, Concept.concept_id == embedding_table.concept_id)
    )

    if concept_filter is not None:
        stmt = concept_filter.apply(stmt)
    return stmt.limit(limit)
