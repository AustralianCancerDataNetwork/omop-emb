from __future__ import annotations

from typing import Type, Optional, TYPE_CHECKING
import logging
from sqlalchemy import  Engine, text, select, Select, case, literal
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from orm_loader.helpers import Base
from omop_alchemy.cdm.model.vocabulary import Concept

from omop_emb.model_registry import EmbeddingModelRecord
from omop_emb.model_registry import get_metadata_schema
from ..base import ConceptIDEmbeddingBase
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter

logger = logging.getLogger(__name__)


_FAISS_REGISTRY_TABLE_CACHE: dict[str, type["FAISSConceptIDEmbeddingRegistry"]] = {}


class FAISSConceptIDEmbeddingRegistry(ConceptIDEmbeddingBase, Base):
    """Registry table to track which concept_ids are present in the FAISS/H5 storage for a specific model."""
    __abstract__ = True

def create_faiss_embedding_registry_table(
    engine: Engine,
    model_record: EmbeddingModelRecord, 
) -> Type[FAISSConceptIDEmbeddingRegistry]:
    """
    Creates a dynamic SQLAlchemy ORM class that tracks which concept_ids 
    are present in the FAISS/H5 storage for a specific model.
    """
    tablename = model_record.storage_identifier

    cached_table = _FAISS_REGISTRY_TABLE_CACHE.get(tablename)
    if cached_table is not None:
        Base.metadata.create_all(engine, tables=[cached_table.__table__])  # type: ignore[arg-type]
        return cached_table

    mapping_table = type(
        f"FAISSConceptIDEmbeddingRegistry_{tablename}",
        (FAISSConceptIDEmbeddingRegistry, ),
        {
            "__tablename__": tablename,
            "__table_args__": {
                "extend_existing": True,
                "schema": get_metadata_schema(),
            },
            "__module__": __name__,
        },
    )
    
    Base.metadata.create_all(engine, tables=[mapping_table.__table__])  # type: ignore[arg-type]
    _FAISS_REGISTRY_TABLE_CACHE[tablename] = mapping_table
    
    logger.debug(
        f"Initialized {FAISSConceptIDEmbeddingRegistry.__name__} table for model '{model_record.model_name}'",
    )

    return mapping_table

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
    if session.bind is None:
        raise ValueError("Session must be bound to an engine to add concept IDs to FAISS registry.")
    if session.bind.dialect.name != "postgresql":
        raise ValueError(f"This function is only implemented for PostgreSQL databases, but got {session.bind.dialect.name}.")

    stmt = insert(registered_table).values(list({"concept_id": cid} for cid in concept_ids))
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
