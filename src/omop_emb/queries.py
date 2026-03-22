from typing import Optional, Tuple, Type

from sqlalchemy import select, func
from sqlalchemy.sql import Select

from omop_alchemy.cdm.model.vocabulary import Concept

from omop_emb.cdm.embeddings import ModelRegistry, EmbeddingBase


def q_embedding_model_table_name(model_name: str) -> Select:
    """Query to get the table name of an embedding model in the database."""
    return select(ModelRegistry.table_name).where(ModelRegistry.name == model_name)

def q_embedding_cosine_similarity(
    embedding_table: Type[EmbeddingBase],
    text_embedding: list[float],
    concept_ids: Optional[Tuple[int, ...]] = None,
    limit: int = 10
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

    # Limit the number of results return to the top N most similar concepts
    stmt = stmt.limit(limit)
    return stmt


def q_get_concepts_without_embedding(
    embedding_table: Type[EmbeddingBase],
    limit: Optional[int] = None
) -> Select:
    """Query to get concepts using NOT EXISTS to avoid joining large tables."""
    
    subq = select(1).where(embedding_table.concept_id == Concept.concept_id)
    
    query = (
        select(Concept.concept_id, Concept.concept_name)
        .where(~subq.exists())
    )

    if limit is not None:
        query = query.limit(limit)
    return query

def q_count_missing_concepts(
    embedding_table: Type[EmbeddingBase]
) -> Select:
    """Query to calculate the total count of missing concepts using NOT EXISTS."""
    
    subq = select(1).where(embedding_table.concept_id == Concept.concept_id)
    
    return (
        select(func.count())
        .select_from(Concept)
        .where(~subq.exists())
    )