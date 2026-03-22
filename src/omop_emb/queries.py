from typing import Optional, Tuple, Type

from sqlalchemy import case, literal, select
from sqlalchemy.sql import Select

from omop_alchemy.cdm.model.vocabulary import Concept

from omop_emb.cdm.embeddings import EmbeddingBase

def q_embedding_cosine_similarity(
    embedding_table: Type[EmbeddingBase],
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
    embedding_table: Type[EmbeddingBase],
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
    embedding_table: Type[EmbeddingBase],
    concept_ids: Tuple[int, ...],
) -> Select:
    return (
        select(
            embedding_table.concept_id,
            embedding_table.embedding,
        )
        .where(embedding_table.concept_id.in_(concept_ids))
    )
