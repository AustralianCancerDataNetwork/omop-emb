from sqlalchemy import select, case, literal, Select, Engine, Integer, delete
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.sql import column, values, cast
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.orm import mapped_column, Mapped, Session
from pgvector.sqlalchemy import Vector
from orm_loader.helpers import Base
from omop_alchemy.cdm.model.vocabulary import Concept

from typing import Type, Tuple, Optional, TYPE_CHECKING, List, Union
import logging
from numpy import ndarray

from omop_emb.config import MetricType
from omop_emb.model_registry import EmbeddingModelRecord
from omop_emb.backends.base_backend import ConceptIDEmbeddingBase
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter, get_similarity_from_distance

logger = logging.getLogger(__name__)


_PGVECTOR_TABLE_CACHE: dict[str, type["PGVectorConceptIDEmbeddingTable"]] = {}
EMBEDDING_COLUMN_NAME = "embedding"


class PGVectorConceptIDEmbeddingTable(ConceptIDEmbeddingBase, Base):
    """Abstract base for Postgres linter support."""
    __abstract__ = True
    embedding: Mapped[list[float]]

def create_pg_embedding_table(
    engine: Engine,
    model_record: EmbeddingModelRecord,
) -> Type[PGVectorConceptIDEmbeddingTable]: # Note the specific Type hint here

    tablename = model_record.storage_identifier
    dimensions = model_record.dimensions

    cached_table = _PGVECTOR_TABLE_CACHE.get(tablename)
    if cached_table is not None:
        Base.metadata.create_all(engine, tables=[cached_table.__table__])  # type: ignore[arg-type]
        return cached_table

    table = type(
        f"PGVectorConceptIDEmbeddingTable_{tablename}",
        (PGVectorConceptIDEmbeddingTable,),
        {
            "__tablename__": tablename,
            EMBEDDING_COLUMN_NAME: mapped_column(Vector(dimensions), nullable=False, index=False),
            "__table_args__": {"extend_existing": True},
            "__module__": __name__,
        }
    )
    Base.metadata.create_all(engine, tables=[table.__table__])  # type: ignore[arg-type]
    _PGVECTOR_TABLE_CACHE[tablename] = table

    logger.debug(
        f"Initialized PGVectorConceptIDEmbeddingTable for model '{model_record.model_name}' "
        f"(dim={dimensions}, index_type={model_record.index_type.value})"
    )
    return table

def delete_pg_embedding_table(
    engine: Engine,
    model_record: EmbeddingModelRecord,
) -> None:
    """Deletes the pgvector embedding table for the specified model. Note that this only drops the table from the database; any associated metadata in the model registry should be handled separately."""
    tablename = model_record.storage_identifier
    cache_key = tablename

    embedding_table = _PGVECTOR_TABLE_CACHE.pop(cache_key, None)
    if embedding_table is not None:
        with engine.begin() as conn:
            stmt = delete(embedding_table)
            conn.execute(stmt)
            logger.info(f"Dropped SQL embedding registry table '{embedding_table.__tablename__}' for model '{model_record.model_name}'.")

def add_embeddings_to_registered_table(
    concept_ids: tuple[int, ...],
    embeddings: ndarray,
    session: Session,
    registered_table: type[PGVectorConceptIDEmbeddingTable],
):
    """Add embeddings to the storage table and update the registry to reflect the new concept_ids present in the index.
    
    Raises
    ------
    sqlalchemy.exc.IntegrityError
        If there is duplicate concept_id entries in the input, or if any of the concept_ids violate database constraints.
    """
    if session.bind is None:
        raise ValueError("Session must be bound to an engine to add embeddings to PGVector registry.")
    if session.bind.dialect.name != "postgresql":
        raise ValueError(f"This function is only implemented for PostgreSQL databases, but got {session.bind.dialect.name}.")

    insert_values = [
        {
            registered_table.concept_id.key: cid,
            registered_table.embedding.key: emb.tolist(),
        }
        for cid, emb in zip(concept_ids, embeddings)
    ]

    stmt = insert(registered_table).values(insert_values)
    session.execute(stmt)
    #upsert_stmt = stmt.on_conflict_do_update(
    #    index_elements=[registered_table.concept_id.key], 
    #    set_={registered_table.embedding.key: stmt.excluded.embedding}
    #)
    #session.execute(upsert_stmt)
    session.commit()


def q_embedding_nearest_concepts(
    embedding_table: Type[PGVectorConceptIDEmbeddingTable],
    query_embeddings: List[List[float]],
    metric_type: MetricType,
    concept_filter: Optional[EmbeddingConceptFilter] = None,
) -> Select:
    """Constructs a SQL query to retrieve the nearest concepts for the given query embeddings, applying the specified metric and filters. The query uses a LATERAL join to compute distances/similarities for each query vector against all candidate concept embeddings, and then applies the necessary OMOP filters before returning the top K results per query vector.
    
    Notes
    -----
    The query is designed to support multiple query embeddings at once. The outer list of the query embedding is of length Q (number of query vectors), and the inner list is of length D (dimensions of each embedding). The steps to construct the query are as follows:

    1. Create a virtual table of Q query vectors tagged with a query ID (q_id) for joining
    2. Define distance and similarity against the virtual query vector using the appropriate pgvector distance functions based on the specified metric
    3. Inner query: Retrieves the nearest concepts and computes standard/active flags, ordering by distance
    4. Apply OMOP filters to the inner query based on the provided concept_filter
    5. Limit the inner query to K results and convert it to a LATERAL subquery
    6. Join the Q vectors to their K nearest neighbors, returning the concept_id, concept_name, standard/active flags, and similarity score for each match. 'literal(True)' forces a CROSS JOIN LATERAL behavior.

    Parameters
    ----------
    embedding_table : Type[PGVectorConceptIDEmbeddingTable]
        The SQLAlchemy model class representing the embedding table for the relevant model.
    query_embeddings : List[List[float]]
        A list of query embeddings, where each embedding is a list of floats. The outer list should have length Q (number of query vectors), and each inner list should have length D (dimensions of the embedding).
    metric_type : MetricType
        The distance metric to use for nearest neighbor search (e.g., COSINE, L2, etc.). This will determine which pgvector distance function is used in the query.
    concept_filter : Optional[EmbeddingConceptFilter], optional
        An optional filter object containing criteria to filter the concepts (e.g., by concept_id, domain, vocabulary, standard_concept flag). Also is used to limit the number of nearest neighbors (K) returned per query vector. 
    limit : int, optional
        The number of nearest neighbors (K) to return for each query embedding, by default 10.
    """
    # Create a virtual table of Q query vectors tagged with a query ID (q_id) for joining
    query_data = [(i, q) for i, q in enumerate(query_embeddings)]
    query_v = values(
        column("q_id", Integer),
        column("q_vec", Vector),
        name="queries"
    ).data(query_data)

    query_vector_cast = cast(query_v.c.q_vec, Vector)
    distance = get_distance(embedding_table, query_vector_cast, metric_type)
    similarity = get_similarity_from_distance(distance, metric_type)

    if not isinstance(similarity, ColumnElement):
        raise TypeError(f"Expected similarity to be a SQL expression column for use in the query construction, but got {type(similarity)}")

    inner_stmt = (
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
            similarity.label("similarity"),
        )
        .join(embedding_table, Concept.concept_id == embedding_table.concept_id)
        .order_by(distance)
    )

    if concept_filter:
        inner_stmt = concept_filter.apply(inner_stmt)

    lateral_subq = inner_stmt.lateral("top_k")

    # Joins the Q vectors to their K nearest neighbors
    stmt = (
        select(
            query_v.c.q_id,
            lateral_subq.c.concept_id,
            lateral_subq.c.concept_name,
            lateral_subq.c.is_standard,
            lateral_subq.c.is_active,
            lateral_subq.c.similarity,
        )
        .select_from(query_v)
        .join(lateral_subq, literal(True))
    )
    
    return stmt


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

def get_distance(
    embedding_table: Type[PGVectorConceptIDEmbeddingTable],
    text_embedding: Union[list[float], ColumnElement],
    metric: MetricType,
):
    if metric == MetricType.COSINE:
        return embedding_table.embedding.cosine_distance(text_embedding)
    elif metric == MetricType.L2:
        return embedding_table.embedding.l2_distance(text_embedding)
    elif metric == MetricType.L1:
        return embedding_table.embedding.l1_distance(text_embedding)
    elif metric == MetricType.HAMMING:
        return embedding_table.embedding.hamming_distance(text_embedding)
    elif metric == MetricType.JACCARD:
        return embedding_table.embedding.jaccard_distance(text_embedding)
    else:
        raise ValueError(f"Unsupported metric type: {metric.value}")
    
