from sqlalchemy import select, case, literal, Select, text, Engine, inspect, Integer
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

from ..config import BackendType, IndexType, MetricType
from ..registry import ModelRegistry
from ..base import ConceptIDEmbeddingBase
from ..embedding_utils import EmbeddingConceptFilter

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
        f"{BackendType.PGVECTOR.value.upper()}_{table_name}",
        (PGVectorConceptIDEmbeddingTable,), # It already inherits from Base and ConceptIDBase
        {
            "__tablename__": table_name,
            # We override the attribute with the specific dimension-aware column
            "embedding": mapped_column(Vector(dimensions), nullable=False, index=False),
            "__table_args__": {"extend_existing": True},
        }
    )
    Base.metadata.create_all(engine, tables=[table.__table__])  # type: ignore[arg-type]

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
    registered_table: type[PGVectorConceptIDEmbeddingTable],
):
    """Add embeddings to the storage table and update the registry to reflect the new concept_ids present in the index.
    
    Raises
    ------
    sqlalchemy.exc.IntegrityError
        If there is duplicate concept_id entries in the input, or if any of the concept_ids violate database constraints.
    """
    assert session.bind is not None, "No engine assigned to session. Unexpected"
    assert session.bind.dialect.name == "postgresql", "Only postgres dialect supported for now."

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


def _create_index_sql(table_name: str, index_type: IndexType) -> Optional[str]:
    if index_type == IndexType.FLAT:
        pass # No additional index needed for flat, as the vector column itself can be used for sequential scan
    else:
        raise ValueError(f"Unsupported resolved index method: {index_type.value}")

def _index_from_storage_identifier(storage_identifier: str) -> str:
    return "idx_" + storage_identifier

def q_embedding_nearest_concepts(
    embedding_table: Type[PGVectorConceptIDEmbeddingTable],
    query_embeddings: List[List[float]],
    metric_type: MetricType,
    concept_filter: Optional[EmbeddingConceptFilter] = None,
    limit: int = 10,
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
        An optional filter object containing criteria to filter the concepts (e.g., by concept_id, domain, vocabulary, standard_concept flag).
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
        if concept_filter.concept_ids is not None:
            inner_stmt = inner_stmt.where(Concept.concept_id.in_(concept_filter.concept_ids))
        if concept_filter.domains is not None:
            inner_stmt = inner_stmt.where(Concept.domain_id.in_(concept_filter.domains))
        if concept_filter.vocabularies is not None:
            inner_stmt = inner_stmt.where(Concept.vocabulary_id.in_(concept_filter.vocabularies))
        if concept_filter.require_standard:
            inner_stmt = inner_stmt.where(Concept.standard_concept.in_(["S", "C"]))

    lateral_subq = inner_stmt.limit(limit).lateral("top_k")

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
    
def get_similarity_from_distance(distance_col, metric: MetricType):
    """
    Helper to map various distance metrics to a 0.0 - 1.0 similarity score.
    """
    if metric == MetricType.COSINE:
        return 1.0 - distance_col
    
    elif metric == MetricType.L2:
        return 1.0 / (1.0 + distance_col)
    
    elif metric == MetricType.L1:
        return 1.0 / (1.0 + distance_col)
    
    elif metric == MetricType.HAMMING:
        # Hamming distance is the number of differing bits.
        # To get similarity, we need to know total bits (dimensions)
        # Assuming you want a normalized score: 1 - (dist / dim)
        # Note: This requires passing 'dimensions' into the helper
        raise NotImplementedError()
        return 1.0 - (distance_col / dimensions) 
        
    elif metric == MetricType.JACCARD:
        return 1.0 - distance_col
        
    else:
        raise ValueError(f"Unsupported metric type: {metric.value}")