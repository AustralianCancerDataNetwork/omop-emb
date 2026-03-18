from typing import Optional, Tuple
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from numpy import ndarray

from .queries import q_embedding_model_table_name, q_embedding_cosine_similarity
from .cdm.embeddings import (
    _MODEL_CACHE, 
    initialize_embedding_tables,
    add_embeddings_to_registered_table
)

class EmbeddingAccessor:
    """SQLAlchemy interface with an active session to obtain embedding """
       
    def get_model_table_name(
        self, 
        session: Session,
        model_name: str
        ) -> Optional[str]:
        query = session.execute(q_embedding_model_table_name(model_name)).all()
        table_names = [row.table_name for row in query]
        if len(table_names) > 1:
            raise RuntimeError(f"Multiple embedding model tables found for model_name='{model_name}'.")
        return table_names[0] if table_names else None
    
    def is_model_registered(
        self, 
        session: Session,
        model_name: str
    ) -> bool:
        return self.get_model_table_name(session=session, model_name=model_name) is not None
    
    def get_similarities(
        self,
        session: Session,
        embedding_model_name: str,
        text_embedding: list[float],
        concept_ids: Optional[Tuple[int, ...]] = None
    ):
        embedding_table = _MODEL_CACHE.get(embedding_model_name)
        if embedding_table is None:
            raise ValueError(f"Embedding model '{embedding_model_name}' not found in cache.")
        
        query = q_embedding_cosine_similarity(
            embedding_table=embedding_table,
            text_embedding=text_embedding,
            concept_ids=concept_ids
        )

        return {row.concept_id: row.similarity for row in session.execute(query).all()}
    
    def initialise_tables(self, engine: Engine):
        return initialize_embedding_tables(engine)
    
    def add_to_db(
        self,
        session: Session,
        concept_ids: Tuple[int, ...],
        embeddings: ndarray,
        model: str
    ):
        assert embeddings.ndim == 2, f"Expected 2 dimensions of embeddings. Got {embeddings.ndim}"
        assert len(concept_ids) == embeddings.shape[0], f"Mismatch between #concept_ids ({len(concept_ids)} and embedding dimensionality ({embeddings.shape[0]})"
        
        return add_embeddings_to_registered_table(
            concept_ids=concept_ids,
            embeddings=embeddings,
            session=session,
            model=model
        )