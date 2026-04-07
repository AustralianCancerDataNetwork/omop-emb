from sqlalchemy import Engine, Integer, ForeignKey, Result
from sqlalchemy.orm import mapped_column, Session
from typing import Mapping, TypeVar, Generic, Type, Dict, Optional
from abc import ABC, abstractmethod

from omop_alchemy.cdm.model.vocabulary import Concept

from .base_backend import EmbeddingBackendBase, require_registered_model
from omop_emb.config import IndexType, MetricType
from ..model_registry import EmbeddingModelRecord
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter


class ConceptIDEmbeddingBase:
    """Abstract Mixin to ensure consistent concept_id handling across backends."""
    __tablename__: str

    concept_id = mapped_column(
        Integer, 
        ForeignKey(Concept.concept_id, ondelete="CASCADE"), 
        primary_key=True
    )
    
T = TypeVar("T", bound=ConceptIDEmbeddingBase)


class DatabaseEmbeddingBackend(EmbeddingBackendBase, ABC, Generic[T]):
    """
    Base class for embedding backends that store embeddings in a relational database.

    This class defines the common interface and shared logic for database-backed embedding backends.
    Subclasses must implement the backend-specific storage and retrieval logic.
    """

    def __abs__(self):
        super().__init__()
        self._db_model_cache: Dict[str, Type[T]] = {}

    @property
    def model_cache(self) -> Dict[str, Type[T]]:
        return self._db_model_cache

    @abstractmethod
    def _create_storage_table(self, engine, model_record: EmbeddingModelRecord) -> Type[T]:
        """Create the backend-specific storage table for a given model registry entry."""
        ...

    @require_registered_model
    def has_any_embeddings(self, session: Session, model_name: str, index_type: IndexType, model_record: EmbeddingModelRecord) -> bool:
        embedding_table = self._get_embedding_table(
            session=session,
            model_name=model_name,
        )
        return session.query(embedding_table.concept_id).limit(1).first() is not None

    def initialise_store(self, engine) -> None:
        super().initialise_store(engine)

        existing_models = self.model_registry.get_registered_models(
            backend_type=self.backend_type
        )

        if not existing_models:
            return

        for model_record in existing_models:
            if model_record.model_name not in self.model_cache:
                dynamic_table = self._create_storage_table(
                    engine=engine, model_record=model_record
                )
                self.model_cache[model_record.model_name] = dynamic_table


    def register_model(
        self, 
        engine: Engine,
        model_name: str, 
        dimensions: int, 
        *, 
        index_type: IndexType, 
        metadata: Mapping[str, object] = {}
    ) -> EmbeddingModelRecord:
        record = super().register_model(
            engine=engine,
            model_name=model_name, 
            dimensions=dimensions, 
            index_type=index_type, 
            metadata=metadata
        )

        # Populate cache
        dynamic_table = self._create_storage_table(engine, record)
        self.model_cache[model_name] = dynamic_table

        return record
    
    def _get_embedding_table(
        self,
        session: Session,
        model_name: str,
    ) -> Type[T]:
        embedding_table = self.model_cache.get(model_name)

        if embedding_table is not None:
            return embedding_table
 
        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Session is not bound to an engine.")
        assert isinstance(bind, Engine), f"Expected session bind to be an Engine. Got {type(bind)}"
        
        self.initialise_store(bind)  # Ensure tables are created and cache is populated
        embedding_table = self.model_cache.get(model_name)
        if embedding_table is None:
            raise ValueError(f"Embedding model '{model_name}' not found in cache.")
        return embedding_table 
    
    @abstractmethod
    @require_registered_model
    def get_concepts_without_embedding(
        self,
        session: Session,
        model_name: str,
        index_type: IndexType,
        model_record: EmbeddingModelRecord,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: Optional[int] = None,
        metric_type: Optional[MetricType] = None,
    ) -> Result:
        """Return concept IDs and names for concepts that do not have embeddings."""
        ...
    
    @abstractmethod
    @require_registered_model
    def get_concepts_without_embedding_count(
        self,
        session: Session,
        model_name: str,
        index_type: IndexType,
        model_record: EmbeddingModelRecord,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        metric_type: Optional[MetricType] = None,
    ) -> int:
        ...