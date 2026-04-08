from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Optional, Sequence, Union, Type, TypeVar, Generic, Dict, Any, Callable, Tuple
from pathlib import Path
from functools import wraps
import os

from numpy import ndarray
from sqlalchemy import Engine, select, Integer, ForeignKey, func, Select, text
from sqlalchemy.orm import Session, mapped_column

from omop_alchemy.cdm.model.vocabulary import Concept

from ..model_registry import EmbeddingModelRecord, ModelRegistryManager
from ..config import BackendType, IndexType, MetricType, ENV_BASE_STORAGE_DIR
from omop_emb.utils.embedding_utils import (
    EmbeddingConceptFilter, 
    NearestConceptMatch, 
)

def require_registered_model(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(
        self, 
        model_name: str, 
        index_type: IndexType,
        *args, **kwargs
    ) -> Any:
        record = self.get_registered_model(model_name=model_name, index_type=index_type)
        
        if record is None:
            raise ValueError(
                f"Embedding model '{model_name}' with index_type='{index_type.value}' is not registered in backend '{self.backend_type.value}'."
            )
        kwargs["_model_record"] = record
        return func(self, model_name, index_type, *args, **kwargs)
        
    return wrapper

class ConceptIDEmbeddingBase:
    """Abstract Mixin to ensure consistent concept_id handling across backends."""
    __tablename__: str

    concept_id = mapped_column(
        Integer, 
        ForeignKey(Concept.concept_id, ondelete="CASCADE"), 
        primary_key=True
    )
    
T = TypeVar("T", bound=ConceptIDEmbeddingBase)

class EmbeddingBackend(ABC, Generic[T]):
    """
    Abstract interface for swappable embedding storage and retrieval backends.

    Design goals
    ------------
    - Keep embedding generation separate from embedding persistence.
    - Support multiple storage/retrieval implementations behind one contract.
    - Local storage of model registry metadata, but allow flexible storage of the actual embeddings (e.g., relational tables, FAISS indices, files in object storage, etc.)

    Notes
    -----
    This interface intentionally still accepts SQLAlchemy ``Engine`` and
    ``Session`` objects. Even with a non-Postgres embedding index such as
    FAISS, the implementation will usually still need OMOP relational access
    to validate models, resolve concept metadata, and apply domain/vocabulary
    filters.
    """
    DEFAULT_BASE_STORAGE_DIR = str(Path.home() / ".omop_emb")
    def __init__(
        self,
        storage_base_dir: Optional[str | Path] = None,
        registry_db_name: Optional[str] = None,
    ):
        super().__init__()
        
        # Local storage for model registry database and more
        storage_base_dir = storage_base_dir or os.getenv(ENV_BASE_STORAGE_DIR) or self.DEFAULT_BASE_STORAGE_DIR
        self._storage_base_dir = Path(storage_base_dir)
        self._storage_base_dir.mkdir(parents=True, exist_ok=True)

        
        self._embedding_table_cache: dict[Tuple[str, BackendType, IndexType], Type[T]] = {}
        self._embedding_model_registry = ModelRegistryManager(
            base_dir=self.storage_base_dir,
            db_file=registry_db_name,
        )

    @property
    def storage_base_dir(self) -> Path:
        """Base directory for any local storage needs of the backend, such as model registry metadata or file-based embedding storage."""
        return self._storage_base_dir

    @property
    def embedding_table_cache(self) -> Dict[Tuple[str, BackendType, IndexType], Type[T]]:
        """In-memory cache of dynamically generated ORM classes for embedding tables."""
        return self._embedding_table_cache
    
    @property
    def embedding_model_registry(self) -> ModelRegistryManager:
        """Manager for embedding model metadata and registry operations."""
        return self._embedding_model_registry

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """General category of backend, used for index compatibility checks."""

    @property
    def backend_name(self) -> str:
        """Stable identifier for this backend implementation."""
        return self.backend_type.value
    
    def _cache_model_record(
        self,
        engine: Engine,
        model_record: EmbeddingModelRecord,
    ) -> None:
        """Cache the given model record in the embedding table cache. This is used to keep track of which models are registered and their associated metadata."""
        dynamic_table = self._create_storage_table(engine=engine, model_record=model_record)
        storage_key = (model_record.model_name, self.backend_type, model_record.index_type)
        self.embedding_table_cache[storage_key] = dynamic_table

    
    @abstractmethod
    def _create_storage_table(self, engine: Engine, model_record: EmbeddingModelRecord) -> Type[T]:
        """
        Backend-specific logic to create the dynamic SQLAlchemy class.

        Parameters
        ----------
        engine : Engine
            SQLAlchemy engine for OMOP CDM database storage.
        model_record : EmbeddingModelRecord
            Registered model metadata used to build backend storage.
        """

    def initialise_store(self, engine: Engine) -> None:
        """
        Initialise the model registry and populate the embedding table cache for existing models.
        Can extend it to include any backend-specific setup steps (e.g., creating extensions, staging directories) as needed.

        Parameters
        ----------
        engine : Engine
            SQLAlchemy engine for OMOP CDM database storage.
        """
        self.pre_initialise_store(engine)

        registered_models = self.embedding_model_registry.get_registered_models_from_db(
            backend_type=self.backend_type
        )
        if registered_models is None:
            return
    
        for model_record in registered_models:
            self._cache_model_record(engine=engine, model_record=model_record)

    def pre_initialise_store(self, engine: Engine) -> None:
        """
        Hook for setup steps that run before store initialization.

        Parameters
        ----------
        engine : Engine
            SQLAlchemy engine for OMOP CDM database storage.
        """
        pass

    def get_registered_model(
        self,
        model_name: str,
        index_type: IndexType,
    ) -> Optional[EmbeddingModelRecord]:
        """Return metadata for one registered model, if present."""
        registered_model = self.embedding_model_registry.get_registered_models_from_db(
            backend_type=self.backend_type,
            model_name=model_name,
            index_type=index_type
        )
        if registered_model is None:
            return None
        
        assert len(registered_model) == 1, (
            f"Expected exactly one registered model for name '{model_name}' and index type '{index_type.value}', but found {len(registered_model)}. This indicates a data integrity issue in the model registry database."
        )
        return registered_model[0]


    def is_model_registered(self, model_name: str, index_type: IndexType) -> bool:
        """Convenience wrapper over ``get_registered_model``."""
        return self.get_registered_model(model_name=model_name, index_type=index_type) is not None
    
    def get_concepts_without_embedding(
        self,
        session: Session,
        model_name: str,
        index_type: IndexType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: Optional[int] = None,
    ) -> Mapping[int, str]:
        """Return concept IDs and names for concepts that do not have embeddings."""
        query = self.q_get_concepts_without_embedding(
            model_name=model_name, 
            index_type=index_type,
            concept_filter=concept_filter,
            limit=limit
        )
        return {row.concept_id: row.concept_name for row in session.execute(query)}
    
    def q_get_concepts_without_embedding(
        self,
        model_name: str,
        index_type: IndexType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: Optional[int] = None,
    ) -> Select:
        embedding_table = self.get_embedding_table(model_name=model_name, index_type=index_type)
        subq = select(1).where(embedding_table.concept_id == Concept.concept_id)

        query = (
            select(Concept.concept_id, Concept.concept_name)
            .where(~subq.exists())
        )

        if concept_filter is not None:
            query = concept_filter.apply(query)

        return query.limit(limit)
    
    def get_concepts_without_embedding_count(
        self,
        session: Session,
        model_name: str,
        index_type: IndexType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> int:
        embedding_table = self.get_embedding_table(model_name=model_name, index_type=index_type)
        subq = select(1).where(embedding_table.concept_id == Concept.concept_id)

        query = (
            select(func.count())
            .select_from(Concept)
            .where(~subq.exists())
        )

        if concept_filter is not None:
            query = concept_filter.apply(query)

        return session.scalar(query)  # type: ignore


    def register_model(
        self,
        engine: Engine,
        model_name: str,
        dimensions: int,
        *,
        index_type: IndexType,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> EmbeddingModelRecord:
        """
        Shared template method for model registration.

        Parameters
        ----------
        engine : Engine
            SQLAlchemy engine for OMOP CDM database storage.
        model_name : str
            Registered name of the embedding model.
        dimensions : int
            Embedding dimensionality $D$ for this model.
        index_type : IndexType
            Storage index type used for this model's embeddings.
        metadata : Optional[Mapping[str, object]]
            Optional metadata persisted with the model registration.
        """
        model_record = self.embedding_model_registry.register_model(
            model_name=model_name,
            dimensions=dimensions,
            backend_type=self.backend_type,
            index_type=index_type,
            metadata=metadata or {},
        )
        # Local caching
        self._cache_model_record(engine=engine, model_record=model_record)
        return model_record

    @abstractmethod
    @require_registered_model
    def upsert_embeddings(
        self,
        model_name: str,
        index_type: IndexType,
        *,
        session: Session,
        concept_ids: Sequence[int],
        embeddings: ndarray,
        _model_record: EmbeddingModelRecord,
    ) -> None:
        """
        Insert or update vector embeddings for a collection of OMOP concept IDs.

        This method handles the persistence of generated embeddings. Depending on the 
        concrete backend implementation, this may involve staging data in a durable store for later indexing
        and (optionally) writing to a high-performance vector index (FAISS), or updateing a relational dataabase 
        table (like pgvector)

        Parameters
        ----------
        model_name : str
            Registered name of the embedding model.
        index_type : IndexType
            Storage index type used for this model's embeddings.
        session : sqlalchemy.orm.Session
            SQLAlchemy session bound to the OMOP CDM database.
        concept_ids : Sequence[int]
            Concept IDs aligned with the rows of ``embeddings``.
        embeddings : numpy.ndarray
            Embedding matrix of shape ``(n_concepts, D)``.
        _model_record : EmbeddingModelRecord
            Internal registered-model record injected by ``@require_registered_model``.

        Returns
        -------
        None

        Notes
        -----
        Implementations should ensure that the order of `concept_ids` strictly 
        matches the row order of the `embeddings` matrix to prevent data 
        misalignment.
        """

    @abstractmethod
    @require_registered_model
    def get_embeddings_by_concept_ids(
        self,
        model_name: str,
        index_type: IndexType,
        *,
        session: Session,
        concept_ids: Sequence[int],
        _model_record: EmbeddingModelRecord,
    ) -> Mapping[int, Sequence[float]]:
        """Fetch stored embedding vectors keyed by concept ID."""


    @abstractmethod
    @require_registered_model
    def get_nearest_concepts(
        self,
        model_name: str,
        index_type: IndexType,
        *,
        session: Session,
        query_embedding: ndarray,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        k: int = 10,
        _model_record: EmbeddingModelRecord,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        """
        Return nearest stored concepts for the query embedding.

        Parameters
        ----------
        model_name : str
            Registered name of the embedding model.
        index_type : IndexType
            Storage index type used for this model's embeddings.
        session : Session
            SQLAlchemy session bound to the OMOP CDM database.
        query_embedding : ndarray
            Query embedding matrix of shape ``(Q, D)``.
        metric_type : MetricType
            Similarity or distance metric for nearest-neighbor search.
        concept_filter : Optional[EmbeddingConceptFilter], optional
            Optional filter restricting which OMOP concepts are considered.
        k : int, optional
            Number of nearest matches to return per query.
        _model_record : EmbeddingModelRecord
            Internal registered-model record injected by ``@require_registered_model``.

        Returns
        -------
        Tuple[Tuple[NearestConceptMatch, ...], ...]
            A tuple of tuples containing nearest concept matches for each query vector. The outer tuple corresponds to the query vectors in order, and each inner tuple contains the nearest matches for that query vector, sorted by similarity. Returned shape is (q, k) where q is the number of query vectors and k is the number of nearest neighbors returned per query.
        """
    
    def has_any_embeddings(
        self, 
        session: Session, 
        model_name: str,
        index_type: IndexType,
    ) -> bool:
        embedding_table = self.get_embedding_table(
            model_name=model_name,
            index_type=index_type,
        )
        return session.query(embedding_table.concept_id).limit(1).first() is not None
    
    def get_embedding_table(
        self,
        model_name: str,
        index_type: IndexType,
    ) -> Type[T]:
        """Return dynamically created ORM class for the embedding table associated with the given model name.
        This relies on the `initialise_store` method having been called to populate the `embedding_table_cache`. 
        If the requested model name is not found in the cache, this indicates a logic error in the calling code (e.g., not initializing the store, or requesting a table for a model that hasn't been registered) rather than a user input error, so a ValueError is raised.
        
        Parameters
        ----------
        model_name : str
            The name of the embedding model whose associated embedding table class is to be retrieved.
        Returns
        -------
        Type[T]
            The dynamically generated SQLAlchemy ORM class corresponding to the embedding table for the specified model.
        
        Raises
        ------
        ValueError
            If the model name is not found in the embedding table cache, indicating that the store was not properly initialized or the model has not been registered.
        """
        storage_key = (model_name, self.backend_type, index_type)
        embedding_table = self.embedding_table_cache.get(storage_key)
        if embedding_table is not None:
            return embedding_table
        else:
            raise ValueError(f"Embedding table for model '{model_name}' with index type '{index_type.value}' and backend '{self.backend_type.value}' not found in cache. Ensure that the store is initialized and the model is registered before attempting to access the embedding table.")
    
    @staticmethod
    def validate_embeddings(embeddings: ndarray, dimensions: int):
        """Validate that the embeddings array has the correct shape and dimensionality.
        Currently enforcing:
        - 2D array (num_embeddings, embedding_dimension)
        - embedding_dimension matches the model configuration
        """

        assert embeddings.ndim == 2, f"Expected 2D array of embeddings. Got {embeddings.ndim}."
        assert embeddings.shape[1] == dimensions, (
            f"Embedding dimensionality ({embeddings.shape[1]}) does not match "
            f"model configuration ({dimensions})."
        )

    @staticmethod
    def validate_embeddings_and_concept_ids(
        embeddings: ndarray, 
        concept_ids: Union[Sequence[int], ndarray],
        dimensions: int
    ):
        EmbeddingBackend.validate_embeddings(embeddings, dimensions=dimensions)

        assert len(concept_ids) == embeddings.shape[0], (
            f"Number of concept IDs ({len(concept_ids)}) does not match number of embeddings ({embeddings.shape[0]})."
        )