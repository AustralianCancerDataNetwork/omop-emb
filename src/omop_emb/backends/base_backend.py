from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Optional, Sequence, Union, Type, TypeVar, Generic, Dict, Any, Callable, Tuple
from functools import wraps
from pathlib import Path

from numpy import ndarray, integer
from sqlalchemy import Engine, select, Integer, ForeignKey, and_, Result

from sqlalchemy.orm import Session, mapped_column

from omop_alchemy.cdm.model.vocabulary import Concept

from ..model_registry import ModelRegistry, ModelRegistryManager, EmbeddingModelRecord
from ..config import BackendType, IndexType, MetricType
from ..utils.errors import ModelRegistrationConflictError
from ..utils.embedding_utils import (
    EmbeddingConceptFilter, 
    NearestConceptMatch, 
)

def require_registered_model(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(
        self, 
        session: Session, 
        model_name: str,
        index_type: IndexType,
        *args, 
        **kwargs) -> Any:
        record = self.get_registered_model(session, model_name, index_type)
        
        if record is None:
            raise ValueError(
                f"Embedding model '{model_name}' with backend type {self.backend_type} and index type '{index_type}' is not registered"
            )
        return func(
            self, 
            session=session, 
            model_name=model_name, 
            index_type=index_type, 
            model_record=record, 
            *args, 
            **kwargs
        )
        
    return wrapper

class EmbeddingBackendBase(ABC):
    """
    Abstract interface for swappable embedding storage and retrieval backends.

    Design goals
    ------------
    - Keep embedding generation separate from embedding persistence.
    - Support multiple storage/retrieval implementations behind one contract.
    - Preserve the current operational needs of ``omop-graph``:
      - model registration
      - embedding population
      - embedding lookup by concept ID
      - similarity lookup over a set of concept IDs
      - nearest-neighbor retrieval with OMOP-oriented filters

    Notes
    -----
    This interface intentionally still accepts SQLAlchemy ``Engine`` and
    ``Session`` objects. Even with a non-Postgres embedding index such as
    FAISS, the implementation will usually still need OMOP relational access
    to validate models, resolve concept metadata, and apply domain/vocabulary
    filters.
    """
    def __init__(self, registry_base_dir: str | Path = ".omop_emb"):
        super().__init__()
        self._model_registry = ModelRegistryManager(base_dir=str(registry_base_dir))

    @property
    def model_registry(self) -> ModelRegistryManager:
        return self._model_registry

    @property
    def requires_database(self) -> bool:
        """Whether this backend requires an OMOP database for core operation."""
        return False

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """General category of backend, used for index compatibility checks."""

    @property
    def backend_name(self) -> str:
        """Stable identifier for this backend implementation."""
        return self.backend_type.value

    def initialise_store(self, engine: Engine) -> None:
        """
        Prepare any required storage structures.
        """
        return


    def get_registered_model(
        self,
        model_name: str,
        index_type: IndexType,
    ) -> Optional[EmbeddingModelRecord]:
        """Return metadata for one registered model, if present."""
        registered_model = self.model_registry.get_registered_models(
            backend_type=self.backend_type,
            model_name=model_name,
            index_type=index_type,
        )
        if registered_model is None:
            return None
        
        assert len(registered_model) == 1, f"Expected exactly one registered model for name '{model_name}' and index type '{index_type}', but found {len(registered_model)}."
        return registered_model[0]

    def is_model_registered(
        self, 
        model_name: str, 
        index_type: IndexType
    ) -> bool:
        """Convenience wrapper over ``get_registered_model``."""
        return self.get_registered_model(
            model_name=model_name, 
            index_type=index_type
        ) is not None

    def register_model(
        self,
        engine: Engine,
        model_name: str,
        dimensions: int,
        *,
        index_type: IndexType,
        metadata: Mapping[str, object] = {},
    ) -> EmbeddingModelRecord:
        """
        Shared template method for model registration.
        """
        record = self.model_registry.register_model(
            model_name=model_name,
            dimensions=dimensions,
            backend_type=self.backend_type,
            index_type=index_type,
            metadata=metadata
        )
        return record

    @abstractmethod
    @require_registered_model
    def upsert_embeddings(
        self,
        session: Session,
        model_name: str,
        index_type: IndexType,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
        embeddings: ndarray,
    ) -> None:
        """
        Insert or update vector embeddings for a collection of OMOP concept IDs.

        This method handles the persistence of generated embeddings. Depending on the 
        concrete backend implementation, this may involve staging data in a durable store for later indexing
        and (optionally) writing to a high-performance vector index (FAISS), or updateing a relational dataabase 
        table (like pgvector)

        Parameters
        ----------
        session : sqlalchemy.orm.Session
            The active database session used for transactional persistence and 
            model metadata updates.
        model_name : str
            The unique identifier or name of the embedding model (e.g., 
            'text-embedding-3-small').
        index_type : IndexType
            The type of index used for this model, which determines the similarity metric and may affect how embeddings are stored and queried.
        model_record : EmbeddingModelRecord
            A record object containing metadata, dimensions, and configuration 
            specific to the embedding model being processed. This is provided by the `require_registered_model` decorator to ensure that the model is registered and its metadata is available for use in the embedding persistence logic.
        concept_ids : Sequence[int]
            A sequence of OMOP standard concept IDs corresponding to the 
            ordered rows in the embeddings array.
        embeddings : numpy.ndarray
            A 2D array of shape (n_concepts, n_dimensions) containing the 
            generated vector representations.

        Returns
        -------
        None

        Notes
        -----
        Implementations should ensure that the order of `concept_ids` strictly 
        matches the row order of the `embeddings` matrix to prevent data 
        misalignment.
        """
        ...

    @abstractmethod
    @require_registered_model
    def get_embeddings_by_concept_ids(
        self,
        session: Session,
        model_name: str,
        index_type: IndexType,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
    ) -> Mapping[int, Sequence[float]]:
        """Fetch stored embedding vectors keyed by concept ID."""
        ...


    @abstractmethod
    @require_registered_model
    def get_nearest_concepts(
        self,
        session: Session,
        model_name: str,
        index_type: IndexType,
        model_record: EmbeddingModelRecord,
        *,
        query_embeddings: ndarray,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        k: int = 10,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        """
        Return nearest stored concepts for the query embedding.

        Parameters
        ----------
        session : Session
            SQLAlchemy session for any required relational access.
        model_name : str
            Name of the embedding model used to create the embeddings.
        index_type : IndexType
            The type of index used for this model, which determines the similarity metric and may affect how embeddings are stored and queried.
        model_record : EmbeddingModelRecord
            Metadata record for the embedding model, which may contain additional configuration or parameters needed for querying. Obtained from `required_registered_model` decorator.
        query_embeddings : ndarray
            The embedding vector to search with. Expected shape is (q, dimension), where q is the number of query vectors and dimension is the size of the embedding space for the model.
        metric_type : MetricType
            The similarity or distance metric to use for nearest neighbor search. This should be compatible with the index type used by the model.
        concept_filter : Optional[EmbeddingConceptFilter], optional
            Optional constraints to apply when retrieving nearest concepts. This allows filtering by concept IDs, domains, vocabularies, or standard flags. By default, no additional filtering is applied.
        k : int, optional
            K nearest neighbors to return for each query vector. Default is 10.

        Returns
        -------
        Tuple[Tuple[NearestConceptMatch, ...], ...]
            A tuple of tuples containing nearest concept matches for each query vector. The outer tuple corresponds to the query vectors in order, and each inner tuple contains the nearest matches for that query vector, sorted by similarity. Returned shape is (q, k) where q is the number of query vectors and k is the number of nearest neighbors returned per query.
        """
        ...
    
    @abstractmethod
    @require_registered_model
    def has_any_embeddings(
        self, 
        session: Session, 
        model_name: str,
        index_type: IndexType,
        model_record: EmbeddingModelRecord,
    ) -> bool:
        """Return True if any embeddings are stored for the given model, False otherwise."""
        ...

    
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
    def validate_concept_ids(concept_ids: Sequence[int] | ndarray):
        """Validate that the concept IDs are in the expected format."""
        if isinstance(concept_ids, ndarray):
            assert concept_ids.ndim == 1, f"Expected 1D array of concept IDs. Got {concept_ids.ndim}."
            assert issubclass(concept_ids.dtype.type, integer), f"Expected integer type for concept IDs. Got {concept_ids.dtype}."
        else:
            assert all(isinstance(cid, int) for cid in concept_ids), "All concept IDs must be integers."

    @staticmethod
    def validate_embeddings_and_concept_ids(
        embeddings: ndarray, 
        concept_ids: Union[Sequence[int], ndarray],
        dimensions: int
    ):
        EmbeddingBackendBase.validate_embeddings(embeddings, dimensions=dimensions)
        EmbeddingBackendBase.validate_concept_ids(concept_ids)

        assert len(concept_ids) == embeddings.shape[0], (
            f"Number of concept IDs ({len(concept_ids)}) does not match number of embeddings ({embeddings.shape[0]})."
        )