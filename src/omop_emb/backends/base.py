from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Union, Type, TypeVar, Generic, Dict, Any, Callable, Tuple
import re
from functools import wraps

from numpy import ndarray
from sqlalchemy import Engine, select, Integer, ForeignKey, func, Select
from sqlalchemy.orm import Session, mapped_column

from omop_alchemy.cdm.model.vocabulary import Concept

from .registry import ModelRegistry, ensure_model_registry_schema
from .config import BackendType, SUPPORTED_INDICES_AND_METRICS_PER_BACKEND, IndexType, MetricType
from .errors import EmbeddingBackendConfigurationError


def require_registered_model(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, session: Session, model_name: str, *args, **kwargs) -> Any:
        record = self.get_registered_model(session, model_name)
        
        if record is None:
            raise ValueError(
                f"Embedding model '{model_name}' is not registered in the FAISS backend."
            )
        return func(self, session, model_name, record, *args, **kwargs)
        
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


@dataclass(frozen=True)
class EmbeddingModelRecord:
    """
    Canonical description of a registered embedding model.

    ``storage_identifier`` is intentionally backend-specific. For example:
    - PostgreSQL backend: dynamic embedding table name
    - FAISS backend: on-disk index path or logical collection name
    """

    model_name: str
    dimensions: int
    backend_type: BackendType
    index_type: IndexType
    storage_identifier: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingConceptFilter:
    """
    Constraints applied during embedding retrieval.

    This mirrors the current OMOP grounding needs without importing
    ``omop_graph`` or its search-constraint objects into ``omop_emb``.
    """

    concept_ids: Optional[tuple[int, ...]] = None
    domains: Optional[tuple[str, ...]] = None
    vocabularies: Optional[tuple[str, ...]] = None
    require_standard: bool = False

    def apply(self, query: Select) -> Select:
        if self.concept_ids is not None:
            query = query.where(Concept.concept_id.in_(self.concept_ids))

        if self.domains is not None:
            query = query.where(Concept.domain_id.in_(self.domains))

        if self.vocabularies is not None:
            query = query.where(Concept.vocabulary_id.in_(self.vocabularies))

        if self.require_standard:
            query = query.where(Concept.standard_concept.in_(["S", "C"]))

        return query


@dataclass(frozen=True)
class NearestConceptMatch:
    """
    Backend-neutral nearest-neighbor payload returned to callers.

    The current resolver layer in ``omop-graph`` needs these fields to build
    ``LabelMatch`` objects and to explain whether a retrieved concept is
    standard and active.
    """

    concept_id: int
    concept_name: str
    similarity: float
    is_standard: bool
    is_active: bool


@dataclass(frozen=True)
class EmbeddingBackendCapabilities:
    """
    Capability flags for a backend implementation.

    These are not used by the current code yet, but they make backend
    differences explicit. For example, a FAISS backend might support nearest
    neighbor search but require explicit refreshes after bulk writes.
    """

    stores_embeddings: bool = True
    supports_incremental_upsert: bool = True
    supports_nearest_neighbor_search: bool = True
    supports_server_side_similarity: bool = True
    supports_filter_by_concept_ids: bool = True
    supports_filter_by_domain: bool = True
    supports_filter_by_vocabulary: bool = True
    supports_filter_by_standard_flag: bool = True
    requires_explicit_index_refresh: bool = False


class EmbeddingBackend(ABC, Generic[T]):
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
    def __init__(self):
        super().__init__()
        self._model_cache: dict[str, Type[T]] = {}

    @property
    def model_cache(self) -> Dict[str, Type[T]]:
        """In-memory cache of dynamically generated ORM classes for embedding tables."""
        return self._model_cache

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """General category of backend, used for index compatibility checks."""

    @property
    def backend_name(self) -> str:
        """Stable identifier for this backend implementation."""
        return self.backend_type.value

    @property
    @abstractmethod
    def capabilities(self) -> EmbeddingBackendCapabilities:
        """Capability flags describing what this backend can do."""

    @abstractmethod
    def initialise_store(self, engine: Engine) -> None:
        """
        Prepare any required storage structures.

        Examples:
        - create registry tables
        - warm caches from a registry
        - create directories or sidecar files
        """

    def list_registered_models(self, session: Session) -> Sequence[EmbeddingModelRecord]:
        """Return all embedding models known to this backend."""
        return tuple(
            self._record_from_registry_row(row)
            for row in session.scalars(
                select(ModelRegistry).where(ModelRegistry.backend_type == self.backend_type)
            ).all()
        )

    def get_registered_model(
        self,
        session: Session,
        model_name: str,
    ) -> Optional[EmbeddingModelRecord]:
        """Return metadata for one registered model, if present."""
        row = session.scalar(
            select(ModelRegistry).where(
                ModelRegistry.model_name == model_name,
                ModelRegistry.backend_type == self.backend_type,
            )
        )
        if row is None:
            return None
        return self._record_from_registry_row(row)

    def is_model_registered(self, session: Session, model_name: str) -> bool:
        """Convenience wrapper over ``get_registered_model``."""
        return self.get_registered_model(session=session, model_name=model_name) is not None
    
    def get_concepts_without_embedding(
        self,
        session: Session,
        model_name: str,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: Optional[int] = None,
    ) -> Mapping[int, str]:
        """Return concept IDs and names for concepts that do not have embeddings."""
        query = self.get_concepts_without_embedding_query(
            session=session, 
            model_name=model_name, 
            concept_filter=concept_filter,
            limit=limit
        )
        return {row.concept_id: row.concept_name for row in session.execute(query)}
    
    def get_concepts_without_embedding_query(
        self,
        session: Session,
        model_name: str,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: Optional[int] = None,
    ) -> Select:
        embedding_table = self._get_embedding_table(session=session, model_name=model_name)
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
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> int:
        embedding_table = self._get_embedding_table(session=session, model_name=model_name)
        subq = select(1).where(embedding_table.concept_id == Concept.concept_id)

        query = (
            select(func.count())
            .select_from(Concept)
            .where(~subq.exists())
        )

        if concept_filter is not None:
            query = concept_filter.apply(query)

        return session.scalar(query)  # type: ignore

    @abstractmethod
    def _create_storage_table(self, engine: Engine, entry: ModelRegistry) -> Type[T]:
        """Backend-specific logic to create the dynamic SQLAlchemy class."""

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
        self.initialise_store(engine)
        with Session(engine, expire_on_commit=False) as session:
            existing_row = session.scalar(
                select(ModelRegistry).where(ModelRegistry.model_name == model_name)
            )
            if existing_row is not None:
                if existing_row.backend_type != self.backend_type:
                    raise EmbeddingBackendConfigurationError(
                        f"Model '{model_name}' is already registered for backend "
                        f"'{existing_row.backend_type}', not '{self.backend_type}'."
                    )
                if existing_row.dimensions != dimensions:
                    raise EmbeddingBackendConfigurationError(
                        f"Model '{model_name}' is already registered with dimensions "
                        f"{existing_row.dimensions}, not {dimensions}."
                    )
                if existing_row.index_type != index_type:
                    raise EmbeddingBackendConfigurationError(
                        f"Model '{model_name}' is already registered with "
                        f"index_method='{existing_row.index_type}', not "
                        f"'{index_type}'. Reuse the existing model "
                        "configuration or register a new model name."
                    )
                # TODO: Is that necessary? I mean we tried to register and it already exists.
                #existing_row.metadata = {**self._coerce_registry_metadata(existing_row.metadata), **metadata}
                #session.commit()
                #return self._record_from_registry_row(existing_row)
        
        ensure_model_registry_schema(engine)
        safe_name = self.safe_model_name(model_name)
        
        new_entry = ModelRegistry(
            model_name=model_name,
            dimensions=dimensions,
            storage_identifier=safe_name,
            index_type=index_type,
            backend_type=self.backend_type,
            metadata=metadata
        )

        with Session(engine, expire_on_commit=False) as session:
            # Add logic here to check for existing records before adding
            session.add(new_entry)
            session.commit()
        
        dynamic_class = self._create_storage_table(engine, new_entry)
        storage_identifier = dynamic_class.__tablename__
        # TODO: Maybe using self._record_from_registry_row
        return EmbeddingModelRecord(
            model_name=model_name,
            dimensions=dimensions,
            backend_type=self.backend_type,
            storage_identifier=storage_identifier,
            index_type=index_type,
            metadata=metadata,
        )

    @abstractmethod
    @require_registered_model
    def upsert_embeddings(
        self,
        session: Session,
        model_name: str,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
        embeddings: ndarray,
    ) -> None:
        """
        Insert or update embeddings for the given concept IDs.

        Implementations may write directly to the serving index, write to a
        durable store and refresh later, or do both.
        """

    @abstractmethod
    def get_embeddings_by_concept_ids(
        self,
        session: Session,
        model_name: str,
        concept_ids: Sequence[int],
    ) -> Mapping[int, Sequence[float]]:
        """Fetch stored embedding vectors keyed by concept ID."""


    @abstractmethod
    def get_nearest_concepts(
        self,
        session: Session,
        model_name: str,
        query_embedding: ndarray,
        metric_type: MetricType,
        *,
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
        query_embedding : ndarray
            The embedding vector to search with. Expected shape is (q, dimension)
            where q is the number of query vectors and dimension is the size of the embedding space for the model.
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

    def refresh_model_index(self, session: Session, model_name: str) -> None:
        """
        Optional hook for derived index maintenance.

        PostgreSQL/pgvector backends may not need this. File-based or FAISS
        backends may choose to rebuild or compact a search index here.
        """
        return None
    
    def has_any_embeddings(self, session: Session, model_name: str) -> bool:
        embedding_table = self._get_embedding_table(
            session=session,
            model_name=model_name,
        )
        return session.query(embedding_table.concept_id).limit(1).first() is not None
    
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

    @staticmethod
    def _coerce_registry_metadata(value: object) -> Mapping[str, object]:
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    @staticmethod
    def _record_from_registry_row(
        row: ModelRegistry,
    ) -> EmbeddingModelRecord:
        return EmbeddingModelRecord(
            model_name=row.model_name,
            dimensions=row.dimensions,
            backend_type=row.backend_type,
            storage_identifier=row.storage_identifier,
            index_type=row.index_type,
            metadata=EmbeddingBackend._coerce_registry_metadata(row.metadata),
        )
    @staticmethod
    def safe_model_name(model_name: str) -> str:
        name = model_name.lower()
        sanitized = re.sub(r"[^\w]+", "_", name)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        return sanitized
    
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