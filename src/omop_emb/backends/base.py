from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from numpy import ndarray
from sqlalchemy import Engine
from sqlalchemy.orm import Session


@dataclass(frozen=True)
class EmbeddingIndexConfig:
    """
    Backend-neutral index configuration for an embedding model.

    Backends may interpret these fields differently:
    - pgvector may use ``index_type`` values such as ``hnsw`` or ``ivfflat``
    - FAISS may use values such as ``IndexFlatIP`` or ``IndexIVFFlat``
    - non-ANN backends may ignore this entirely
    """

    index_type: Optional[str] = None
    distance_metric: str = "cosine"
    parameters: Mapping[str, object] = field(default_factory=dict)


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
    backend_name: str
    storage_identifier: Optional[str] = None
    index_config: Optional[EmbeddingIndexConfig] = None
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


class EmbeddingBackend(ABC):
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

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Stable identifier for this backend implementation."""

    @property
    def capabilities(self) -> EmbeddingBackendCapabilities:
        """Capability flags describing what this backend can do."""
        return EmbeddingBackendCapabilities()

    @abstractmethod
    def initialise_store(self, engine: Engine) -> None:
        """
        Prepare any required storage structures.

        Examples:
        - create registry tables
        - warm caches from a registry
        - create directories or sidecar files
        """

    @abstractmethod
    def list_registered_models(self, session: Session) -> Sequence[EmbeddingModelRecord]:
        """Return all embedding models known to this backend."""

    @abstractmethod
    def get_registered_model(
        self,
        session: Session,
        model_name: str,
    ) -> Optional[EmbeddingModelRecord]:
        """Return metadata for one registered model, if present."""

    def is_model_registered(self, session: Session, model_name: str) -> bool:
        """Convenience wrapper over ``get_registered_model``."""
        return self.get_registered_model(session=session, model_name=model_name) is not None

    @abstractmethod
    def register_model(
        self,
        engine: Engine,
        model_name: str,
        dimensions: int,
        *,
        index_config: Optional[EmbeddingIndexConfig] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> EmbeddingModelRecord:
        """
        Register a new embedding model and provision backend storage for it.
        """

    @abstractmethod
    def has_any_embeddings(self, session: Session, model_name: str) -> bool:
        """Return True if any concept embeddings are stored for the model."""

    @abstractmethod
    def upsert_embeddings(
        self,
        session: Session,
        model_name: str,
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
    def get_similarities(
        self,
        session: Session,
        model_name: str,
        query_embedding: Sequence[float],
        *,
        concept_ids: Optional[Sequence[int]] = None,
    ) -> Mapping[int, float]:
        """
        Return similarity scores between a query embedding and stored vectors.

        This supports scorer-stage use cases where the candidate concept set is
        already known and only the semantic score must be computed.
        """

    @abstractmethod
    def get_nearest_concepts(
        self,
        session: Session,
        model_name: str,
        query_embedding: Sequence[float],
        *,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: int = 10,
    ) -> Sequence[NearestConceptMatch]:
        """
        Return nearest stored concepts for the query embedding.

        Backends that cannot apply every filter inside the search engine should
        still honor this contract by composing with OMOP SQL metadata and
        post-filtering as needed.
        """

    def refresh_model_index(self, session: Session, model_name: str) -> None:
        """
        Optional hook for derived index maintenance.

        PostgreSQL/pgvector backends may not need this. File-based or FAISS
        backends may choose to rebuild or compact a search index here.
        """

