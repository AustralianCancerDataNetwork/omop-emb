from __future__ import annotations

import os
from typing import Literal, Mapping, Optional, Sequence, Tuple, List

import numpy as np
from numpy import ndarray
from sqlalchemy import Engine, Select
from sqlalchemy.orm import Session

from omop_emb.embeddings import EmbeddingClient, get_provider_from_provider_type

from .backends import get_embedding_backend
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter
from omop_emb.model_registry import EmbeddingModelRecord
from .config import (
    BackendType,
    IndexType,
    MetricType,
    ProviderType,
    ENV_DOCUMENT_EMBEDDING_PREFIX,
    ENV_QUERY_EMBEDDING_PREFIX,
)



def list_registered_models(
    backend_name_or_type: Optional[str | BackendType] = None,
    provider_type: Optional[ProviderType] = None,
    model_name: Optional[str] = None,
    index_type: Optional[IndexType] = None,
    storage_base_dir: Optional[str] = None,
    registry_db_name: Optional[str] = None,
) -> tuple[EmbeddingModelRecord, ...]:
    """List registered models from the backend registry.

    Standalone function for administrative/discovery use (e.g. CLI export).
    Not scoped to a single model — queries across the registry.

    Parameters
    ----------
    backend_name_or_type : str | BackendType, optional
        Backend to query. Falls back to ``OMOP_EMB_BACKEND`` env var.
    provider_type : ProviderType, optional
        Filter by provider type.
    model_name : str, optional
        Filter by model name.
    index_type : IndexType, optional
        Filter by index type.
    storage_base_dir : str, optional
        Base directory for backend storage.
    registry_db_name : str, optional
        Custom model registry database filename.

    Returns
    -------
    tuple[EmbeddingModelRecord, ...]
        Matching registered models (empty tuple if none found).
    """
    backend = get_embedding_backend(
        backend_name_or_type=backend_name_or_type,
        storage_base_dir=storage_base_dir,
        registry_db_name=registry_db_name,
    )
    return backend.get_registered_models(
        model_name=model_name,
        index_type=index_type,
        provider_type=provider_type,
    )

def migrate_legacy_registry_row(
    backend_type: BackendType,
    provider_type: ProviderType,
    model_name: str,
    dimensions: int,
    index_type: IndexType,
    metadata: dict,
    storage_identifier: str,
    storage_base_dir: Optional[str] = None,
) -> EmbeddingModelRecord:
    """Migrate a single legacy registry row into the local metadata.db.

    Used by CLI migration commands to populate the registry with explicit
    ``provider_type`` from legacy data or command-line arguments.

    This bypasses model-name validation (legacy names may not satisfy
    current provider rules, e.g. untagged Ollama names).

    Parameters
    ----------
    backend_type : BackendType
        Target backend type.
    provider_type : ProviderType
        Provider type to assign to the migrated row.
    model_name : str
        Model name from the legacy registry (used verbatim).
    dimensions : int
        Embedding dimensionality.
    index_type : IndexType
        Vector index type.
    metadata : dict
        Arbitrary metadata from the legacy row.
    storage_identifier : str
        Backend-specific storage identifier (e.g. table name).
    storage_base_dir : str, optional
        Base directory for the local metadata registry.

    Returns
    -------
    EmbeddingModelRecord
        The newly created registry entry.
    """
    backend = get_embedding_backend(
        backend_name_or_type=backend_type,
        storage_base_dir=storage_base_dir,
    )
    return backend._embedding_model_registry.register_model(
        model_name=model_name,
        provider_type=provider_type,
        dimensions=dimensions,
        backend_type=backend_type,
        index_type=index_type,
        metadata=metadata or {},
        storage_identifier=storage_identifier,
    )


class EmbeddingReaderInterface:
    """
    Backend-neutral reader interface for embedding query operations.

    Responsibilities
    ----------------
    - retrieve stored concept embeddings
    - search nearest neighbors
    - list registered models
    - initialize the backend store (read-only usage)

    This class is designed for read-only operations and does not require an
    ``EmbeddingClient``. All registry queries use the ``provider_type`` held
    at construction time.

    Notes
    -----
    We are requiring a canonical model name at construction time to ensure that the reader is always
    pointing to a valid model.
    For definition of "canonical model name", see the provider's implementation of
    :class:`~omop_emb.embeddings.EmbeddingProvider.canonical_model_name`.
    """

    def __init__(
        self,
        canonical_model_name: str,
        provider_name_or_type: str | ProviderType,
        backend_name_or_type: Optional[str | BackendType] = None,
        storage_base_dir: Optional[str] = None,
        registry_db_name: Optional[str] = None,
    ):
        """Initialize embedding reader.

        Parameters
        ----------
        canonical_model_name : str
            Canonical name of the embedding model.
        provider_name_or_type : str | ProviderType
            Provider type used for model lookups (OLLAMA, OPENAI, etc.)
        backend_name_or_type : str | BackendType, optional
            Embedding backend name or type (pgvector, faiss, etc.).
            Falls back to ``OMOP_EMB_BACKEND`` environment variable.
        storage_base_dir : Optional[str]
            Base directory for backend storage
        registry_db_name : Optional[str]
            Custom model registry database filename
        """
        if isinstance(provider_name_or_type, str):
            provider_type = ProviderType(provider_name_or_type)
        elif isinstance(provider_name_or_type, ProviderType):
            provider_type = provider_name_or_type
        else:
            raise ValueError(
                f"Invalid provider_name_or_type: expected str or ProviderType, got {type(provider_name_or_type).__name__}."
             )
        
        provider = get_provider_from_provider_type(provider_type)
        if canonical_model_name != provider.canonical_model_name(canonical_model_name):
            raise ValueError(
                f"Canonical model name validation failed for provider {provider_type}: "
                f"'{canonical_model_name}' is not valid. Expected form: '{provider.canonical_model_name(canonical_model_name)}'."
            )

        self._backend = get_embedding_backend(
            backend_name_or_type=backend_name_or_type,
            storage_base_dir=storage_base_dir,
            registry_db_name=registry_db_name,
        )
        self._backend_type = self._backend.backend_type
        self._provider_type = provider_type
        self._canonical_model_name = canonical_model_name

    @property
    def backend_type(self) -> BackendType:
        """Backend type for this reader."""
        return self._backend_type
    
    @property
    def canonical_model_name(self) -> str:
        """Canonical model name for this reader."""
        return self._canonical_model_name
    
    @property
    def provider_type(self) -> ProviderType:
        """Provider type for this reader."""
        return self._provider_type

    def initialise_store(self, engine: Engine) -> None:
        """Initialize the backend store."""
        self._backend.initialise_store(engine)

    def get_model_table_name(
        self,
        index_type: IndexType,
    ) -> Optional[str]:
        """Get the storage identifier (table name) for a registered model."""
        record = self._backend.get_registered_model(
            model_name=self.canonical_model_name,
            index_type=index_type,
            provider_type=self.provider_type,
        )
        return record.storage_identifier if record is not None else None

    def is_model_registered(
        self,
        index_type: IndexType,
    ) -> bool:
        """Check if a model is registered in the backend."""
        return self._backend.is_model_registered(
            model_name=self.canonical_model_name,
            index_type=index_type,
            provider_type=self.provider_type,
        )

    def has_any_embeddings(
        self,
        session: Session,
        index_type: IndexType,
    ) -> bool:
        """Check if any embeddings exist for a model."""
        return self._backend.has_any_embeddings(
            session=session,
            model_name=self.canonical_model_name,
            provider_type=self.provider_type,
            index_type=index_type,
        )

    def get_nearest_concepts(
        self,
        session: Session,
        index_type: IndexType,
        query_embedding: np.ndarray,
        *,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> Tuple[Mapping[int, float], ...]:
        """Return nearest stored concepts for the query embedding.

        Parameters
        ----------
        session : Session
            SQLAlchemy session for any required relational access.
        index_type : IndexType
            The type of vector index used to store the embeddings.
        query_embedding : ndarray
            The embedding vector to search with. Expected shape is (q, dimension)
            where q is the number of query vectors.
        metric_type : MetricType
            The similarity or distance metric to use for nearest neighbor search.
        concept_filter : Optional[EmbeddingConceptFilter], optional
            A filter to specify which concepts to consider as potential nearest neighbors.

        Returns
        -------
        Tuple[Mapping[int, float], ...]
            A tuple of dictionaries containing nearest concept matches for each query vector.
        """
        if not isinstance(metric_type, MetricType):
            raise TypeError(
                f"metric_type must be MetricType, got {type(metric_type).__name__}."
            )
        nearest_concepts = self._backend.get_nearest_concepts(
            session=session,
            model_name=self.canonical_model_name,
            provider_type=self.provider_type,
            index_type=index_type,
            query_embeddings=query_embedding,
            concept_filter=concept_filter,
            metric_type=metric_type,
        )
        return tuple(
            {match.concept_id: match.similarity for match in matches_per_query}
            for matches_per_query in nearest_concepts
        )

    def get_embeddings_by_concept_ids(
        self,
        session: Session,
        index_type: IndexType,
        concept_ids: Tuple[int, ...],
    ) -> Mapping[int, Sequence[float]]:
        """Get embeddings for specific concept IDs."""
        return self._backend.get_embeddings_by_concept_ids(
            session=session,
            model_name=self.canonical_model_name,
            index_type=index_type,
            concept_ids=concept_ids,
            provider_type=self.provider_type,
        )

    def get_concepts_without_embedding(
        self,
        *,
        session: Session,
        index_type: IndexType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: Optional[int] = None,
    ) -> Mapping[int, str]:
        """Get concept IDs and names for concepts without embeddings."""
        return self._backend.get_concepts_without_embedding(
            session=session,
            model_name=self.canonical_model_name,
            provider_type=self.provider_type,
            index_type=index_type,
            concept_filter=concept_filter,
            limit=limit,
        )

    def q_get_concepts_without_embedding(
        self,
        *,
        index_type: IndexType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: Optional[int] = None,
    ) -> Select:
        """Query for concepts without embeddings."""
        return self._backend.q_get_concepts_without_embedding(
            model_name=self.canonical_model_name,
            provider_type=self.provider_type,
            index_type=index_type,
            concept_filter=concept_filter,
            limit=limit,
        )

    def get_concepts_without_embedding_count(
        self,
        *,
        session: Session,
        index_type: IndexType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> int:
        """Count concepts without embeddings."""
        return self._backend.get_concepts_without_embedding_count(
            session=session,
            model_name=self.canonical_model_name,
            provider_type=self.provider_type,
            index_type=index_type,
            concept_filter=concept_filter,
        )

class EmbeddingWriterInterface(EmbeddingReaderInterface):
    """
    Backend-neutral interface for embedding write and query operations.

    Extends ``EmbeddingReaderInterface`` with write capabilities. Requires an
    ``EmbeddingClient`` to validate and generate embeddings.

    Responsibilities
    ----------------
    - register embedding models
    - generate embeddings with an ``EmbeddingClient``
    - upsert concept embeddings through the backend
    - provide a reusable in-process cache for query-text embeddings
    - all reader responsibilities (retrieve, search, list)

    Notes
    -----
    All methods that accept a model identifier expect a *canonical* model name.
    Model name validation is automatic via the ``embedding_client``'s provider.
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        backend_name_or_type: Optional[str | BackendType] = None,
        storage_base_dir: Optional[str] = None,
        registry_db_name: Optional[str] = None,
    ):
        """Initialize embedding interface.

        Parameters
        ----------
        embedding_client : EmbeddingClient
            Required. Used to derive the provider for model name validation
            and to generate embeddings for upsert and query operations.
        backend_name_or_type : str | BackendType, optional
            Embedding backend name or type (pgvector, faiss, etc.).
            Falls back to ``OMOP_EMB_BACKEND`` environment variable.
        storage_base_dir : Optional[str]
            Base directory for backend storage
        registry_db_name : Optional[str]
            Custom model registry database filename
        """
        self._embedding_client = embedding_client
        super().__init__(
            canonical_model_name=embedding_client.canonical_model_name,
            backend_name_or_type=backend_name_or_type,
            provider_name_or_type=embedding_client.provider.provider_type,
            storage_base_dir=storage_base_dir,
            registry_db_name=registry_db_name,
        )

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension from the client."""
        return self._embedding_client.embedding_dim

    def setup_and_register_model(
        self,
        engine: Engine,
        index_type: IndexType,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Register the embedding model and initialize the store.

        Register FIRST, then initialize — ensures atomic state.

        Parameters
        ----------
        engine : Engine
            SQLAlchemy engine for the target database.
        index_type : IndexType
            Backend-specific index type to register.
        metadata : Optional[Mapping[str, object]]
            Arbitrary metadata to attach to the registry entry.
        """
        self.register_model(
            engine=engine,
            index_type=index_type,
            metadata=metadata,
        )
        self.initialise_store(engine)

    def register_model(
        self,
        engine: Engine,
        index_type: IndexType,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> EmbeddingModelRecord:
        """Register an embedding model in the backend registry.

        Parameters
        ----------
        engine : Engine
            SQLAlchemy engine for the target database.
        index_type : IndexType
            Backend-specific index type.
        metadata : Optional[Mapping[str, object]]
            Arbitrary metadata to attach to the registry entry.

        Returns
        -------
        EmbeddingModelRecord
            The newly created or existing registry entry.
        """
        return self._backend.register_model(
            engine=engine,
            model_name=self.canonical_model_name,
            provider_type=self._embedding_client.provider.provider_type,
            dimensions=self.embedding_dim,
            index_type=index_type,
            metadata=metadata or {},
        )

    def add_to_db(
        self,
        session: Session,
        index_type: IndexType,
        concept_ids: Tuple[int, ...],
        embeddings: ndarray,
    ) -> None:
        """Add embeddings to the database.

        Parameters
        ----------
        session : Session
            SQLAlchemy session bound to the OMOP CDM database.
        index_type : IndexType
            Backend-specific index type.
        concept_ids : Tuple[int, ...]
            Concept IDs aligned with the rows of embeddings.
        embeddings : ndarray
            Embedding matrix of shape (n_concepts, D).
        """
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D embedding array, got ndim={embeddings.ndim}."
            )
        if len(concept_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Number of concept IDs ({len(concept_ids)}) does not match "
                f"number of embeddings ({embeddings.shape[0]})."
            )
        self._backend.upsert_embeddings(
            session=session,
            model_name=self.canonical_model_name,
            provider_type=self._embedding_client.provider.provider_type,
            index_type=index_type,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )

    def embed_texts(
        self,
        texts: str | Tuple[str, ...] | List[str],
        *,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Generate embeddings for texts.

        Parameters
        ----------
        texts : str | Tuple[str, ...] | List[str]
            Text(s) to embed.
        batch_size : Optional[int]
            Batch size for embedding generation.

        Returns
        -------
        np.ndarray
            Embedding matrix of shape (n_texts, dimensions).
        """
        return self._embedding_client.embeddings(texts, batch_size=batch_size)

    def upsert_concept_embeddings(
        self,
        *,
        session: Session,
        index_type: IndexType,
        concept_ids: Sequence[int],
        embeddings: ndarray,
    ) -> None:
        """Upsert concept embeddings to the backend.

        Parameters
        ----------
        session : Session
            SQLAlchemy session.
        index_type : IndexType
            Backend-specific index type.
        concept_ids : Sequence[int]
            Concept IDs aligned with embeddings.
        embeddings : ndarray
            Embedding matrix of shape (n_concepts, D).
        """
        self._backend.upsert_embeddings(
            session=session,
            model_name=self.canonical_model_name,
            provider_type=self._embedding_client.provider.provider_type,
            index_type=index_type,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )

    def embed_and_upsert_concepts(
        self,
        *,
        session: Session,
        index_type: IndexType,
        concept_ids: Sequence[int],
        concept_texts: Sequence[str],
        batch_size: Optional[int] = None,
    ) -> ndarray:
        """Generate embeddings for concepts and upsert them to the backend.

        Parameters
        ----------
        session : Session
            SQLAlchemy session.
        index_type : IndexType
            Backend-specific index type.
        concept_ids : Sequence[int]
            Concept IDs.
        concept_texts : Sequence[str]
            Text(s) to embed for each concept.
        batch_size : Optional[int]
            Batch size for embedding generation.

        Returns
        -------
        ndarray
            Embedding matrix generated.
        """
        if len(concept_ids) != len(concept_texts):
            raise ValueError(
                f"Mismatch between #concept_ids ({len(concept_ids)}) and "
                f"#concept_texts ({len(concept_texts)})."
            )
        embeddings = self.embed_texts(
            list(concept_texts),
            batch_size=batch_size,
        )
        self.upsert_concept_embeddings(
            session=session,
            index_type=index_type,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )
        return embeddings

    def get_nearest_concepts_by_texts(
        self,
        session: Session,
        index_type: IndexType,
        query_texts: str | Tuple[str, ...] | List[str],
        *,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[Mapping[int, float], ...]:
        """Return nearest stored concepts for query texts.

        Convenience wrapper that embeds the query texts before performing
        the nearest neighbor search.

        Parameters
        ----------
        session : Session
            SQLAlchemy session for any required relational access.
        index_type : IndexType
            The type of vector index used to store the embeddings.
        query_texts : str | Tuple[str, ...] | List[str]
            The text(s) to embed and search with.
        metric_type : MetricType
            The similarity or distance metric to use.
        concept_filter : Optional[EmbeddingConceptFilter], optional
            A filter to specify which concepts to consider.
        batch_size : Optional[int], optional
            Batch size for embedding generation.

        Returns
        -------
        Tuple[Mapping[int, float], ...]
            A tuple of dictionaries containing nearest concept matches.
        """
        if isinstance(query_texts, str):
            query_texts = (query_texts,)
        elif isinstance(query_texts, (list, tuple)):
            query_texts = tuple(query_texts)
        else:
            raise ValueError(
                f"Invalid type for query_texts: {type(query_texts)}. "
                f"Expected str, list, or tuple."
            )
        query_embeddings = self.embed_texts(query_texts, batch_size=batch_size)
        return self.get_nearest_concepts(
            session=session,
            index_type=index_type,
            query_embedding=query_embeddings,
            metric_type=metric_type,
            concept_filter=concept_filter,
        )


class EmbeddingInterface(EmbeddingWriterInterface):
    """Compatibility facade over the provider-aware reader/writer interfaces.

    This preserves the older branch-local API while delegating storage and
    registry behavior to the upstream provider-aware implementation.
    """

    def __init__(
        self,
        embedding_client: Optional[EmbeddingClient] = None,
        backend=None,
        storage_base_dir: Optional[str] = None,
        registry_db_name: Optional[str] = None,
        backend_name: Optional[str | BackendType] = None,
    ):
        if embedding_client is None and backend is None and backend_name is None:
            raise ValueError("EmbeddingInterface requires an embedding_client or backend/backend_name.")

        if backend is not None:
            self._embedding_client = embedding_client
            self._backend = backend
            self._backend_type = backend.backend_type
            self._provider_type = embedding_client.provider.provider_type if embedding_client is not None else ProviderType.OLLAMA
            self._canonical_model_name = embedding_client.canonical_model_name if embedding_client is not None else ""
            return

        if embedding_client is None:
            self._embedding_client = None
            self._backend = get_embedding_backend(
                backend_name_or_type=backend_name,
                storage_base_dir=storage_base_dir,
                registry_db_name=registry_db_name,
            )
            self._backend_type = self._backend.backend_type
            self._provider_type = ProviderType.OLLAMA
            self._canonical_model_name = ""
            return

        super().__init__(
            embedding_client=embedding_client,
            backend_name_or_type=backend_name,
            storage_base_dir=storage_base_dir,
            registry_db_name=registry_db_name,
        )

    @property
    def backend(self):
        return self._backend

    @property
    def embedding_client(self) -> EmbeddingClient:
        return self._embedding_client

    @classmethod
    def from_backend_name(
        cls,
        embedding_client: Optional[EmbeddingClient] = None,
        backend_name: Optional[str | BackendType] = None,
        storage_base_dir: Optional[str] = None,
        registry_db_name: Optional[str] = None,
    ) -> "EmbeddingInterface":
        return cls(
            embedding_client=embedding_client,
            backend_name=backend_name,
            storage_base_dir=storage_base_dir,
            registry_db_name=registry_db_name,
        )

    def initialise_tables(self, engine: Engine) -> None:
        self.initialise_store(engine)

    @staticmethod
    def _embedding_prefix_for_role(text_role: Literal["document", "query"]) -> str:
        if text_role == "query":
            return os.getenv(ENV_QUERY_EMBEDDING_PREFIX, "")
        return os.getenv(ENV_DOCUMENT_EMBEDDING_PREFIX, "")

    @classmethod
    def _apply_embedding_prefix(
        cls,
        texts: str | Tuple[str, ...] | List[str],
        *,
        text_role: Literal["document", "query"],
    ) -> str | Tuple[str, ...] | List[str]:
        prefix = cls._embedding_prefix_for_role(text_role)
        if not prefix:
            return texts
        if isinstance(texts, str):
            return f"{prefix}{texts}"
        if isinstance(texts, tuple):
            return tuple(f"{prefix}{text}" for text in texts)
        if isinstance(texts, list):
            return [f"{prefix}{text}" for text in texts]
        raise ValueError(f"Invalid type for texts: {type(texts)}. Expected str, list, or tuple.")

    def ensure_model_registered(
        self,
        *,
        engine: Engine,
        session: Session,
        model_name: str,
        dimensions: int,
        index_type: IndexType,
        metadata: Mapping[str, object] = {},
        overwrite_existing_conflicts: bool = False,
    ) -> EmbeddingModelRecord:
        if overwrite_existing_conflicts:
            self.backend.delete_model(
                engine=engine,
                session=session,
                model_name=model_name,
                provider_type=self.provider_type,
            )
            return self.backend.register_model(
                engine=engine,
                model_name=model_name,
                provider_type=self.provider_type,
                dimensions=dimensions,
                index_type=index_type,
                metadata=metadata,
            )

        existing = self.backend.get_registered_model(
            model_name=model_name,
            provider_type=self.provider_type,
            index_type=index_type,
        )
        if existing is None and self.backend.has_stale_model_artifacts(model_name):
            raise RuntimeError(
                f"Backend artifacts already exist for model '{model_name}' but no matching "
                "SQL registration was found. Re-run with "
                "`--overwrite-model-registration` to force a clean rebuild."
            )
        if existing is not None:
            if existing.dimensions != dimensions:
                raise RuntimeError(
                    f"Model '{model_name}' is already registered with dimensions "
                    f"{existing.dimensions}, not {dimensions}."
                )
            if existing.metadata != metadata:
                raise RuntimeError(
                    f"Model '{model_name}' is already registered with different metadata."
                )
            return existing

        return self.backend.register_model(
            engine=engine,
            model_name=model_name,
            provider_type=self.provider_type,
            dimensions=dimensions,
            index_type=index_type,
            metadata=metadata,
        )

    def embed_texts(
        self,
        texts: str | Tuple[str, ...] | List[str],
        *,
        embedding_client: Optional[EmbeddingClient] = None,
        batch_size: Optional[int] = None,
        text_role: Literal["document", "query"] = "document",
    ) -> np.ndarray:
        client = embedding_client or self.embedding_client
        if client is None:
            raise RuntimeError(f"No embedding client is configured for {self.__class__.__name__}.")
        prefixed_texts = self._apply_embedding_prefix(texts, text_role=text_role)
        return client.embeddings(prefixed_texts, batch_size=batch_size)

    def upsert_concept_embeddings(
        self,
        *,
        session: Session,
        model_name: str,
        index_type: IndexType,
        concept_ids: Sequence[int],
        embeddings: ndarray,
    ) -> None:
        self.backend.upsert_embeddings(
            session=session,
            model_name=model_name,
            provider_type=self.provider_type,
            index_type=index_type,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )

    def embed_and_upsert_concepts(
        self,
        *,
        session: Session,
        model_name: str,
        index_type: IndexType,
        concept_ids: Sequence[int],
        concept_texts: Sequence[str],
        embedding_client: Optional[EmbeddingClient] = None,
        batch_size: Optional[int] = None,
    ) -> ndarray:
        embeddings = self.embed_texts(
            list(concept_texts),
            embedding_client=embedding_client,
            batch_size=batch_size,
            text_role="document",
        )
        self.upsert_concept_embeddings(
            session=session,
            model_name=model_name,
            index_type=index_type,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )
        return embeddings

    def get_nearest_concepts_by_texts(
        self,
        session: Session,
        embedding_model_name: str,
        index_type: IndexType,
        query_texts: str | Tuple[str, ...] | List[str],
        *,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[Mapping[int, float], ...]:
        query_embeddings = self.embed_texts(
            query_texts,
            batch_size=batch_size,
            text_role="query",
        )
        return self.get_nearest_concepts(
            session=session,
            index_type=index_type,
            query_embedding=query_embeddings,
            metric_type=metric_type,
            concept_filter=concept_filter,
        )

    def get_concepts_without_embedding_count(
        self,
        *,
        session: Session,
        model_name: str,
        index_type: IndexType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> int:
        return super().get_concepts_without_embedding_count(
            session=session,
            index_type=index_type,
            concept_filter=concept_filter,
        )

    def get_concepts_without_embedding_query(
        self,
        *,
        session: Session,
        model_name: str,
        index_type: IndexType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: Optional[int] = None,
    ) -> Select:
        del session
        return super().q_get_concepts_without_embedding(
            index_type=index_type,
            concept_filter=concept_filter,
            limit=limit,
        )

    def rebuild_model_indexes(
        self,
        *,
        session: Session,
        model_name: str,
        metric_types: Optional[Sequence[MetricType]] = None,
        batch_size: int = 100_000,
    ) -> None:
        rebuild = getattr(self.backend, "rebuild_model_indexes", None)
        if rebuild is None:
            raise NotImplementedError(
                f"Backend {self.backend.backend_name!r} does not implement explicit index rebuilds."
            )
        rebuild(
            session=session,
            model_name=model_name,
            provider_type=self.provider_type,
            metric_types=metric_types,
            batch_size=batch_size,
        )
