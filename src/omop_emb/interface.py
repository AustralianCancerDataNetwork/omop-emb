from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence, Tuple, List

import numpy as np
from numpy import ndarray
from sqlalchemy import Engine, Select
from sqlalchemy.orm import Session

from omop_llm import LLMClient

from .backends import (
    EmbeddingBackend,
    get_embedding_backend,
)
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter
from omop_emb.model_registry import EmbeddingModelRecord
from .config import BackendType, IndexType, MetricType


@dataclass
class EmbeddingInterface:
    """
    Backend-neutral interface for embedding operations.

    Responsibilities
    ----------------
    - initialize the selected backend store
    - ensure an embedding model is registered
    - generate embeddings with an ``LLMClient``
    - upsert concept embeddings through the selected backend
    - provide a reusable in-process cache for query-text embeddings

    """
    embedding_client: Optional[LLMClient] = None
    backend: EmbeddingBackend = field(default_factory=get_embedding_backend)


    @property
    def embedding_dim(self) -> Optional[int]:
        if self.embedding_client is not None:
            return self.embedding_client.embedding_dim
        return None

    @classmethod
    def from_backend_name(
        cls,
        embedding_client: Optional[LLMClient] = None,
        backend_name: Optional[str | BackendType] = None,
        storage_base_dir: Optional[str] = None,
        registry_db_name: Optional[str] = None,
    ) -> EmbeddingInterface:
        """Create an interface by resolving and constructing a backend.

        Parameters
        ----------
        embedding_client : LLMClient, optional
            Optional default client used for text embedding generation.
        backend_name : str | BackendType, optional
            Backend selector passed to ``get_embedding_backend``.
            Resolution order:
            1. explicit ``backend_name`` argument
            2. ``OMOP_EMB_BACKEND`` environment variable
        storage_base_dir : str, optional
            Optional storage directory forwarded to backend constructor.
            Backend constructors apply fallback resolution (typically explicit arg,
            then ``OMOP_EMB_BASE_STORAGE_DIR``, then backend default).
        registry_db_name : str, optional
            Optional registry database filename forwarded to the backend.
        """
        return cls(
            backend=get_embedding_backend(
                backend_name=backend_name,
                storage_base_dir=storage_base_dir,
                registry_db_name=registry_db_name,
            ),
            embedding_client=embedding_client,
        )
    
    def setup_and_register_model(
        self,
        engine: Engine,
        model_name: str,
        dimensions: int,
        index_type: IndexType,
        metadata: Mapping[str, object] = {},
    ) -> None:
        self.initialise_store(engine)
        self.register_model(
            engine=engine,
            model_name=model_name,
            dimensions=dimensions,
            index_type=index_type,
            metadata=metadata,
        )
    
    def initialise_store(self, engine: Engine) -> None:
        self.backend.initialise_store(engine)

    def get_model_table_name(
        self,
        model_name: str,
        index_type: IndexType,
    ) -> Optional[str]:
        """
        Legacy helper preserved for compatibility.

        Historically this returned a PostgreSQL table name. Under the backend
        abstraction it returns the backend-specific storage identifier.
        """
        record = self.backend.get_registered_model(model_name=model_name, index_type=index_type)
        return record.storage_identifier if record is not None else None

    def is_model_registered(
        self,
        model_name: str,
        index_type: IndexType,
    ) -> bool:
        return self.backend.is_model_registered(model_name=model_name, index_type=index_type)

    def register_model(
        self,
        engine: Engine,
        model_name: str,
        dimensions: int,
        index_type: IndexType,
        metadata: Mapping[str, object] = {},
    ) -> EmbeddingModelRecord:
        return self.backend.register_model(
            engine=engine,
            model_name=model_name,
            dimensions=dimensions,
            index_type=index_type,
            metadata=metadata,
        )

    def has_any_embeddings(
        self,
        session: Session,
        embedding_model_name: str,
        index_type: IndexType,
    ) -> bool:
        return self.backend.has_any_embeddings(
            session=session,
            model_name=embedding_model_name,
            index_type=index_type,
        )

    def get_nearest_concepts(
        self,
        session: Session,
        model_name: str,
        index_type: IndexType,
        query_embedding: np.ndarray,
        *,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> Tuple[Mapping[int, float], ...]:
        """
        Return nearest stored concepts for the query embedding.

        Parameters
        ----------
        session : Session
            SQLAlchemy session for any required relational access.
        model_name : str
            Name of the embedding model used to create the embeddings.
        index_type : IndexType
            The type of vector index used to store the embeddings.
        query_embedding : ndarray
            The embedding vector to search with. Expected shape is (q, dimension)
            where q is the number of query vectors and dimension is the size of the embedding space for the model.
        metric_type : MetricType
            The similarity or distance metric to use for nearest neighbor search. This must be compatible with the index type used by the database.
        concept_filter : Optional[EmbeddingConceptFilter], optional
            A filter to specify which concepts to consider as potential nearest neighbors. The `limit` field of this filter determines the number of neighbors returned.

        Returns
        -------
        Tuple[Mapping[int, float], ...]
            A tuple of dictionaries containing nearest concept matches for each query vector. The outer tuple corresponds to the query vectors in order, and each inner dictionary contains the nearest matches for that query vector, sorted by similarity. Returned shape is (q, limit) where q is the number of query vectors and limit is the number of nearest neighbors returned per query as determined by the `concept_filter` argument or backend default.
        """
        if not isinstance(metric_type, MetricType):
            raise TypeError(
                f"metric_type must be MetricType, got {type(metric_type).__name__}."
            )
        nearest_concepts = self.backend.get_nearest_concepts(
            session=session,
            model_name=model_name,
            index_type=index_type,
            query_embeddings=query_embedding,
            concept_filter=concept_filter,
            metric_type=metric_type
        )
        return tuple({match.concept_id: match.similarity for match in matches_per_query} for matches_per_query in nearest_concepts)
    
    def get_nearest_concepts_by_texts(
        self,
        session: Session,
        embedding_model_name: str,
        index_type: IndexType,
        query_texts: str | Tuple[str, ...] | List[str],
        *,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        batch_size: Optional[int] = None
    ) -> Tuple[Mapping[int, float], ...]:
        """
        Return nearest stored concepts for the query embedding. Convenience wrapper that embeds the query texts before performing the nearest neighbor search.

        The number of neighbors returned is determined by the `limit` field of the `concept_filter` argument. If `limit` is not set, a backend default may be used.

        Parameters
        ----------
        session : Session
            SQLAlchemy session for any required relational access.
        embedding_model_name : str
            Name of the embedding model used to create the embeddings.
        index_type : IndexType
            The type of vector index used to store the embeddings.
        query_texts : str | Tuple[str, ...] | List[str]
            The text(s) to embed and search with. If a single string is provided, it will be embedded and searched as one query. If a tuple or list of strings is provided, each string will be embedded and searched separately, and the results will be returned in the same order as the input texts.
        metric_type : MetricType
            The similarity or distance metric to use for nearest neighbor search. This should be compatible with the index type used by the model.
        concept_filter : Optional[EmbeddingConceptFilter], optional
            A filter to specify which concepts to consider as potential nearest neighbors. The `limit` field of this filter determines the number of neighbors returned.
        batch_size : Optional[int], optional
            If provided, this batch size will be used when embedding the query texts. If not provided, the default batch size of the embedding client will be used.

        Returns
        -------
        Tuple[Mapping[int, float], ...]
            A tuple of dictionaries containing nearest concept matches for each query vector. The outer tuple corresponds to the query vectors in order, and each inner dictionary contains the nearest matches for that query vector, sorted by similarity. Returned shape is (q, limit) where q is the number of query vectors and limit is the number of nearest neighbors returned per query.
        """
        if isinstance(query_texts, str):
            query_texts = (query_texts,)
        elif isinstance(query_texts, (list, tuple)):
            query_texts = tuple(query_texts)
        else:
            raise ValueError(f"Invalid type for query_texts: {type(query_texts)}. Expected str, list, or tuple.")
        query_embeddings = self.embed_texts(query_texts, batch_size=batch_size)
        return self.get_nearest_concepts(
            session=session,
            model_name=embedding_model_name,
            index_type=index_type,
            query_embedding=query_embeddings,
            metric_type=metric_type,
            concept_filter=concept_filter,
        )

    def get_embeddings_by_concept_ids(
        self,
        session: Session,
        embedding_model_name: str,
        index_type: IndexType,
        concept_ids: Tuple[int, ...],
    ) -> Mapping[int, Sequence[float]]:
        return self.backend.get_embeddings_by_concept_ids(
            session=session,
            model_name=embedding_model_name,
            index_type=index_type,
            concept_ids=concept_ids,
        )

    def initialise_tables(self, engine: Engine):
        """
        Legacy name preserved for compatibility.
        """

        return self.backend.initialise_store(engine)

    def add_to_db(
        self,
        session: Session,
        index_type: IndexType,
        concept_ids: Tuple[int, ...],
        embeddings: ndarray,
        model: str,
    ):
        assert embeddings.ndim == 2, f"Expected 2 dimensions of embeddings. Got {embeddings.ndim}"
        assert len(concept_ids) == embeddings.shape[0], (
            f"Mismatch between #concept_ids ({len(concept_ids)}) and embedding "
            f"dimensionality ({embeddings.shape[0]})"
        )

        return self.backend.upsert_embeddings(
            session=session,
            model_name=model,
            index_type=index_type,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )

    def embed_texts(
        self,
        texts: str | Tuple[str, ...] | List[str],
        *,
        embedding_client: Optional[LLMClient] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        client = embedding_client or self.embedding_client
        if client is None:
            raise RuntimeError(f"No embedding client is configured for {self.__class__.__name__}.")
        return client.embeddings(texts, batch_size=batch_size)

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
        embedding_client: Optional[LLMClient] = None,
        batch_size: Optional[int] = None,
    ) -> ndarray:
        if len(concept_ids) != len(concept_texts):
            raise ValueError(
                f"Mismatch between #concept_ids ({len(concept_ids)}) and "
                f"#concept_texts ({len(concept_texts)})."
            )
        embeddings = self.embed_texts(
            list(concept_texts),
            embedding_client=embedding_client,
            batch_size=batch_size,
        )
        self.upsert_concept_embeddings(
            session=session,
            model_name=model_name,
            index_type=index_type,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )
        return embeddings

    def get_concepts_without_embedding(
        self,
        *,
        session: Session,
        model_name: str,
        index_type: IndexType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: Optional[int] = None,
    ) -> Mapping[int, str]:
        return self.backend.get_concepts_without_embedding(
            session=session,
            model_name=model_name,
            index_type=index_type,
            concept_filter=concept_filter,
            limit=limit,
        )
    
    def q_get_concepts_without_embedding(
        self,
        *,
        model_name: str,
        index_type: IndexType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: Optional[int] = None,
    ) -> Select:
        return self.backend.q_get_concepts_without_embedding(
            model_name=model_name,
            index_type=index_type,
            concept_filter=concept_filter,
            limit=limit,
        )
    
    def get_concepts_without_embedding_count(
        self,
        *,
        session: Session,
        model_name: str,
        index_type: IndexType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> int:
        return self.backend.get_concepts_without_embedding_count(
            session=session,
            model_name=model_name,
            index_type=index_type,
            concept_filter=concept_filter,
        )
