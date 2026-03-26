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
    EmbeddingConceptFilter,
    EmbeddingModelRecord,
    get_embedding_backend,
)
from .backends.config import IndexType, MetricType


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
    embedding_client: LLMClient
    backend: EmbeddingBackend = field(default_factory=get_embedding_backend)
    _query_embedding_cache: dict[tuple[str, str], np.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @property
    def embedding_dim(self) -> int:
        return self.embedding_client.embedding_dim

    @classmethod
    def from_backend_name(
        cls,
        embedding_client: LLMClient,
        backend_name: Optional[str] = None,
        *,
        faiss_base_dir: Optional[str] = None,
    ) -> EmbeddingInterface:
        return cls(
            backend=get_embedding_backend(
                backend_name=backend_name,
                faiss_base_dir=faiss_base_dir,
            ),
            embedding_client=embedding_client,
        )
    
    def initialise_store(self, engine: Engine) -> None:
        self.backend.initialise_store(engine)

    def get_model_table_name(
        self,
        session: Session,
        model_name: str,
    ) -> Optional[str]:
        """
        Legacy helper preserved for compatibility.

        Historically this returned a PostgreSQL table name. Under the backend
        abstraction it returns the backend-specific storage identifier.
        """
        record = self.backend.get_registered_model(session=session, model_name=model_name)
        return record.storage_identifier if record is not None else None

    def is_model_registered(
        self,
        session: Session,
        model_name: str,
    ) -> bool:
        return self.backend.is_model_registered(session=session, model_name=model_name)


    def has_any_embeddings(
        self,
        session: Session,
        embedding_model_name: str,
    ) -> bool:
        return self.backend.has_any_embeddings(
            session=session,
            model_name=embedding_model_name,
        )

    def get_nearest_concepts(
        self,
        session: Session,
        embedding_model_name: str,
        query_embedding: np.ndarray,
        *,
        metric_type: MetricType = MetricType.COSINE,
        concept_ids: Optional[Tuple[int, ...]] = None,
        domains: Optional[Tuple[str, ...]] = None,
        vocabularies: Optional[Tuple[str, ...]] = None,
        require_standard: bool = False,
        k: int = 10,
    ) -> Tuple[Mapping[int, float], ...]:
        
        """
        Return nearest stored concepts for the query embedding.

        Parameters
        ----------
        session : Session
            SQLAlchemy session for any required relational access.
        embedding_model_name : str
            Name of the embedding model used to create the embeddings.
        query_embedding : ndarray
            The embedding vector to search with. Expected shape is (q, dimension)
            where q is the number of query vectors and dimension is the size of the embedding space for the model.
        metric_type : MetricType
            The similarity or distance metric to use for nearest neighbor search. This should be compatible with the index type used by the model.
        concept_ids : Optional[Tuple[int, ...]], optional
            If provided, only consider these concept_ids as potential nearest neighbors.
        domains : Optional[Tuple[str, ...]], optional
            If provided, only consider concepts within these OMOP domains as potential nearest neighbors.
        vocabularies : Optional[Tuple[str, ...]], optional
            If provided, only consider concepts from these vocabularies as potential nearest neighbors.
        require_standard : bool, optional
            If True, only consider standard concepts as potential nearest neighbors. By default False.
        k : int, optional
            K nearest neighbors to return for each query vector. Default is 10.

        Returns
        -------
        Tuple[Mapping[int, float], ...]
            A tuple of dictionaries containing nearest concept matches for each query vector. The outer tuple corresponds to the query vectors in order, and each inner dictionary contains the nearest matches for that query vector, sorted by similarity. Returned shape is (q, k) where q is the number of query vectors and k is the number of nearest neighbors returned per query.
        """
        

        concept_filter = EmbeddingConceptFilter(
            concept_ids=concept_ids,
            domains=domains,
            vocabularies=vocabularies,
            require_standard=require_standard,
        )
        nearest_concepts = self.backend.get_nearest_concepts(
            session=session,
            model_name=embedding_model_name,
            query_embedding=query_embedding,
            concept_filter=concept_filter,
            metric_type=metric_type,
            k=k
        )
        return tuple({match_per_query.concept_id: match_per_query.similarity for match_per_query in match} for match in nearest_concepts)

    def get_embeddings_by_concept_ids(
        self,
        session: Session,
        embedding_model_name: str,
        concept_ids: Tuple[int, ...],
    ) -> Mapping[int, Sequence[float]]:
        return self.backend.get_embeddings_by_concept_ids(
            session=session,
            model_name=embedding_model_name,
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
            concept_ids=concept_ids,
            embeddings=embeddings,
        )
    
    def ensure_model_registered(
        self,
        *,
        engine: Engine,
        session: Session,
        model_name: str,
        dimensions: int,
        index_type: IndexType,
        metadata: Mapping[str, object] = {},
    ) -> EmbeddingModelRecord:
        """
        Ensure the embedding model exists in the selected backend.
        """

        existing = self.backend.get_registered_model(
            session=session,
            model_name=model_name,
        )
        if existing is not None:
            return existing
        else:
            return self.backend.register_model(
                engine=engine,
                model_name=model_name,
                dimensions=dimensions,
                index_type=index_type,
                metadata=metadata,
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
            raise RuntimeError("No embedding client is configured for EmbeddingService.")
        return client.embeddings(texts, batch_size=batch_size)

    def upsert_concept_embeddings(
        self,
        *,
        session: Session,
        model_name: str,
        concept_ids: Sequence[int],
        embeddings: ndarray,
    ) -> None:
        self.backend.upsert_embeddings(
            session=session,
            model_name=model_name,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )

    def embed_and_upsert_concepts(
        self,
        *,
        session: Session,
        model_name: str,
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
            concept_ids=concept_ids,
            embeddings=embeddings,
        )
        return embeddings

    def get_concepts_without_embedding(
        self,
        *,
        session: Session,
        model_name: str,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: Optional[int] = None,
    ) -> Mapping[int, str]:
        return self.backend.get_concepts_without_embedding(
            session=session,
            model_name=model_name,
            concept_filter=concept_filter,
            limit=limit,
        )
    
    def get_concepts_without_embedding_query(
        self,
        *,
        session: Session,
        model_name: str,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: Optional[int] = None,
    ) -> Select:
        return self.backend.get_concepts_without_embedding_query(
            session=session,
            model_name=model_name,
            concept_filter=concept_filter,
            limit=limit,
        )
    
    def get_concepts_without_embedding_count(
        self,
        *,
        session: Session,
        model_name: str,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> int:
        return self.backend.get_concepts_without_embedding_count(
            session=session,
            model_name=model_name,
            concept_filter=concept_filter,
        )

    def get_or_create_query_embedding(
        self,
        *,
        session: Session,
        model_name: str,
        query_text: str,
        embedding_client: Optional[LLMClient] = None,
        reusable_concept_ids: Optional[Sequence[int]] = None,
        cache_result: bool = True,
    ) -> np.ndarray:
        """
        Reuse or compute an embedding for a free-text query.

        Resolution order:
        1. in-process query cache
        2. stored embeddings for any supplied reusable concept IDs
        3. on-the-fly embedding generation via ``LLMClient``

        This does not currently persist arbitrary query-text embeddings to a
        backend-level query cache. It provides process-local caching only.
        """

        cache_key = (model_name, query_text)
        cached = self._query_embedding_cache.get(cache_key)
        if cached is not None:
            return cached

        if reusable_concept_ids:
            stored_embeddings = self.backend.get_embeddings_by_concept_ids(
                session=session,
                model_name=model_name,
                concept_ids=reusable_concept_ids,
            )
            for concept_id in reusable_concept_ids:
                reusable = stored_embeddings.get(int(concept_id))
                if reusable is not None:
                    vector = np.asarray(reusable, dtype=np.float32).reshape(1, -1)
                    if cache_result:
                        self._query_embedding_cache[cache_key] = vector
                    return vector

        # Future consideration: if repeated free-text queries become common, add
        # an explicit backend-level query embedding cache instead of recomputing
        # here and only storing the result in the process-local cache.
        vector = self.embed_texts(
            query_text,
            embedding_client=embedding_client,
        ).astype(np.float32, copy=False)
        if cache_result:
            self._query_embedding_cache[cache_key] = vector
        return vector

    def clear_query_cache(self) -> None:
        self._query_embedding_cache.clear()

    def cached_query_keys(self) -> Iterable[tuple[str, str]]:
        return tuple(self._query_embedding_cache.keys())