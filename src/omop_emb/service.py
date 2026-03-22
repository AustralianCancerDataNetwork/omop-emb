from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
from numpy import ndarray
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from omop_llm import LLMClient

from .backends import (
    EmbeddingBackend,
    EmbeddingIndexConfig,
    EmbeddingModelRecord,
    get_embedding_backend,
)


@dataclass
class EmbeddingService:
    """
    Backend-neutral orchestration layer for embedding workflows.

    Responsibilities
    ----------------
    - initialize the selected backend store
    - ensure an embedding model is registered
    - generate embeddings with an ``LLMClient``
    - upsert concept embeddings through the selected backend
    - provide a reusable in-process cache for query-text embeddings

    Notes
    -----
    This service deliberately does not own OMOP concept search logic. When a
    caller wants to reuse an already-stored concept embedding for a query text,
    it can pass candidate concept IDs discovered elsewhere.
    """

    backend: EmbeddingBackend = field(default_factory=get_embedding_backend)
    embedding_client: Optional[LLMClient] = None
    _query_embedding_cache: dict[tuple[str, str], np.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @classmethod
    def from_backend_name(
        cls,
        backend_name: Optional[str] = None,
        *,
        faiss_base_dir: Optional[str] = None,
        embedding_client: Optional[LLMClient] = None,
    ) -> "EmbeddingService":
        return cls(
            backend=get_embedding_backend(
                backend_name=backend_name,
                faiss_base_dir=faiss_base_dir,
            ),
            embedding_client=embedding_client,
        )

    def initialise_store(self, engine: Engine) -> None:
        self.backend.initialise_store(engine)

    def ensure_model_registered(
        self,
        *,
        engine: Engine,
        session: Session,
        model_name: str,
        dimensions: Optional[int] = None,
        index_config: Optional[EmbeddingIndexConfig] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> EmbeddingModelRecord:
        """
        Ensure the embedding model exists in the selected backend.

        ``dimensions`` may be omitted if an ``embedding_client`` is attached,
        in which case the service will infer the dimension from the client.
        """

        existing = self.backend.get_registered_model(
            session=session,
            model_name=model_name,
        )
        if existing is not None:
            return self.backend.register_model(
                engine=engine,
                model_name=model_name,
                dimensions=dimensions or existing.dimensions,
                index_config=index_config,
                metadata=metadata,
            )

        if dimensions is None:
            if self.embedding_client is None:
                raise RuntimeError(
                    f"Model '{model_name}' is not registered and no dimensions were "
                    "supplied. Provide dimensions explicitly or configure an embedding client."
                )
            dimensions = self.embedding_client.embedding_dim

        return self.backend.register_model(
            engine=engine,
            model_name=model_name,
            dimensions=dimensions,
            index_config=index_config,
            metadata=metadata,
        )

    def embed_texts(
        self,
        texts: str | Sequence[str],
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

    def get_missing_concept_ids(
        self,
        *,
        session: Session,
        model_name: str,
        concept_ids: Sequence[int],
    ) -> tuple[int, ...]:
        existing = self.backend.get_embeddings_by_concept_ids(
            session=session,
            model_name=model_name,
            concept_ids=concept_ids,
        )
        return tuple(int(concept_id) for concept_id in concept_ids if int(concept_id) not in existing)

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
