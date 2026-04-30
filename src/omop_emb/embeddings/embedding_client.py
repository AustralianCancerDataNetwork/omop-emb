"""Embedding client for OMOP concept vectorisation.

Wraps any OpenAI-compatible endpoint to provide batched text embedding with
numpy output, automatic embedding-dimension discovery, and cosine-similarity
helpers.  The canonical model name (``self.model``) is the stable key stored
in the omop-emb registry.
"""

from __future__ import annotations

import logging
logger = logging.getLogger(__name__)
import os
from typing import Any, List, Optional, Tuple, Union, Dict
from enum import StrEnum

import numpy as np
from openai import OpenAI

from .embedding_providers import EmbeddingProvider, get_provider_for_api_base
from omop_emb.config import (
    ENV_DOCUMENT_EMBEDDING_PREFIX,
    ENV_QUERY_EMBEDDING_PREFIX,
    ENV_EMBEDDING_DIM,
)


class EmbeddingRole(StrEnum):
    """Enum for embedding roles, used to apply different prefixes to texts based on their role."""
    DOCUMENT = "document"
    QUERY = "query"


class EmbeddingClientError(RuntimeError):
    """Custom exception for embedding client runtime errors."""
    pass


class EmbeddingClient:
    """Client for generating text embeddings over any OpenAI-compatible endpoint.

    Parameters
    ----------
    model : str
        Model name.  Canonicalised by the provider on construction
        (e.g. ``'llama3'`` → ``'llama3:8b'`` for Ollama).  After
        construction ``self.model`` is the stable key used in the omop-emb
        registry.
    api_base : str
        API endpoint base URL, e.g. ``'http://localhost:11434/v1'``.
    api_key : str, optional
        API key.  Defaults to ``'ollama'`` (ignored by Ollama, required by
        the OpenAI SDK).
    embedding_batch_size : int, optional
        Number of texts per API call.  Default is 32.
    provider : EmbeddingProvider, optional
        Controls model-name canonicalisation and embedding-dimension
        discovery.  Inferred from *api_base* / *api_key* when omitted.
    """

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str = "ollama",
        embedding_batch_size: int = 32,
        provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        if provider is None:
            provider = get_provider_for_api_base(api_base, api_key)
        self._provider = provider
        self._model = provider.canonical_model_name(model)
        self._embedding_batch_size = embedding_batch_size
        self._embedding_dim: Optional[int] = None
        self._base_client = OpenAI(base_url=api_base, api_key=api_key)
        doc_prefix, query_prefix = self.load_embedding_prefixes()

        self._embedding_prefixes = {
            EmbeddingRole.DOCUMENT: doc_prefix,
            EmbeddingRole.QUERY: query_prefix,
        }

        logger.info(
            f"{EmbeddingClient.__name__} initialised for model={self._model!r}.\n"
            f"URL: {self._base_client.base_url} | Provider: {type(self._provider).__name__}"
        )

    @property
    def provider(self) -> EmbeddingProvider:
        return self._provider

    @property
    def canonical_model_name(self) -> str:
        return self._model

    @property
    def api_base(self):
        return self._base_client.base_url

    @property
    def api_key(self) -> str:
        return self._base_client.api_key

    @property
    def embedding_batch_size(self) -> int:
        return self._embedding_batch_size

    @property
    def base_client(self) -> OpenAI:
        return self._base_client
    
    def embedding_role_prefixes(self) -> Dict[EmbeddingRole, str]:
        """Return a mapping of embedding roles to their configured prefixes."""
        return self._embedding_prefixes


    @property
    def embedding_dim(self) -> int:
        """Embedding vector dimension, resolved on first access and cached.

        Resolution order:
        1. ``OMOP_EMB_EMBEDDING_DIM`` environment variable (explicit override).
        2. Provider API discovery (e.g. Ollama ``/api/show``).
        3. Live probe: embed the string ``"test"`` and read the returned shape.
           One extra API call, but works for any OpenAI-compatible endpoint
           that does not expose a dimension-discovery route.
        """
        if self._embedding_dim is not None:
            return self._embedding_dim

        env_val = os.getenv(ENV_EMBEDDING_DIM)
        if env_val is not None:
            self._embedding_dim = int(env_val)
            logger.debug(f"Embedding dimension set from env {ENV_EMBEDDING_DIM}={self._embedding_dim}.")
            return self._embedding_dim

        try:
            self._embedding_dim = self._provider.get_embedding_dim(
                self._model, self._base_client.base_url
            )
            return self._embedding_dim
        except NotImplementedError:
            pass

        logger.info(
            "Provider cannot discover embedding dimension automatically. "
            "Probing via a test API call — this happens once and is then cached."
        )
        response = self._base_client.embeddings.create(model=self._model, input=["test"])
        self._embedding_dim = len(response.data[0].embedding)
        logger.info(f"Embedding dimension discovered via live probe: {self._embedding_dim}.")
        return self._embedding_dim

    def embeddings(
        self,
        text: Union[str, List[str], Tuple[str, ...]],
        embedding_role: EmbeddingRole,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Generate embeddings for one or more texts.

        Parameters
        ----------
        text : str | list[str] | tuple[str, ...]
            Input text(s) to embed.
        embedding_role : EmbeddingRole
            Role of the input text(s), used to apply different prefixes based on the role.
        batch_size : int, optional
            Overrides ``embedding_batch_size`` for this call.

        Returns
        -------
        np.ndarray
            2-D float array of shape ``(n_texts, embedding_dim)``.
        """
        if batch_size is None:
            batch_size = self._embedding_batch_size

        if isinstance(text, str):
            text = (text,)
        elif isinstance(text, list):
            text = tuple(text)
            
        text = self._apply_embedding_prefix(text, text_role=embedding_role)

        buffer: list[list[float]] = []
        for start in range(0, len(text), batch_size):
            chunk = text[start : start + batch_size]
            logger.debug(f"Embedding batch [{start}:{start + len(chunk)}]")
            response = self._base_client.embeddings.create(
                model=self._model, input=chunk
            )
            buffer.extend(emb.embedding for emb in response.data)

        result = np.array(buffer)
        if result.ndim != 2:
            raise RuntimeError(f"Expected 2-D embedding array, got shape {result.shape}")
        if result.shape[0] != len(text):
            raise RuntimeError(f"Expected {len(text)} embeddings, got {result.shape[0]}")
        return result

    def similarity(
        self,
        terms: Union[str, List[str], np.ndarray],
        terms_to_match: Union[str, List[str], np.ndarray],
        terms_role: EmbeddingRole,
        terms_to_match_role: EmbeddingRole,
        **kwargs: Any,
    ) -> np.ndarray:
        """Cosine-similarity matrix between two sets of terms or embeddings."""
        if isinstance(terms, (str, list)):
            terms = self.embeddings(terms, embedding_role=terms_role, **kwargs)
        if isinstance(terms_to_match, (str, list)):
            terms_to_match = self.embeddings(terms_to_match, embedding_role=terms_to_match_role, **kwargs)
        return self.cosine_similarity(terms, terms_to_match)

    @staticmethod
    def cosine_similarity(vecs_a: np.ndarray, vecs_b: np.ndarray) -> np.ndarray:
        """Cosine similarity between row-vector matrices (MxD, NxD → MxN)."""
        if vecs_a.ndim != 2 or vecs_b.ndim != 2:
            raise RuntimeError(f"Expected 2-D arrays, got shapes {vecs_a.shape} and {vecs_b.shape}")
        norm_a = np.linalg.norm(vecs_a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(vecs_b, axis=1, keepdims=True)
        norm_a[norm_a == 0] = 1e-10
        norm_b[norm_b == 0] = 1e-10
        return np.dot(vecs_a / norm_a, (vecs_b / norm_b).T)

    @staticmethod
    def l2_norm(vecs_a: np.ndarray, vecs_b: np.ndarray) -> float:
        """L2 norm between row-vector matrices (MxD, NxD → MxN)."""
        return float(np.linalg.norm(vecs_a - vecs_b))
    

    def euclidean_distance(
        self,
        text1: str,
        text2: str,
        text1_role: EmbeddingRole,
        text2_role: EmbeddingRole,
    ) -> float:
        """Euclidean distance between embeddings of two texts."""
        a = self.embeddings(text1, embedding_role=text1_role)
        b = self.embeddings(text2, embedding_role=text2_role)
        return float(np.linalg.norm(a - b))
        

    @staticmethod
    def load_embedding_prefixes() -> Tuple[str, str]:
        """Load embedding prefixes for document and query roles from environment variables.

        Returns
        -------
        Tuple[str, str]
            A tuple containing the document embedding prefix and the query embedding prefix.
        """
        document_embedding_prefix = os.getenv(ENV_DOCUMENT_EMBEDDING_PREFIX, "")
        query_embedding_prefix = os.getenv(ENV_QUERY_EMBEDDING_PREFIX, "")

        for role, prefix, var_name in [
            (EmbeddingRole.DOCUMENT, document_embedding_prefix, ENV_DOCUMENT_EMBEDDING_PREFIX),
            (EmbeddingRole.QUERY, query_embedding_prefix, ENV_QUERY_EMBEDDING_PREFIX),
        ]:
            if prefix:
                logger.info(
                    f"{role.value.capitalize()} embedding prefix loaded from {var_name}={prefix!r}. "
                    f"All {role.value} texts will be prepended with this prefix."
                )
            else:
                logger.warning(
                    f"{role.value.capitalize()} embedding prefix is not set ({var_name} is empty). "
                    f"This is fine for symmetric models. For asymmetric models (e.g. nomic-embed-text, "
                    f"E5, BGE), set {var_name} to the required task prefix.\n"
                    f"Example (nomic-embed-text): {ENV_DOCUMENT_EMBEDDING_PREFIX}='search_document: ' for {EmbeddingRole.DOCUMENT.value} "
                    f"and {ENV_QUERY_EMBEDDING_PREFIX}='search_query: ' for {EmbeddingRole.QUERY.value}."
                )

        return document_embedding_prefix, query_embedding_prefix
    
    def _apply_embedding_prefix(
        self,
        texts: str | Tuple[str, ...] | List[str],
        *,
        text_role: EmbeddingRole,
    ) -> str | Tuple[str, ...] | List[str]:
        
        try:
            prefix = self._embedding_prefixes[text_role]
        except KeyError:
            raise ValueError(f"Invalid embedding role {text_role!r}. Expected one of {[role.value for role in EmbeddingRole]}.")
        
        if not prefix:
            return texts
        if isinstance(texts, str):
            return f"{prefix}{texts}"
        if isinstance(texts, tuple):
            return tuple(f"{prefix}{text}" for text in texts)
        if isinstance(texts, list):
            return [f"{prefix}{text}" for text in texts]
        raise ValueError(f"Invalid type for texts: {type(texts)}. Expected str, list, or tuple.")
