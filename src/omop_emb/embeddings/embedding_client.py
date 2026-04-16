"""Embedding client for OMOP concept vectorisation.

Wraps any OpenAI-compatible endpoint to provide batched text embedding with
numpy output, automatic embedding-dimension discovery, and cosine-similarity
helpers.  The canonical model name (``self.model``) is the stable key stored
in the omop-emb registry.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union

import numpy as np
from openai import OpenAI

from .embedding_providers import EmbeddingProvider, get_provider_for_api_base

logger = logging.getLogger(__name__)

CHAT_MESSAGE_DICT: TypeAlias = Dict[str, str]


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
        logger.info(f"EmbeddingClient initialised for model={self._model!r}")

    @property
    def provider(self) -> EmbeddingProvider:
        return self._provider

    @property
    def model(self) -> str:
        """Canonical model name — the stable key stored in the omop-emb registry."""
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

    @property
    def embedding_dim(self) -> int:
        """Embedding vector dimension, auto-discovered on first access.

        For Ollama endpoints this queries ``POST /api/show``.  For
        OpenAI-compatible providers the dimension must be supplied at
        construction time via a provider that implements
        :meth:`~omop_emb.config.EmbeddingProvider.get_embedding_dim`.

        Raises
        ------
        ValueError
            If the provider cannot determine the dimension from the API.
        NotImplementedError
            If the provider requires an explicit dimension
            (see :class:`~omop_emb.config.OpenAICompatProvider`).
        """
        if self._embedding_dim is None:
            self._embedding_dim = self._provider.get_embedding_dim(
                self._model, self._base_client.base_url
            )
        return self._embedding_dim

    def embeddings(
        self,
        text: Union[str, List[str], Tuple[str, ...]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Generate embeddings for one or more texts.

        Parameters
        ----------
        text : str | list[str] | tuple[str, ...]
            Input text(s) to embed.
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
        **kwargs: Any,
    ) -> np.ndarray:
        """Cosine-similarity matrix between two sets of terms or embeddings."""
        if isinstance(terms, (str, list)):
            terms = self.embeddings(terms, **kwargs)
        if isinstance(terms_to_match, (str, list)):
            terms_to_match = self.embeddings(terms_to_match, **kwargs)
        return self.cosine_similarity(terms, terms_to_match)

    @staticmethod
    def cosine_similarity(vecs_a: np.ndarray, vecs_b: np.ndarray) -> np.ndarray:
        """Cosine similarity between row-vector matrices (M×D, N×D → M×N)."""
        if vecs_a.ndim != 2 or vecs_b.ndim != 2:
            raise RuntimeError(f"Expected 2-D arrays, got shapes {vecs_a.shape} and {vecs_b.shape}")
        norm_a = np.linalg.norm(vecs_a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(vecs_b, axis=1, keepdims=True)
        norm_a[norm_a == 0] = 1e-10
        norm_b[norm_b == 0] = 1e-10
        return np.dot(vecs_a / norm_a, (vecs_b / norm_b).T)

    def euclidean_distance(self, text1: str, text2: str) -> float:
        """Euclidean distance between embeddings of two texts."""
        a = self.embeddings(text1)
        b = self.embeddings(text2)
        return float(np.linalg.norm(a - b))
