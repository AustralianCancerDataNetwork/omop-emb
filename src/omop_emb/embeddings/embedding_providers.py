"""Embedding provider abstractions for model name canonicalisation and dimension discovery.

An EmbeddingProvider encapsulates the two things that vary across embedding backends:

- **Model name canonicalisation** — e.g. Ollama requires a tag such as
  ``llama3:8b`` while OpenAI-style names carry no tags.
- **Embedding dimension retrieval** — Ollama exposes a ``/api/show`` endpoint;
  OpenAI-compatible APIs do not have an equivalent.

The provider is inferred automatically from the ``api_base`` URL via
:func:`get_provider_for_api_base`, but can also be supplied explicitly to
:class:`~omop_emb.embedding_client.EmbeddingClient` for custom or future backends.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from httpx import URL
import requests

from omop_emb.config import ProviderType


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Subclass this to support a new embedding backend. The two abstract methods
    capture the only behaviour that differs between providers at the
    ``EmbeddingClient`` level; everything else (batched embedding calls via the
    OpenAI-compatible ``/v1/embeddings`` endpoint, similarity helpers, etc.)
    is shared and lives in ``EmbeddingClient`` directly.
    """
    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type as a value of the ProviderType enum."""
        ...

    @abstractmethod
    def canonical_model_name(self, name: str) -> str:
        """Return the canonical form of *name* for this provider.

        The canonical form is the identifier used as a stable key in the
        embedding registry and passed verbatim to the API.
        Implementations should be idempotent — calling this on an already-
        canonical name must return the same string unchanged.

        Parameters
        ----------
        name : str
            Raw model name as supplied by the caller, e.g. ``'llama3'`` or
            ``'text-embedding-3-small'``.

        Returns
        -------
        str
            Canonical model name, e.g. ``'llama3:8b'`` or
            ``'text-embedding-3-small'``.
        """
        ...

    @abstractmethod
    def get_embedding_dim(self, model: str, api_base: URL) -> int:
        """Return the embedding dimension for *model* served at *api_base*.

        Parameters
        ----------
        model : str
            Canonical model name (already processed by
            :meth:`canonical_model_name`).
        api_base : URL
            Base URL of the API endpoint, e.g.
            ``'http://localhost:11434/v1'``.

        Returns
        -------
        int
            Number of dimensions in the embedding vector.

        Raises
        ------
        ValueError
            If the dimension cannot be determined from the API response.
        NotImplementedError
            If the provider does not support automatic dimension retrieval.
        """
        ...


class OllamaProvider(EmbeddingProvider):
    """Provider for models served by Ollama.

    Canonical model names must include an explicit, immutable tag (``name:tag``).
    Both untagged names and the mutable ``:latest`` tag are rejected.
    Embedding dimensions are retrieved via Ollama's ``POST /api/show`` endpoint.
    """

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OLLAMA

    def canonical_model_name(self, name: str) -> str:
        """Require an explicit, immutable model tag.

        Rejects both untagged names and the mutable ``:latest`` tag.

        Parameters
        ----------
        name : str
            Model name with explicit tag, e.g. ``'llama3:8b'`` or
            ``'nomic-embed-text:v1.5'``.

        Returns
        -------
        str
            The input name, validated and stripped of whitespace.

        Raises
        ------
        ValueError
            If the name has no tag, or if the tag is ``:latest``.
        """
        name = name.strip()
        if ":" not in name:
            raise ValueError(
                f"Ollama model name {name!r} must include an explicit tag. "
                f"Use a specific version (e.g. '{name}:8b') instead of relying on "
                f"the mutable ':latest' pointer. Running 'ollama pull {name}' can "
                f"silently change which model version ':latest' refers to, breaking "
                f"consistency between stored embeddings and new query embeddings."
            )

        model_part, tag = name.rsplit(":", 1)
        if tag == "latest":
            raise ValueError(
                f"Ollama model name {name!r} uses the mutable ':latest' tag. "
                f"':latest' can change between 'ollama pull' runs, breaking "
                f"consistency between stored embeddings and new query embeddings. "
                f"Use an explicit, immutable tag (e.g. '<model_name>:8b')."
            )

        return name

    def get_embedding_dim(self, model: str, api_base: URL) -> int:
        """Query ``POST /api/show`` for the embedding dimension.

        Parameters
        ----------
        model : str
            Canonical model name (with tag).
        api_base : URL
            Ollama API base URL, e.g. ``'http://localhost:11434/v1'``.

        Returns
        -------
        int
            Embedding vector dimension.

        Raises
        ------
        ValueError
            If model info or the embedding length key is absent from the
            Ollama response.
        """
        base_url = str(api_base).replace("/v1", "").rstrip("/")
        response = requests.post(f"{base_url}/api/show", json={"name": model}).json()
        model_info = response.get("model_info", {})

        embedding_keys = [k for k in model_info if "embedding_length" in k]
        if len(embedding_keys) == 1:
            return int(model_info[embedding_keys[0]])

        raise ValueError(
            f"Could not determine embedding dimension from Ollama response for "
            f"model '{model}'. Response: {response}"
        )


def get_provider_for_api_base(
    api_base: str,
    api_key: str = "ollama",
) -> EmbeddingProvider:
    """Infer the appropriate :class:`EmbeddingProvider` from *api_base*.

    Detection rules (evaluated in order):

    1. ``'ollama'`` appears anywhere in *api_base* → :class:`OllamaProvider`
    2. *api_base* is a localhost/loopback URL **and** *api_key* is ``'ollama'``
       → :class:`OllamaProvider`
    3. All other URLs → :exc:`ValueError`

    Pass a provider instance explicitly to
    :class:`~omop_emb.embedding_client.EmbeddingClient` to override this inference
    for custom or future backends.

    Parameters
    ----------
    api_base : str
        Base URL of the API endpoint.
    api_key : str, optional
        API key, used as a secondary Ollama signal when the URL alone is
        ambiguous (e.g. a plain ``localhost`` URL).  Default is ``'ollama'``.

    Returns
    -------
    EmbeddingProvider
        A provider instance appropriate for *api_base*.

    Raises
    ------
    ValueError
        If the endpoint cannot be identified as an Ollama instance.
    """
    is_local = "localhost" in api_base or "127.0.0.1" in api_base
    is_ollama = "ollama" in api_base or (is_local and api_key == "ollama")

    if is_ollama:
        return OllamaProvider()
    raise ValueError(
        f"Cannot infer provider from api_base={api_base!r}. "
        "Only Ollama endpoints are supported. Pass an explicit provider "
        "instance to EmbeddingClient to override."
    )


def get_provider_from_provider_type(provider_type: ProviderType) -> EmbeddingProvider:
    """Map a ProviderType enum to an EmbeddingProvider instance."""
    if provider_type == ProviderType.OLLAMA:
        return OllamaProvider()
    raise ValueError(f"Unsupported provider type: {provider_type}")