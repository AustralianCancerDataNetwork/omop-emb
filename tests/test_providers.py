"""Unit tests for EmbeddingProvider implementations and the factory."""

import pytest

from omop_emb.embeddings import (
    OllamaProvider,
    OpenAICompatProvider,
    get_provider_for_api_base,
)


class TestOllamaProviderCanonicalModelName:
    """OllamaProvider.canonical_model_name enforces an explicit tag."""

    def test_raises_for_untagged_name(self):
        """Untagged names must be rejected — :latest is mutable and unsafe."""
        with pytest.raises(ValueError, match="must include an explicit tag"):
            OllamaProvider().canonical_model_name("pseudo-model")

    def test_error_message_names_the_model(self):
        """The error names the offending model so the caller knows what to fix."""
        with pytest.raises(ValueError, match="pseudo-model"):
            OllamaProvider().canonical_model_name("pseudo-model")

    def test_error_message_explains_mutability(self):
        """The error explains *why*."""
        with pytest.raises(ValueError, match="mutable"):
            OllamaProvider().canonical_model_name("pseudo-model")

    def test_preserves_explicit_tag(self):
        assert OllamaProvider().canonical_model_name("llama3:8b") == "llama3:8b"

    def test_rejects_latest_tag(self):
        """Even explicit :latest is rejected — it is mutable and unsafe."""
        with pytest.raises(ValueError, match="mutable"):
            OllamaProvider().canonical_model_name("llama3:latest")

    def test_idempotent(self):
        """Calling twice on an already-tagged name must return the same string."""
        provider = OllamaProvider()
        canonical = provider.canonical_model_name("llama3:8b")
        assert provider.canonical_model_name(canonical) == canonical

    def test_strips_whitespace_before_validation(self):
        """Whitespace is stripped before the tag check — not a bypass."""
        with pytest.raises(ValueError):
            OllamaProvider().canonical_model_name("  pseudo-model  ")

    def test_strips_whitespace_with_explicit_tag(self):
        assert OllamaProvider().canonical_model_name("  llama3:8b  ") == "llama3:8b"


class TestOpenAICompatProviderCanonicalModelName:
    def test_returns_name_unchanged(self):
        assert OpenAICompatProvider().canonical_model_name("text-embedding-3-small") == "text-embedding-3-small"

    def test_strips_whitespace(self):
        assert OpenAICompatProvider().canonical_model_name("  text-embedding-3-small  ") == "text-embedding-3-small"

    def test_get_embedding_dim_raises(self):
        with pytest.raises(NotImplementedError):
            OpenAICompatProvider().get_embedding_dim("text-embedding-3-small", "https://api.openai.com/v1")


class TestGetProviderForApiBase:
    def test_ollama_in_url(self):
        assert isinstance(get_provider_for_api_base("http://ollama.internal/v1"), OllamaProvider)

    def test_localhost_with_ollama_key(self):
        assert isinstance(get_provider_for_api_base("http://localhost:11434/v1", "ollama"), OllamaProvider)

    def test_localhost_with_real_key_is_openai_compat(self):
        assert isinstance(get_provider_for_api_base("http://localhost:8000/v1", "sk-real-key"), OpenAICompatProvider)

    def test_openai_api(self):
        assert isinstance(get_provider_for_api_base("https://api.openai.com/v1", "sk-abc"), OpenAICompatProvider)

    def test_127_0_0_1_with_ollama_key(self):
        assert isinstance(get_provider_for_api_base("http://127.0.0.1:11434/v1", "ollama"), OllamaProvider)
