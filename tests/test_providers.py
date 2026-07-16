"""Unit tests for EmbeddingProvider implementations and the factory."""

import pytest

from omop_emb.config import ProviderType
from omop_emb.embeddings import (
    OllamaProvider,
    OpenAIProvider,
    get_provider_from_provider_type,
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

    def test_raises_for_empty_name(self):
        with pytest.raises(ValueError, match="must not be empty"):
            OllamaProvider().canonical_model_name("")

    def test_raises_for_whitespace_only_name(self):
        with pytest.raises(ValueError, match="must not be empty"):
            OllamaProvider().canonical_model_name("   ")


class TestOpenAIProviderCanonicalModelName:
    """OpenAIProvider.canonical_model_name applies no tag normalisation."""

    def test_returns_name_unchanged(self):
        assert (
            OpenAIProvider().canonical_model_name("text-embedding-3-large")
            == "text-embedding-3-large"
        )

    def test_strips_whitespace(self):
        assert (
            OpenAIProvider().canonical_model_name("  text-embedding-3-large  ")
            == "text-embedding-3-large"
        )

    def test_idempotent(self):
        provider = OpenAIProvider()
        canonical = provider.canonical_model_name("text-embedding-3-large")
        assert provider.canonical_model_name(canonical) == canonical

    def test_untagged_name_is_not_rejected(self):
        """Unlike Ollama, a bare name with no ':tag' is perfectly valid."""
        assert OpenAIProvider().canonical_model_name("text-embedding-3-large")

    def test_raises_for_empty_name(self):
        with pytest.raises(ValueError, match="must not be empty"):
            OpenAIProvider().canonical_model_name("")

    def test_raises_for_whitespace_only_name(self):
        with pytest.raises(ValueError, match="must not be empty"):
            OpenAIProvider().canonical_model_name("   ")


@pytest.mark.unit
class TestOpenAIProviderGetEmbeddingDim:
    def test_returns_none(self):
        """No discovery endpoint exists; EmbeddingClient falls back to a live probe."""
        assert (
            OpenAIProvider().get_embedding_dim(
                "text-embedding-3-large", "https://api.openai.com/v1"
            )
            is None
        )


@pytest.mark.unit
class TestGetProviderFromProviderType:
    def test_ollama_type_returns_ollama_provider(self):
        provider = get_provider_from_provider_type(ProviderType.OLLAMA)
        assert isinstance(provider, OllamaProvider)

    def test_ollama_result_has_correct_provider_type(self):
        assert (
            get_provider_from_provider_type(ProviderType.OLLAMA).provider_type
            == ProviderType.OLLAMA
        )

    def test_openai_type_returns_openai_provider(self):
        provider = get_provider_from_provider_type(ProviderType.OPENAI)
        assert isinstance(provider, OpenAIProvider)

    def test_openai_result_has_correct_provider_type(self):
        assert (
            get_provider_from_provider_type(ProviderType.OPENAI).provider_type
            == ProviderType.OPENAI
        )

    def test_each_call_returns_a_fresh_instance(self):
        """Provider instances must not be shared/cached between calls."""
        a = get_provider_from_provider_type(ProviderType.OLLAMA)
        b = get_provider_from_provider_type(ProviderType.OLLAMA)
        assert a is not b
