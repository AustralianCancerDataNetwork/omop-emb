"""Unit tests for EmbeddingProvider implementations and the factory."""

import pytest

from omop_emb.config import ProviderType
from omop_emb.embeddings import (
    OllamaProvider,
    get_provider_for_api_base,
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


class TestGetProviderForApiBase:
    def test_ollama_in_url(self):
        assert isinstance(get_provider_for_api_base("http://ollama.internal/v1"), OllamaProvider)

    def test_localhost_with_ollama_key(self):
        assert isinstance(get_provider_for_api_base("http://localhost:11434/v1", "ollama"), OllamaProvider)

    def test_127_0_0_1_with_ollama_key(self):
        assert isinstance(get_provider_for_api_base("http://127.0.0.1:11434/v1", "ollama"), OllamaProvider)

    def test_non_ollama_url_raises(self):
        with pytest.raises(ValueError, match="Only Ollama"):
            get_provider_for_api_base("http://localhost:8000/v1", "sk-real-key")

    def test_openai_api_raises(self):
        with pytest.raises(ValueError, match="Only Ollama"):
            get_provider_for_api_base("https://api.openai.com/v1", "sk-abc")

    def test_ollama_detection_is_case_sensitive(self):
        """'Ollama' with a capital O is not detected as Ollama — raises ValueError."""
        with pytest.raises(ValueError, match="Only Ollama"):
            get_provider_for_api_base("http://Ollama.internal/v1")

    def test_arbitrary_remote_url_raises(self):
        with pytest.raises(ValueError, match="Only Ollama"):
            get_provider_for_api_base("https://embeddings.example.com/v1", "sk-x")


@pytest.mark.unit
class TestGetProviderFromProviderType:
    def test_ollama_type_returns_ollama_provider(self):
        provider = get_provider_from_provider_type(ProviderType.OLLAMA)
        assert isinstance(provider, OllamaProvider)

    def test_ollama_result_has_correct_provider_type(self):
        assert get_provider_from_provider_type(ProviderType.OLLAMA).provider_type == ProviderType.OLLAMA

    def test_each_call_returns_a_fresh_instance(self):
        """Provider instances must not be shared/cached between calls."""
        a = get_provider_from_provider_type(ProviderType.OLLAMA)
        b = get_provider_from_provider_type(ProviderType.OLLAMA)
        assert a is not b
