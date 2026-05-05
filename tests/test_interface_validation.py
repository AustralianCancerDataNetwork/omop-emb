"""Validation tests for EmbeddingInterface input contracts and naming guarantees."""

from unittest.mock import Mock, MagicMock

import numpy as np
import pytest

from omop_emb.config import IndexType, MetricType, BackendType, ProviderType
from omop_emb.embeddings import OllamaProvider
from omop_emb.backends.index_config import FlatIndexConfig
from omop_emb.model_registry.model_registry_manager import RegistryManager


def _make_mock_client(model_name: str = "test-model:v1") -> Mock:
    mock_client = Mock()
    mock_client.canonical_model_name = model_name
    mock_client.embedding_dim = 1
    mock_client.provider = Mock()
    mock_client.provider.canonical_model_name.side_effect = lambda name: name
    mock_client.provider.provider_type = ProviderType.OLLAMA
    return mock_client


def _make_mock_backend() -> Mock:
    backend = Mock()
    backend.backend_type = BackendType.PGVECTOR
    backend.get_registered_model.return_value = None
    return backend


@pytest.mark.unit
class TestCanonicalModelName:
    """The storage layer receives and preserves the canonical model name verbatim."""

    def test_interface_stores_name_verbatim(self):
        from omop_emb.interface import EmbeddingWriterInterface

        backend = _make_mock_backend()
        interface = EmbeddingWriterInterface(
            backend=backend,
            metric_type=MetricType.L2,
            embedding_client=_make_mock_client("pseudo-model:v1"),
        )

        backend.register_model = Mock(return_value=Mock(
            model_name="pseudo-model:v1",
            provider_type=ProviderType.OLLAMA,
            storage_identifier="pgvector_pseudo_model_v1",
        ))

        interface.register_model(index_config=FlatIndexConfig())

        call_kwargs = backend.register_model.call_args.kwargs
        assert call_kwargs["model_name"] == "pseudo-model:v1"
        assert call_kwargs["provider_type"] == ProviderType.OLLAMA

    def test_ollama_provider_rejects_untagged_name(self):
        provider = OllamaProvider()
        with pytest.raises(ValueError, match="must include an explicit tag"):
            provider.canonical_model_name("pseudo-model")

    def test_explicit_tag_is_not_modified(self):
        provider = OllamaProvider()
        assert provider.canonical_model_name("nomic-embed-text:v1.5") == "nomic-embed-text:v1.5"

    def test_storage_name_reflects_tag(self):
        safe = RegistryManager.safe_model_name("pseudo-model:v1")
        assert "v1" in safe, f"Expected 'v1' in safe name, got: {safe!r}"

    def test_safe_model_name_lowercases(self):
        assert RegistryManager.safe_model_name("MyModel:V2") == "mymodel_v2"

    def test_storage_name_contains_backend_and_model(self):
        name = RegistryManager.storage_name(
            safe_model_name="pseudo_model_v1",
            backend_type=BackendType.PGVECTOR,
        )
        assert "pgvector" in name
        assert "v1" in name
