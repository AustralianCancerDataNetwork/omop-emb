"""Validation tests for EmbeddingInterface input contracts and naming guarantees."""

from unittest.mock import Mock, patch

import pytest
from oa_configurator.resolver import ResolvedModel, ResolvedProvider

from omop_emb.config import MetricType
from omop_emb.backends.index_config import FlatIndexConfig
from omop_emb.model_registry.model_registry_manager import RegistryManager


def _make_mock_backend() -> Mock:
    backend = Mock()
    backend.backend_name = "pgvector"
    backend.get_registered_model.return_value = None
    return backend


@pytest.mark.unit
class TestCanonicalModelName:
    """The storage layer receives and preserves the canonical model name verbatim."""

    def test_interface_stores_name_verbatim(self):
        from omop_emb.interface import EmbeddingWriterInterface

        backend = _make_mock_backend()

        with patch("omop_emb.interface.build_model_backend_from_resolved") as mock_build_model_backend:
            model_backend = Mock()
            model_backend.model = "pseudo-model:v1"
            model_backend.provider = "ollama"
            model_backend.dimensions.return_value = 1
            mock_build_model_backend.return_value = model_backend

            resolved = ResolvedModel(
                name="test-model",
                provider=ResolvedProvider(
                    name="test-provider", provider="ollama", base_url="http://localhost:11434", api_key=None
                ),
                model="pseudo-model",
                embedding_dim=None,
                document_prefix=None,
                query_prefix=None,
                embeddings=True,
                tool_use=False,
                structured_output=False,
                extended_thinking=False,
                configuration={},
            )
            interface = EmbeddingWriterInterface(
                backend=backend,
                metric_type=MetricType.L2,
                resolved_model=resolved,
            )

        backend.register_model = Mock(
            return_value=Mock(
                model_name="pseudo-model:v1",
                provider_type="ollama",
                storage_identifier="pgvector_pseudo_model_v1",
            )
        )

        interface.register_model(index_config=FlatIndexConfig())

        call_kwargs = backend.register_model.call_args.kwargs
        assert call_kwargs["model_name"] == "pseudo-model:v1"
        assert call_kwargs["provider_type"] == "ollama"

    def test_storage_name_reflects_tag(self):
        safe = RegistryManager.safe_model_name("pseudo-model:v1")
        assert "v1" in safe, f"Expected 'v1' in safe name, got: {safe!r}"

    def test_safe_model_name_lowercases(self):
        assert RegistryManager.safe_model_name("MyModel:V2") == "mymodel_v2"
