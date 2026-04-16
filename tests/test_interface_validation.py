"""Validation tests for strict EmbeddingInterface input contracts."""

from unittest.mock import Mock

import numpy as np
import pytest

from omop_emb.config import IndexType, MetricType
from omop_emb.embeddings import OllamaProvider
from omop_emb.interface import EmbeddingInterface


@pytest.mark.unit
class TestInterfaceValidation:
    def test_get_nearest_concepts_requires_index_type(self):
        """Index type is required as part of the strict core interface."""
        interface = EmbeddingInterface(backend=Mock(), embedding_client=Mock())
        kwargs = {
            "session": Mock(),
            "canonical_model_name": "test-model:latest",
            "query_embedding": np.zeros((1, 1), dtype=np.float32),
            "metric_type": MetricType.COSINE,
        }

        with pytest.raises(TypeError):
            interface.get_nearest_concepts(**kwargs)

    def test_get_nearest_concepts_requires_metric_type(self):
        """Metric type is required as part of the strict core interface."""
        interface = EmbeddingInterface(backend=Mock(), embedding_client=Mock())
        kwargs = {
            "session": Mock(),
            "canonical_model_name": "test-model:latest",
            "index_type": IndexType.FLAT,
            "query_embedding": np.zeros((1, 1), dtype=np.float32),
        }

        with pytest.raises(TypeError):
            interface.get_nearest_concepts(**kwargs)

    def test_get_nearest_concepts_rejects_non_enum_metric_type(self):
        """Core interface rejects non-MetricType values with a clear error."""
        interface = EmbeddingInterface(backend=Mock(), embedding_client=Mock())

        with pytest.raises(TypeError, match="metric_type must be MetricType"):
            interface.get_nearest_concepts(
                session=Mock(),
                canonical_model_name="test-model:latest",
                index_type=IndexType.FLAT,
                query_embedding=np.zeros((1, 1), dtype=np.float32),
                metric_type="cosine",  # type: ignore[arg-type]
            )


@pytest.mark.unit
class TestCanonicalModelName:
    """Verify that the storage layer receives and preserves the canonical model name.

    Tag canonicalisation is the EmbeddingClient/OllamaProvider's job, not the
    interface's.  These tests confirm the contract: whatever canonical_model_name
    is passed to EmbeddingInterface methods is stored verbatim — no implicit
    :latest appending happens inside the storage layer.
    """

    def test_interface_stores_name_verbatim(self):
        """EmbeddingInterface passes canonical_model_name through unchanged."""
        mock_backend = Mock()
        mock_backend.register_model.return_value = Mock(
            model_name="pseudo-model:v1",
            storage_identifier="pgvector_pseudo_model_v1_flat",
        )
        interface = EmbeddingInterface(backend=mock_backend)

        interface.register_model(
            engine=Mock(),
            canonical_model_name="pseudo-model:v1",
            dimensions=768,
            index_type=IndexType.FLAT,
        )

        call_kwargs = mock_backend.register_model.call_args.kwargs
        assert call_kwargs["model_name"] == "pseudo-model:v1"

    def test_ollama_provider_rejects_untagged_name(self):
        """OllamaProvider raises when no tag is present.

        The interface expects a canonical model name (with tag).  The provider
        enforces this at source so the registry is never written with an
        ambiguous name.
        """
        provider = OllamaProvider()
        with pytest.raises(ValueError, match="must include an explicit tag"):
            provider.canonical_model_name("pseudo-model")

    def test_explicit_tag_is_not_modified(self):
        """An explicitly tagged name must pass through unchanged end-to-end."""
        provider = OllamaProvider()
        assert provider.canonical_model_name("nomic-embed-text:v1.5") == "nomic-embed-text:v1.5"

    def test_storage_name_reflects_tag(self):
        """The storage identifier derived from a canonical name includes the tag.

        safe_model_name('pseudo-model:v1') -> 'pseudo_model_v1'
        This ensures the tag is visible in table/directory names, making
        the exact model version traceable from the storage identifier alone.
        """
        from omop_emb.model_registry.model_registry_manager import ModelRegistryManager
        safe = ModelRegistryManager.safe_model_name("pseudo-model:v1")
        assert "v1" in safe, f"Expected 'v1' in safe name, got: {safe!r}"
