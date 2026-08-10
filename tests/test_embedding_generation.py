"""Unit tests for embedding generation.

Covers ``EmbeddingReaderInterface.generate_embeddings()`` and
``EmbeddingWriterInterface``'s non-CDM-dependent behavior (construction,
``embedding_dim`` caching, ``embed_texts``). Model calling itself
(construction, canonicalization, dimension discovery, batching, and
role-prefix application) is ``omop_llm``'s own tested responsibility:
``ModelBackend`` is mocked here.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest
from oa_configurator.resolver import ResolvedModel, ResolvedProvider

from omop_emb.config import MetricType
from omop_emb.interface import (
    EmbeddingRole,
    EmbeddingReaderInterface,
    EmbeddingWriterInterface,
)

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = "nomic-embed-text:v1.5"


def _make_resolved_model(
    *,
    provider: str = "ollama",
    model: str = OLLAMA_MODEL,
    base_url: str | None = OLLAMA_BASE,
    api_key: str | None = "ollama",
    embedding_dim: int | None = None,
    document_prefix: str | None = None,
    query_prefix: str | None = None,
    configuration: dict | None = None,
) -> ResolvedModel:
    return ResolvedModel(
        name="test-model",
        provider=ResolvedProvider(name="test-provider", provider=provider, base_url=base_url, api_key=api_key),
        model=model,
        embedding_dim=embedding_dim,
        document_prefix=document_prefix,
        query_prefix=query_prefix,
        embeddings=True,
        tool_use=False,
        structured_output=False,
        extended_thinking=False,
        configuration=configuration or {},
    )


def _mock_model_backend(
    vectors: list[list[float]] | None = None,
    dim: int | None = None,
    model: str = OLLAMA_MODEL,
    provider: str = "ollama",
) -> Mock:
    backend = Mock()
    backend.model = model
    backend.provider = provider
    if vectors is not None:
        backend.embed_texts.return_value = vectors
    if dim is not None:
        backend.dimensions.return_value = dim
    return backend


def _mock_storage_backend() -> Mock:
    backend = Mock()
    backend.backend_name = "pgvector"
    backend.get_registered_model.return_value = None
    return backend


# ---------------------------------------------------------------------------
# EmbeddingReaderInterface.generate_embeddings(): shape validation, role/batch_size passthrough
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateEmbeddings:
    def test_single_string_returns_2d_array(self):
        backend = _mock_model_backend(vectors=[[0.1, 0.2, 0.3]])
        result = EmbeddingReaderInterface.generate_embeddings(
            backend, "hello", role=EmbeddingRole.DOCUMENT
        )
        assert result.shape == (1, 3)

    def test_list_input_returns_correct_shape(self):
        backend = _mock_model_backend(vectors=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        result = EmbeddingReaderInterface.generate_embeddings(
            backend, ["a", "b", "c"], role=EmbeddingRole.DOCUMENT
        )
        assert result.shape == (3, 2)

    def test_values_preserved_in_output(self):
        backend = _mock_model_backend(vectors=[[1.0, 2.0]])
        result = EmbeddingReaderInterface.generate_embeddings(
            backend, "text", role=EmbeddingRole.DOCUMENT
        )
        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0])

    def test_batch_size_forwarded_to_backend(self):
        backend = _mock_model_backend(vectors=[[1.0]])
        EmbeddingReaderInterface.generate_embeddings(
            backend, ["a"], role=EmbeddingRole.DOCUMENT, batch_size=8
        )
        backend.embed_texts.assert_called_once_with(["a"], role=EmbeddingRole.DOCUMENT, batch_size=8)

    def test_role_forwarded_to_backend(self):
        backend = _mock_model_backend(vectors=[[0.1]])
        EmbeddingReaderInterface.generate_embeddings(backend, "diabetes", role=EmbeddingRole.QUERY)
        backend.embed_texts.assert_called_once_with(["diabetes"], role=EmbeddingRole.QUERY, batch_size=None)

    def test_raises_on_non_2d_result(self):
        backend = _mock_model_backend(vectors=[0.1, 0.2, 0.3])  # ty: ignore[invalid-argument-type] - purposefully wrong shape
        with pytest.raises(RuntimeError, match="2-D"):
            EmbeddingReaderInterface.generate_embeddings(backend, "hello", role=EmbeddingRole.DOCUMENT)

    def test_raises_on_row_count_mismatch(self):
        backend = _mock_model_backend(vectors=[[0.1, 0.2]])  # 1 row for 2 texts
        with pytest.raises(RuntimeError, match="Expected 2 embeddings"):
            EmbeddingReaderInterface.generate_embeddings(backend, ["a", "b"], role=EmbeddingRole.DOCUMENT)


# ---------------------------------------------------------------------------
# EmbeddingWriterInterface: construction builds the ModelBackend from a ResolvedModel
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbeddingWriterInterfaceConstruction:
    def test_model_backend_built_from_resolved_model(self):
        resolved = _make_resolved_model(model="nomic-embed-text")
        with patch("omop_emb.interface.build_model_backend_from_resolved") as mock_build:
            mock_build.return_value = _mock_model_backend(dim=768)

            EmbeddingWriterInterface(
                backend=_mock_storage_backend(),
                metric_type=MetricType.COSINE,
                resolved_model=resolved,
            )

        mock_build.assert_called_once_with(resolved)

    def test_canonical_model_name_read_from_backend(self):
        resolved = _make_resolved_model()
        with patch("omop_emb.interface.build_model_backend_from_resolved") as mock_build:
            mock_build.return_value = _mock_model_backend(model="nomic-embed-text:v1.5")
            iface = EmbeddingWriterInterface(
                backend=_mock_storage_backend(),
                metric_type=MetricType.COSINE,
                resolved_model=resolved,
            )
        assert iface.canonical_model_name == "nomic-embed-text:v1.5"

    def test_provider_type_read_from_backend(self):
        resolved = _make_resolved_model(provider="anthropic")
        with patch("omop_emb.interface.build_model_backend_from_resolved") as mock_build:
            mock_build.return_value = _mock_model_backend(provider="anthropic")
            iface = EmbeddingWriterInterface(
                backend=_mock_storage_backend(),
                metric_type=MetricType.COSINE,
                resolved_model=resolved,
            )
        assert iface.provider_type == "anthropic"


# ---------------------------------------------------------------------------
# EmbeddingWriterInterface.embedding_dim: lazy resolution, cached
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbeddingDimCaching:
    def test_dimensions_resolved_and_cached(self):
        resolved = _make_resolved_model()
        with patch("omop_emb.interface.build_model_backend_from_resolved") as mock_build:
            model_backend = _mock_model_backend(dim=768)
            mock_build.return_value = model_backend
            iface = EmbeddingWriterInterface(
                backend=_mock_storage_backend(),
                metric_type=MetricType.COSINE,
                resolved_model=resolved,
            )
        assert iface.embedding_dim == 768
        assert iface.embedding_dim == 768
        model_backend.dimensions.assert_called_once()


# ---------------------------------------------------------------------------
# EmbeddingWriterInterface.embed_texts(): default vs. per-call batch size
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbeddingWriterInterfaceEmbedTexts:
    def _make_interface(self, model_backend: Mock) -> EmbeddingWriterInterface:
        resolved = _make_resolved_model()
        with patch("omop_emb.interface.build_model_backend_from_resolved") as mock_build:
            mock_build.return_value = model_backend
            return EmbeddingWriterInterface(
                backend=_mock_storage_backend(),
                metric_type=MetricType.COSINE,
                resolved_model=resolved,
                embedding_batch_size=16,
            )

    def test_uses_default_batch_size(self):
        model_backend = _mock_model_backend(vectors=[[1.0]])
        iface = self._make_interface(model_backend)
        iface.embed_texts(["a"], role=EmbeddingRole.DOCUMENT)
        model_backend.embed_texts.assert_called_once_with(["a"], role=EmbeddingRole.DOCUMENT, batch_size=16)

    def test_batch_size_override(self):
        model_backend = _mock_model_backend(vectors=[[1.0]])
        iface = self._make_interface(model_backend)
        iface.embed_texts(["a"], role=EmbeddingRole.DOCUMENT, batch_size=4)
        model_backend.embed_texts.assert_called_once_with(["a"], role=EmbeddingRole.DOCUMENT, batch_size=4)
