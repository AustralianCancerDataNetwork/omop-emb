"""Unit tests for embedding generation.

Covers ``EmbeddingRole`` prefixing, ``EmbeddingReaderInterface.generate_embeddings()``, and
``EmbeddingWriterInterface``'s non-CDM-dependent behavior (construction,
``embedding_dim`` caching, ``embed_texts``). Model calling itself
(construction, canonicalization, dimension discovery, batching) is
``omop_llm``'s own tested responsibility: ``ModelBackend`` is mocked here.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from omop_emb.config import MetricType, OmopEmbConfig
from omop_emb.interface import (
    EmbeddingRole,
    EmbeddingReaderInterface,
    EmbeddingWriterInterface,
)

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = "nomic-embed-text:v1.5"


def _make_emb_config(doc_prefix: str = "", query_prefix: str = "") -> OmopEmbConfig:
    return OmopEmbConfig(
        document_embedding_prefix=doc_prefix,
        query_embedding_prefix=query_prefix,
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
# EmbeddingReaderInterface.generate_embeddings(): shape validation, prefixing, batching passthrough
# ---------------------------------------------------------------------------

_EMPTY_PREFIXES = {EmbeddingRole.DOCUMENT: "", EmbeddingRole.QUERY: ""}


@pytest.mark.unit
class TestGenerateEmbeddings:
    def test_single_string_returns_2d_array(self):
        backend = _mock_model_backend(vectors=[[0.1, 0.2, 0.3]])
        result = EmbeddingReaderInterface.generate_embeddings(
            backend, "hello", embedding_role=EmbeddingRole.DOCUMENT, prefixes=_EMPTY_PREFIXES
        )
        assert result.shape == (1, 3)

    def test_list_input_returns_correct_shape(self):
        backend = _mock_model_backend(vectors=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        result = EmbeddingReaderInterface.generate_embeddings(
            backend, ["a", "b", "c"], embedding_role=EmbeddingRole.DOCUMENT, prefixes=_EMPTY_PREFIXES
        )
        assert result.shape == (3, 2)

    def test_values_preserved_in_output(self):
        backend = _mock_model_backend(vectors=[[1.0, 2.0]])
        result = EmbeddingReaderInterface.generate_embeddings(
            backend, "text", embedding_role=EmbeddingRole.DOCUMENT, prefixes=_EMPTY_PREFIXES
        )
        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0])

    def test_batch_size_forwarded_to_backend(self):
        backend = _mock_model_backend(vectors=[[1.0]])
        EmbeddingReaderInterface.generate_embeddings(
            backend, ["a"], embedding_role=EmbeddingRole.DOCUMENT, prefixes=_EMPTY_PREFIXES, batch_size=8
        )
        backend.embed_texts.assert_called_once_with(["a"], batch_size=8)

    def test_prefix_applied_before_embedding(self):
        backend = _mock_model_backend(vectors=[[0.1]])
        EmbeddingReaderInterface.generate_embeddings(
            backend,
            "diabetes",
            embedding_role=EmbeddingRole.DOCUMENT,
            prefixes={EmbeddingRole.DOCUMENT: "passage: ", EmbeddingRole.QUERY: ""},
        )
        backend.embed_texts.assert_called_once_with(["passage: diabetes"], batch_size=None)

    def test_no_prefix_passes_text_verbatim(self):
        backend = _mock_model_backend(vectors=[[0.1]])
        EmbeddingReaderInterface.generate_embeddings(
            backend, "diabetes", embedding_role=EmbeddingRole.DOCUMENT, prefixes=_EMPTY_PREFIXES
        )
        backend.embed_texts.assert_called_once_with(["diabetes"], batch_size=None)

    def test_prefixes_loaded_from_config_when_omitted(self, monkeypatch):
        monkeypatch.setattr(
            OmopEmbConfig, "get_config", lambda: _make_emb_config(doc_prefix="passage: ")
        )
        backend = _mock_model_backend(vectors=[[0.1]])
        EmbeddingReaderInterface.generate_embeddings(backend, "diabetes", embedding_role=EmbeddingRole.DOCUMENT)
        backend.embed_texts.assert_called_once_with(["passage: diabetes"], batch_size=None)

    def test_raises_on_non_2d_result(self):
        backend = _mock_model_backend(vectors=[0.1, 0.2, 0.3])  # ty: ignore[invalid-argument-type] - purposefully wrong shape
        with pytest.raises(RuntimeError, match="2-D"):
            EmbeddingReaderInterface.generate_embeddings(
                backend, "hello", embedding_role=EmbeddingRole.DOCUMENT, prefixes=_EMPTY_PREFIXES
            )

    def test_raises_on_row_count_mismatch(self):
        backend = _mock_model_backend(vectors=[[0.1, 0.2]])  # 1 row for 2 texts
        with pytest.raises(RuntimeError, match="Expected 2 embeddings"):
            EmbeddingReaderInterface.generate_embeddings(
                backend, ["a", "b"], embedding_role=EmbeddingRole.DOCUMENT, prefixes=_EMPTY_PREFIXES
            )

    def test_raises_on_partial_prefixes_missing_query(self):
        backend = _mock_model_backend(vectors=[[0.1]])
        with pytest.raises(ValueError, match="QUERY"):
            EmbeddingReaderInterface.generate_embeddings(
                backend,
                "diabetes",
                embedding_role=EmbeddingRole.DOCUMENT,
                prefixes={EmbeddingRole.DOCUMENT: "passage: "},
            )

    def test_raises_on_partial_prefixes_missing_document(self):
        backend = _mock_model_backend(vectors=[[0.1]])
        with pytest.raises(ValueError, match="DOCUMENT"):
            EmbeddingReaderInterface.generate_embeddings(
                backend,
                "diabetes",
                embedding_role=EmbeddingRole.QUERY,
                prefixes={EmbeddingRole.QUERY: "query: "},
            )

    def test_raises_on_empty_dict_prefixes(self):
        """An empty dict is a partial mapping too: both roles are missing."""
        backend = _mock_model_backend(vectors=[[0.1]])
        with pytest.raises(ValueError, match="DOCUMENT"):
            EmbeddingReaderInterface.generate_embeddings(
                backend, "diabetes", embedding_role=EmbeddingRole.DOCUMENT, prefixes={}
            )

    def test_backend_not_called_when_prefixes_invalid(self):
        backend = _mock_model_backend(vectors=[[0.1]])
        with pytest.raises(ValueError):
            EmbeddingReaderInterface.generate_embeddings(
                backend,
                "diabetes",
                embedding_role=EmbeddingRole.DOCUMENT,
                prefixes={EmbeddingRole.DOCUMENT: "passage: "},
            )
        backend.embed_texts.assert_not_called()


# ---------------------------------------------------------------------------
# EmbeddingReaderInterface.load_embedding_prefixes()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadEmbeddingPrefixes:
    def test_returns_empty_strings_when_not_configured(self, monkeypatch):
        monkeypatch.setattr(OmopEmbConfig, "get_config", lambda: _make_emb_config())
        prefixes = EmbeddingReaderInterface.load_embedding_prefixes()
        assert prefixes[EmbeddingRole.DOCUMENT] == ""
        assert prefixes[EmbeddingRole.QUERY] == ""

    def test_returns_configured_prefixes(self, monkeypatch):
        monkeypatch.setattr(
            OmopEmbConfig,
            "get_config",
            lambda: _make_emb_config("search_document: ", "search_query: "),
        )
        prefixes = EmbeddingReaderInterface.load_embedding_prefixes()
        assert prefixes[EmbeddingRole.DOCUMENT] == "search_document: "
        assert prefixes[EmbeddingRole.QUERY] == "search_query: "

    def test_logs_warning_when_prefix_not_set(self, monkeypatch, caplog):
        monkeypatch.setattr(OmopEmbConfig, "get_config", lambda: _make_emb_config())
        import logging

        with caplog.at_level(logging.WARNING, logger="omop_emb.interface"):
            EmbeddingReaderInterface.load_embedding_prefixes()
        warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("omop-config configure omop_emb" in m for m in warning_messages)

    def test_complete_override_returned_as_is(self):
        override = {EmbeddingRole.DOCUMENT: "passage: ", EmbeddingRole.QUERY: "query: "}
        assert EmbeddingReaderInterface.load_embedding_prefixes(override) == override

    def test_override_bypasses_config(self, monkeypatch):
        monkeypatch.setattr(
            OmopEmbConfig, "get_config", lambda: _make_emb_config("from_config: ", "from_config: ")
        )
        override = _EMPTY_PREFIXES
        assert EmbeddingReaderInterface.load_embedding_prefixes(override) == override

    def test_raises_on_partial_override(self):
        with pytest.raises(ValueError, match="QUERY"):
            EmbeddingReaderInterface.load_embedding_prefixes({EmbeddingRole.DOCUMENT: "passage: "})

    def test_raises_on_empty_dict_override(self):
        with pytest.raises(ValueError, match="DOCUMENT"):
            EmbeddingReaderInterface.load_embedding_prefixes({})


# ---------------------------------------------------------------------------
# EmbeddingWriterInterface: construction builds the ModelBackend
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbeddingWriterInterfaceConstruction:
    def test_model_backend_built_with_provider_model_and_base_url(self, monkeypatch):
        monkeypatch.setattr(OmopEmbConfig, "get_config", lambda: _make_emb_config())
        with patch("omop_emb.interface.build_model_backend") as mock_build_model_backend:
            mock_build_model_backend.return_value = _mock_model_backend(dim=768)

            EmbeddingWriterInterface(
                backend=_mock_storage_backend(),
                metric_type=MetricType.COSINE,
                model="nomic-embed-text",
                provider_type="ollama",
                api_base=OLLAMA_BASE,
                api_key="ollama",
            )

        mock_build_model_backend.assert_called_once_with(
            "ollama", "nomic-embed-text", base_url=OLLAMA_BASE, api_key="ollama", configuration=None
        )

    def test_config_embedding_dim_passed_as_configuration_override(self, monkeypatch):
        monkeypatch.setattr(
            OmopEmbConfig, "get_config", lambda: OmopEmbConfig(embedding_dim=512)
        )
        with patch("omop_emb.interface.build_model_backend") as mock_build_model_backend:
            mock_build_model_backend.return_value = _mock_model_backend(dim=512)
            EmbeddingWriterInterface(
                backend=_mock_storage_backend(),
                metric_type=MetricType.COSINE,
                model=OLLAMA_MODEL,
                provider_type="ollama",
                api_base=OLLAMA_BASE,
            )
        mock_build_model_backend.assert_called_once_with(
            "ollama", OLLAMA_MODEL, base_url=OLLAMA_BASE, api_key="ollama",
            configuration={"embedding_dim": 512},
        )

    def test_canonical_model_name_read_from_backend(self, monkeypatch):
        monkeypatch.setattr(OmopEmbConfig, "get_config", lambda: _make_emb_config())
        with patch("omop_emb.interface.build_model_backend") as mock_build_model_backend:
            mock_build_model_backend.return_value = _mock_model_backend(model="nomic-embed-text:v1.5")
            iface = EmbeddingWriterInterface(
                backend=_mock_storage_backend(),
                metric_type=MetricType.COSINE,
                model="nomic-embed-text",
                provider_type="ollama",
                api_base=OLLAMA_BASE,
            )
        assert iface.canonical_model_name == "nomic-embed-text:v1.5"

    def test_provider_type_read_from_backend(self, monkeypatch):
        monkeypatch.setattr(OmopEmbConfig, "get_config", lambda: _make_emb_config())
        with patch("omop_emb.interface.build_model_backend") as mock_build_model_backend:
            mock_build_model_backend.return_value = _mock_model_backend(provider="anthropic")
            iface = EmbeddingWriterInterface(
                backend=_mock_storage_backend(),
                metric_type=MetricType.COSINE,
                model=OLLAMA_MODEL,
                provider_type="anthropic",
                api_base=OLLAMA_BASE,
            )
        assert iface.provider_type == "anthropic"


# ---------------------------------------------------------------------------
# EmbeddingWriterInterface.embedding_dim: lazy resolution, cached
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbeddingDimCaching:
    def test_dimensions_resolved_and_cached(self, monkeypatch):
        monkeypatch.setattr(OmopEmbConfig, "get_config", lambda: _make_emb_config())
        with patch("omop_emb.interface.build_model_backend") as mock_build_model_backend:
            model_backend = _mock_model_backend(dim=768)
            mock_build_model_backend.return_value = model_backend
            iface = EmbeddingWriterInterface(
                backend=_mock_storage_backend(),
                metric_type=MetricType.COSINE,
                model=OLLAMA_MODEL,
                provider_type="ollama",
                api_base=OLLAMA_BASE,
            )
        assert iface.embedding_dim == 768
        assert iface.embedding_dim == 768
        model_backend.dimensions.assert_called_once()


# ---------------------------------------------------------------------------
# EmbeddingWriterInterface.embed_texts(): default vs. per-call batch size
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbeddingWriterInterfaceEmbedTexts:
    def _make_interface(self, monkeypatch, model_backend: Mock) -> EmbeddingWriterInterface:
        monkeypatch.setattr(OmopEmbConfig, "get_config", lambda: _make_emb_config())
        with patch("omop_emb.interface.build_model_backend") as mock_build_model_backend:
            mock_build_model_backend.return_value = model_backend
            return EmbeddingWriterInterface(
                backend=_mock_storage_backend(),
                metric_type=MetricType.COSINE,
                model=OLLAMA_MODEL,
                provider_type="ollama",
                api_base=OLLAMA_BASE,
                embedding_batch_size=16,
            )

    def test_uses_default_batch_size(self, monkeypatch):
        model_backend = _mock_model_backend(vectors=[[1.0]])
        iface = self._make_interface(monkeypatch, model_backend)
        iface.embed_texts(["a"], embedding_role=EmbeddingRole.DOCUMENT)
        model_backend.embed_texts.assert_called_once_with(["a"], batch_size=16)

    def test_batch_size_override(self, monkeypatch):
        model_backend = _mock_model_backend(vectors=[[1.0]])
        iface = self._make_interface(monkeypatch, model_backend)
        iface.embed_texts(["a"], embedding_role=EmbeddingRole.DOCUMENT, batch_size=4)
        model_backend.embed_texts.assert_called_once_with(["a"], batch_size=4)
