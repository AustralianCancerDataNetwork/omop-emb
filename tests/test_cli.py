"""Tests for CLI configuration helpers."""

from __future__ import annotations

from unittest.mock import Mock, PropertyMock

import pytest

from omop_emb.cli import (
    _normalize_api_base,
    _normalize_embedding_path,
    _resolve_embedding_dim,
    _resolve_model_name,
)
from omop_emb.interface import EmbeddingInterface


@pytest.mark.unit
class TestCliHelpers:
    def test_resolve_model_name_prefers_cli_value(self, monkeypatch):
        monkeypatch.setenv("OMOP_EMB_MODEL", "env-model")

        assert _resolve_model_name("cli-model") == "cli-model"

    def test_resolve_model_name_uses_env_value(self, monkeypatch):
        monkeypatch.setenv("OMOP_EMB_MODEL", "env-model")

        assert _resolve_model_name(None) == "env-model"

    def test_resolve_embedding_dim_prefers_cli_value(self):
        interface = Mock(spec=EmbeddingInterface)

        assert _resolve_embedding_dim(interface, 1024) == 1024

    def test_resolve_embedding_dim_uses_env_value(self, monkeypatch):
        interface = Mock(spec=EmbeddingInterface)
        monkeypatch.setenv("OMOP_EMB_EMBEDDING_DIM", "1536")

        assert _resolve_embedding_dim(interface, None) == 1536

    def test_resolve_embedding_dim_falls_back_to_client(self, monkeypatch):
        monkeypatch.delenv("OMOP_EMB_EMBEDDING_DIM", raising=False)
        interface = Mock(spec=EmbeddingInterface)
        type(interface).embedding_dim = PropertyMock(return_value=768)

        assert _resolve_embedding_dim(interface, None) == 768

    def test_resolve_embedding_dim_raises_with_guidance(self, monkeypatch):
        monkeypatch.delenv("OMOP_EMB_EMBEDDING_DIM", raising=False)
        interface = Mock(spec=EmbeddingInterface)
        type(interface).embedding_dim = PropertyMock(
            side_effect=NotImplementedError("ollama-only")
        )

        with pytest.raises(RuntimeError, match="--embedding-dim"):
            _resolve_embedding_dim(interface, None)

    def test_normalize_api_base_strips_embeddings_suffix(self):
        assert _normalize_api_base("http://localhost:8000/v1/embeddings", "/embeddings") == "http://localhost:8000/v1"

    def test_normalize_api_base_leaves_base_url_unchanged(self):
        assert _normalize_api_base("http://localhost:8000/v1", "/embeddings") == "http://localhost:8000/v1"

    def test_normalize_api_base_strips_custom_embedding_path(self):
        assert _normalize_api_base("http://localhost:8000/v1/embed", "/embed") == "http://localhost:8000/v1"

    def test_normalize_embedding_path_adds_leading_slash(self):
        assert _normalize_embedding_path("embed") == "/embed"
