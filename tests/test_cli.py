"""Tests for CLI configuration helpers and user-facing messages."""

from __future__ import annotations

from unittest.mock import Mock, PropertyMock

import pytest
import sqlalchemy as sa

import omop_emb.cli as cli_module
from omop_emb.backends.config import BackendType, IndexType
from omop_emb.cli import (
    _build_backend_metadata,
    _build_concept_filter,
    _normalize_api_base,
    _normalize_embedding_path,
    _resolve_api_key,
    _resolve_embedding_dim,
    add_embeddings,
    _resolve_model_name,
    switch_index_type,
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

    def test_resolve_api_key_uses_cli_value(self, monkeypatch):
        monkeypatch.setenv("OMOP_EMB_API_KEY", "env-key")

        assert _resolve_api_key("cli-key") == "cli-key"

    def test_resolve_api_key_uses_env_value(self, monkeypatch):
        monkeypatch.setenv("OMOP_EMB_API_KEY", "env-key")

        assert _resolve_api_key(None) == "env-key"

    def test_resolve_api_key_allows_missing_value(self, monkeypatch):
        monkeypatch.delenv("OMOP_EMB_API_KEY", raising=False)

        assert _resolve_api_key(None) is None

    def test_normalize_api_base_strips_embeddings_suffix(self):
        assert _normalize_api_base("http://localhost:8000/v1/embeddings", "/embeddings") == "http://localhost:8000/v1"

    def test_normalize_api_base_leaves_base_url_unchanged(self):
        assert _normalize_api_base("http://localhost:8000/v1", "/embeddings") == "http://localhost:8000/v1"

    def test_normalize_api_base_strips_custom_embedding_path(self):
        assert _normalize_api_base("http://localhost:8000/v1/embed", "/embed") == "http://localhost:8000/v1"

    def test_normalize_embedding_path_adds_leading_slash(self):
        assert _normalize_embedding_path("embed") == "/embed"

    def test_build_concept_filter_returns_none_when_unfiltered(self):
        assert _build_concept_filter(standard_only=False, vocabularies=None) is None

    def test_build_concept_filter_returns_filter_when_constrained(self):
        concept_filter = _build_concept_filter(
            standard_only=True,
            vocabularies=["SNOMED"],
        )

        assert concept_filter is not None
        assert concept_filter.require_standard is True
        assert concept_filter.vocabularies == ("SNOMED",)

    def test_build_backend_metadata_includes_hnsw_settings_for_faiss(self):
        metadata = _build_backend_metadata(
            backend_type=BackendType.FAISS,
            index_type=IndexType.HNSW,
            hnsw_num_neighbors=48,
            hnsw_ef_search=96,
            hnsw_ef_construction=240,
        )

        assert metadata["hnsw_num_neighbors"] == 48
        assert metadata["hnsw_ef_search"] == 96
        assert metadata["hnsw_ef_construction"] == 240

    def test_build_backend_metadata_strips_hnsw_settings_for_flat(self):
        metadata = _build_backend_metadata(
            backend_type=BackendType.FAISS,
            index_type=IndexType.FLAT,
            existing_metadata={
                "hnsw_num_neighbors": 48,
                "hnsw_ef_search": 96,
                "custom": "keep-me",
            },
        )

        assert "hnsw_num_neighbors" not in metadata
        assert "hnsw_ef_search" not in metadata
        assert metadata["custom"] == "keep-me"

    def test_add_embeddings_prints_faiss_rebuild_note(self, monkeypatch, capsys):
        class FakeResult:
            def partitions(self, batch_size):
                return iter(())

        class FakeReaderSession:
            bind = object()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def scalar(self, query):
                return 0

            def execute(self, query):
                return FakeResult()

        class FakeWriterSession:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

        class FakeInterface:
            def __init__(self):
                self.backend = Mock()
                self.backend.backend_type = BackendType.FAISS

            def initialise_store(self, engine):
                return None

            def ensure_model_registered(self, **kwargs):
                return None

            def get_concepts_without_embedding_count(self, **kwargs):
                return 0

            def get_concepts_without_embedding_query(self, **kwargs):
                return (
                    sa.select(
                        sa.literal(0).label("concept_id"),
                        sa.literal("").label("concept_name"),
                    )
                    .where(sa.false())
                )

            def embed_and_upsert_concepts(self, **kwargs):
                return None

        fake_interface = FakeInterface()
        fake_engine = object()
        fake_sessions = iter((FakeReaderSession(), FakeWriterSession()))

        monkeypatch.setattr(cli_module, "_create_engine_from_env", lambda: fake_engine)
        monkeypatch.setattr(cli_module, "get_metadata_schema", lambda: "public")
        monkeypatch.setattr(
            cli_module,
            "_create_embedding_interface",
            lambda **kwargs: (fake_interface, "tei-qwen:intfloat/multilingual-e5-large-instruct"),
        )
        monkeypatch.setattr(cli_module, "_resolve_embedding_dim", lambda interface, embedding_dim: 1024)
        monkeypatch.setattr(cli_module, "Session", lambda engine: next(fake_sessions))

        add_embeddings(
            api_base="http://localhost:14000/v1",
            api_key=None,
            index_type=IndexType.FLAT,
            batch_size=32,
            model="tei-qwen:intfloat/multilingual-e5-large-instruct",
            embedding_dim=1024,
            embedding_path="/embeddings",
            overwrite_model_registration=False,
            backend_name="faiss",
            faiss_base_dir="/tmp/faiss",
            standard_only=False,
            vocabularies=None,
            num_embeddings=0,
        )

        stdout = capsys.readouterr().out
        assert "Rebuild the metric-specific FAISS index before search" in stdout
        assert "omop-emb rebuild-index --model tei-qwen:intfloat/multilingual-e5-large-instruct" in stdout

    def test_switch_index_type_updates_faiss_model_and_rebuilds(self, monkeypatch, capsys):
        class FakeSession:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

        fake_backend = Mock(spec=cli_module.FaissEmbeddingBackend)
        fake_backend.backend_type = BackendType.FAISS
        fake_backend.get_registered_model.return_value = Mock(
            metadata={"custom": "keep-me"},
        )

        fake_interface = Mock(spec=EmbeddingInterface)
        fake_interface.backend = fake_backend

        monkeypatch.setattr(cli_module, "_create_engine_from_env", lambda: object())
        monkeypatch.setattr(cli_module, "get_metadata_schema", lambda: "public")
        monkeypatch.setattr(
            cli_module.EmbeddingInterface,
            "from_backend_name",
            classmethod(lambda cls, **kwargs: fake_interface),
        )
        monkeypatch.setattr(cli_module, "Session", lambda engine: FakeSession())

        switch_index_type(
            model="tei-qwen:intfloat/multilingual-e5-large-instruct",
            backend_name="faiss",
            faiss_base_dir="/tmp/faiss",
            index_type=IndexType.HNSW,
            metric_types=None,
            hnsw_num_neighbors=48,
            hnsw_ef_search=96,
            hnsw_ef_construction=240,
            batch_size=1000,
            rebuild=True,
        )

        fake_backend.update_model_index_configuration.assert_called_once()
        update_kwargs = fake_backend.update_model_index_configuration.call_args.kwargs
        assert update_kwargs["index_type"] == IndexType.HNSW
        assert update_kwargs["metadata"]["hnsw_num_neighbors"] == 48
        assert update_kwargs["metadata"]["hnsw_ef_search"] == 96
        assert update_kwargs["metadata"]["hnsw_ef_construction"] == 240
        fake_interface.rebuild_model_indexes.assert_called_once()
        stdout = capsys.readouterr().out
        assert "Updated model 'tei-qwen:intfloat/multilingual-e5-large-instruct' to index_type=hnsw." in stdout
