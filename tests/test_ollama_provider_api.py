"""Unit tests for OllamaProvider.get_embedding_dim() HTTP interaction.

All tests mock requests.post so no real Ollama instance is needed.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from httpx import URL

from omop_emb.embeddings import OllamaProvider

OLLAMA_V1_BASE = URL("http://localhost:11434/v1")


def _mock_post(json_data: dict) -> Mock:
    response = Mock()
    response.json.return_value = json_data
    mock = Mock(return_value=response)
    return mock


def _patch_post(json_data: dict):
    return patch(
        "omop_emb.embeddings.embedding_providers.requests.post",
        _mock_post(json_data),
    )


@pytest.mark.unit
class TestOllamaProviderGetEmbeddingDim:
    def test_happy_path_returns_correct_dimension(self):
        json_resp = {"model_info": {"llama.embedding_length": 768}}
        with _patch_post(json_resp):
            dim = OllamaProvider().get_embedding_dim("nomic-embed-text:v1.5", OLLAMA_V1_BASE)
        assert dim == 768

    def test_returns_int_not_float(self):
        json_resp = {"model_info": {"llama.embedding_length": 384.0}}
        with _patch_post(json_resp):
            dim = OllamaProvider().get_embedding_dim("model:tag", OLLAMA_V1_BASE)
        assert isinstance(dim, int)

    def test_strips_v1_suffix_from_url_before_calling_api_show(self):
        """POST /api/show is on the Ollama base URL, not /v1."""
        with patch("omop_emb.embeddings.embedding_providers.requests.post") as mock_post:
            mock_post.return_value = Mock(
                json=Mock(return_value={"model_info": {"x.embedding_length": 512}})
            )
            OllamaProvider().get_embedding_dim("model:tag", OLLAMA_V1_BASE)
        called_url: str = mock_post.call_args[0][0]
        assert "/v1" not in called_url
        assert called_url.endswith("/api/show")

    def test_model_name_sent_in_request_body(self):
        with patch("omop_emb.embeddings.embedding_providers.requests.post") as mock_post:
            mock_post.return_value = Mock(
                json=Mock(return_value={"model_info": {"x.embedding_length": 128}})
            )
            OllamaProvider().get_embedding_dim("nomic-embed-text:v1.5", OLLAMA_V1_BASE)
        assert mock_post.call_args[1]["json"] == {"name": "nomic-embed-text:v1.5"}

    def test_raises_when_model_info_absent(self):
        with _patch_post({}):
            with pytest.raises(ValueError, match="Could not determine embedding dimension"):
                OllamaProvider().get_embedding_dim("model:tag", OLLAMA_V1_BASE)

    def test_raises_when_no_embedding_length_key(self):
        json_resp = {"model_info": {"llama.context_length": 4096, "llama.head_count": 32}}
        with _patch_post(json_resp):
            with pytest.raises(ValueError, match="Could not determine embedding dimension"):
                OllamaProvider().get_embedding_dim("model:tag", OLLAMA_V1_BASE)

    def test_raises_when_multiple_embedding_length_keys(self):
        """Ambiguous response with more than one embedding_length key must fail clearly."""
        json_resp = {
            "model_info": {
                "llama.embedding_length": 768,
                "other.embedding_length": 512,
            }
        }
        with _patch_post(json_resp):
            with pytest.raises(ValueError):
                OllamaProvider().get_embedding_dim("model:tag", OLLAMA_V1_BASE)

    def test_error_message_includes_model_name(self):
        with _patch_post({}):
            with pytest.raises(ValueError, match="bad-model:v1"):
                OllamaProvider().get_embedding_dim("bad-model:v1", OLLAMA_V1_BASE)

    def test_error_message_includes_api_response(self):
        """Surfacing the raw response helps the caller diagnose unexpected Ollama replies."""
        response_body = {"unexpected": "payload"}
        with _patch_post(response_body):
            with pytest.raises(ValueError, match="unexpected"):
                OllamaProvider().get_embedding_dim("model:tag", OLLAMA_V1_BASE)

    def test_works_without_v1_in_base_url(self):
        """A bare Ollama URL (no /v1 suffix) should still hit /api/show correctly."""
        bare_base = URL("http://localhost:11434")
        with patch("omop_emb.embeddings.embedding_providers.requests.post") as mock_post:
            mock_post.return_value = Mock(
                json=Mock(return_value={"model_info": {"llama.embedding_length": 256}})
            )
            dim = OllamaProvider().get_embedding_dim("model:tag", bare_base)
        called_url: str = mock_post.call_args[0][0]
        assert called_url.endswith("/api/show")
        assert dim == 256
