"""Unit tests for EmbeddingClient.

All tests mock the OpenAI HTTP client so no real network calls are made.
Embedding vectors are controlled deterministically via return_value / side_effect.
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from omop_emb.config import ProviderType
from omop_emb.embeddings import EmbeddingClient, EmbeddingRole, OllamaProvider, OpenAIProvider
from omop_emb.embeddings.embedding_client import EmbeddingClientError
from omop_emb.config import ENV_DOCUMENT_EMBEDDING_PREFIX, ENV_QUERY_EMBEDDING_PREFIX

OLLAMA_BASE = "http://localhost:11434/v1"
OPENAI_BASE = "https://api.openai.com/v1"
OLLAMA_MODEL = "nomic-embed-text:v1.5"


def _make_embedding_response(vectors: list[list[float]]) -> Mock:
    """Minimal mock resembling openai.types.CreateEmbeddingResponse."""
    response = Mock()
    response.data = [Mock(embedding=v) for v in vectors]
    return response


@pytest.fixture
def mock_openai():
    """Patch the OpenAI constructor so no real HTTP client is created.

    Yields (mock_cls, openai_instance) — any EmbeddingClient built inside
    this fixture has openai_instance as its _base_client.
    """
    with patch("omop_emb.embeddings.embedding_client.OpenAI") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield mock_cls, instance


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEmbeddingClientInit:
    def test_explicit_provider_is_used(self, mock_openai):
        provider = OllamaProvider()
        client = EmbeddingClient(model=OLLAMA_MODEL, api_base=OLLAMA_BASE, provider=provider)
        assert client.provider is provider

    def test_model_is_canonicalized_at_construction(self, mock_openai):
        """OllamaProvider rejects untagged names; the error must surface from __init__."""
        with pytest.raises(ValueError, match="must include an explicit tag"):
            EmbeddingClient(model="nomic-embed-text", api_base=OLLAMA_BASE, provider=OllamaProvider())

    def test_custom_batch_size_is_stored(self, mock_openai):
        client = EmbeddingClient(
            model=OLLAMA_MODEL, api_base=OLLAMA_BASE, embedding_batch_size=8, provider=OllamaProvider()
        )
        assert client.embedding_batch_size == 8

    def test_ollama_provider_inferred_from_ollama_url(self, mock_openai):
        client = EmbeddingClient(model=OLLAMA_MODEL, api_base="http://ollama.internal/v1")
        assert isinstance(client.provider, OllamaProvider)

    def test_openai_provider_inferred_from_openai_url(self, mock_openai):
        client = EmbeddingClient(model="text-embedding-3-small", api_base=OPENAI_BASE, api_key="sk-real")
        assert isinstance(client.provider, OpenAIProvider)

    def test_canonical_model_name_stored_after_provider_processing(self, mock_openai):
        client = EmbeddingClient(model=OLLAMA_MODEL, api_base=OLLAMA_BASE, provider=OllamaProvider())
        assert client.canonical_model_name == OLLAMA_MODEL


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEmbeddingClientProperties:
    def test_provider_type_reflects_ollama(self, mock_openai):
        client = EmbeddingClient(model=OLLAMA_MODEL, api_base=OLLAMA_BASE, provider=OllamaProvider())
        assert client.provider.provider_type == ProviderType.OLLAMA

    def test_provider_type_reflects_openai(self, mock_openai):
        client = EmbeddingClient(
            model="text-embedding-3-small", api_base=OPENAI_BASE, api_key="sk-x", provider=OpenAIProvider()
        )
        assert client.provider.provider_type == ProviderType.OPENAI

    def test_api_key_accessible(self, mock_openai):
        _, openai_instance = mock_openai
        openai_instance.api_key = "my-key"
        client = EmbeddingClient(
            model=OLLAMA_MODEL, api_base=OLLAMA_BASE, api_key="my-key", provider=OllamaProvider()
        )
        assert client.api_key == "my-key"

    def test_base_client_is_the_openai_instance(self, mock_openai):
        _, openai_instance = mock_openai
        client = EmbeddingClient(model=OLLAMA_MODEL, api_base=OLLAMA_BASE, provider=OllamaProvider())
        assert client.base_client is openai_instance


# ---------------------------------------------------------------------------
# embedding_dim: lazy loading and caching
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEmbeddingDim:
    def _mock_provider(self, dim: int) -> Mock:
        p = Mock(spec=OllamaProvider)
        p.canonical_model_name.side_effect = lambda n: n
        p.provider_type = ProviderType.OLLAMA
        p.get_embedding_dim.return_value = dim
        return p

    def test_delegates_to_provider_get_embedding_dim(self, mock_openai):
        provider = self._mock_provider(768)
        client = EmbeddingClient(model=OLLAMA_MODEL, api_base=OLLAMA_BASE, provider=provider)
        assert client.embedding_dim == 768
        provider.get_embedding_dim.assert_called_once()

    def test_cached_after_first_access(self, mock_openai):
        provider = self._mock_provider(384)
        client = EmbeddingClient(model=OLLAMA_MODEL, api_base=OLLAMA_BASE, provider=provider)
        _ = client.embedding_dim
        _ = client.embedding_dim
        provider.get_embedding_dim.assert_called_once()

    def test_openai_provider_falls_back_to_live_probe(self, mock_openai):
        """OpenAI provider doesn't support get_embedding_dim; falls back to a live probe."""
        _, openai_instance = mock_openai
        openai_instance.embeddings.create.return_value = _make_embedding_response([[0.1] * 1536])

        client = EmbeddingClient(
            model="text-embedding-3-small", api_base=OPENAI_BASE, api_key="sk-x", provider=OpenAIProvider()
        )
        assert client.embedding_dim == 1536
        openai_instance.embeddings.create.assert_called_once()


# ---------------------------------------------------------------------------
# embeddings(): batching, shapes, input coercions
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEmbeddings:
    @pytest.fixture
    def client(self, mock_openai):
        _, openai_instance = mock_openai
        c = EmbeddingClient(
            model=OLLAMA_MODEL, api_base=OLLAMA_BASE, embedding_batch_size=32, provider=OllamaProvider()
        )
        return c, openai_instance

    def test_single_string_returns_2d_array(self, client):
        c, oi = client
        oi.embeddings.create.return_value = _make_embedding_response([[0.1, 0.2, 0.3]])
        result = c.embeddings("hello", embedding_role=EmbeddingRole.DOCUMENT)
        assert result.ndim == 2
        assert result.shape == (1, 3)

    def test_list_input_returns_correct_shape(self, client):
        c, oi = client
        oi.embeddings.create.return_value = _make_embedding_response([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        result = c.embeddings(["a", "b", "c"], embedding_role=EmbeddingRole.DOCUMENT)
        assert result.shape == (3, 2)

    def test_tuple_input_is_accepted(self, client):
        c, oi = client
        oi.embeddings.create.return_value = _make_embedding_response([[1.0, 0.0]])
        result = c.embeddings(("only",), embedding_role=EmbeddingRole.DOCUMENT)
        assert result.shape == (1, 2)

    def test_values_preserved_in_output(self, client):
        c, oi = client
        oi.embeddings.create.return_value = _make_embedding_response([[1.0, 2.0]])
        result = c.embeddings("text", embedding_role=EmbeddingRole.DOCUMENT)
        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0])

    def test_batching_splits_texts_into_chunks(self, mock_openai):
        _, oi = mock_openai
        c = EmbeddingClient(
            model=OLLAMA_MODEL, api_base=OLLAMA_BASE, embedding_batch_size=2, provider=OllamaProvider()
        )
        oi.embeddings.create.side_effect = [
            _make_embedding_response([[1.0, 0.0], [0.0, 1.0]]),
            _make_embedding_response([[0.5, 0.5]]),
        ]
        result = c.embeddings(["a", "b", "c"], embedding_role=EmbeddingRole.DOCUMENT)
        assert oi.embeddings.create.call_count == 2
        assert result.shape == (3, 2)

    def test_texts_exactly_filling_batch_produce_one_api_call(self, mock_openai):
        _, oi = mock_openai
        c = EmbeddingClient(
            model=OLLAMA_MODEL, api_base=OLLAMA_BASE, embedding_batch_size=3, provider=OllamaProvider()
        )
        oi.embeddings.create.return_value = _make_embedding_response([[1.0], [2.0], [3.0]])
        result = c.embeddings(["a", "b", "c"], embedding_role=EmbeddingRole.DOCUMENT)
        assert oi.embeddings.create.call_count == 1
        assert result.shape == (3, 1)

    def test_per_call_batch_size_override(self, mock_openai):
        _, oi = mock_openai
        c = EmbeddingClient(
            model=OLLAMA_MODEL, api_base=OLLAMA_BASE, embedding_batch_size=32, provider=OllamaProvider()
        )
        oi.embeddings.create.side_effect = [
            _make_embedding_response([[1.0]]),
            _make_embedding_response([[0.0]]),
        ]
        result = c.embeddings(["a", "b"], embedding_role=EmbeddingRole.DOCUMENT, batch_size=1)
        assert oi.embeddings.create.call_count == 2
        assert result.shape == (2, 1)

    def test_batched_results_concatenated_in_order(self, mock_openai):
        _, oi = mock_openai
        c = EmbeddingClient(
            model=OLLAMA_MODEL, api_base=OLLAMA_BASE, embedding_batch_size=1, provider=OllamaProvider()
        )
        oi.embeddings.create.side_effect = [
            _make_embedding_response([[1.0, 0.0]]),
            _make_embedding_response([[0.0, 1.0]]),
        ]
        result = c.embeddings(["first", "second"], embedding_role=EmbeddingRole.DOCUMENT)
        np.testing.assert_array_almost_equal(result[0], [1.0, 0.0])
        np.testing.assert_array_almost_equal(result[1], [0.0, 1.0])


# ---------------------------------------------------------------------------
# cosine_similarity(): numerical correctness and edge cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = np.array([[1.0, 0.0, 0.0]])
        np.testing.assert_almost_equal(EmbeddingClient.cosine_similarity(v, v)[0, 0], 1.0)

    def test_orthogonal_vectors_score_zero(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        np.testing.assert_almost_equal(EmbeddingClient.cosine_similarity(a, b)[0, 0], 0.0)

    def test_opposite_vectors_score_minus_one(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[-1.0, 0.0]])
        np.testing.assert_almost_equal(EmbeddingClient.cosine_similarity(a, b)[0, 0], -1.0)

    def test_output_shape_is_m_by_n(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert EmbeddingClient.cosine_similarity(a, b).shape == (3, 2)

    def test_scale_invariant(self):
        """Vectors pointing in the same direction must always score 1.0."""
        a = np.array([[3.0, 4.0]])
        b = np.array([[6.0, 8.0]])
        np.testing.assert_almost_equal(EmbeddingClient.cosine_similarity(a, b)[0, 0], 1.0)

    def test_zero_vector_does_not_raise_or_produce_nan(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[1.0, 0.0]])
        result = EmbeddingClient.cosine_similarity(a, b)
        assert np.isfinite(result).all()

    def test_raises_for_1d_input(self):
        with pytest.raises(RuntimeError, match="2-D"):
            EmbeddingClient.cosine_similarity(np.array([1.0, 0.0]), np.array([[1.0, 0.0]]))

    def test_raises_for_0d_input(self):
        with pytest.raises(RuntimeError, match="2-D"):
            EmbeddingClient.cosine_similarity(np.array(1.0), np.array([[1.0]]))


# ---------------------------------------------------------------------------
# similarity(): dispatch between string/list/array inputs
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSimilarity:
    @pytest.fixture
    def client(self, mock_openai):
        _, oi = mock_openai
        c = EmbeddingClient(model=OLLAMA_MODEL, api_base=OLLAMA_BASE, provider=OllamaProvider())
        return c, oi

    def test_string_inputs_trigger_embedding_calls(self, client):
        c, oi = client
        oi.embeddings.create.return_value = _make_embedding_response([[1.0, 0.0]])
        c.similarity("foo", "bar", EmbeddingRole.QUERY, EmbeddingRole.DOCUMENT)
        assert oi.embeddings.create.call_count == 2

    def test_list_inputs_trigger_embedding_calls(self, client):
        c, oi = client
        oi.embeddings.create.side_effect = [
            _make_embedding_response([[1.0, 0.0], [0.0, 1.0]]),
            _make_embedding_response([[1.0, 0.0]]),
        ]
        c.similarity(["a", "b"], ["c"], EmbeddingRole.QUERY, EmbeddingRole.DOCUMENT)
        assert oi.embeddings.create.call_count == 2

    def test_array_inputs_bypass_embedding_calls(self, client):
        c, oi = client
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        c.similarity(a, b, EmbeddingRole.QUERY, EmbeddingRole.DOCUMENT)
        oi.embeddings.create.assert_not_called()

    def test_array_inputs_return_correct_similarity(self, client):
        c, _ = client
        a = np.array([[1.0, 0.0]])
        b = np.array([[1.0, 0.0]])
        np.testing.assert_almost_equal(
            c.similarity(a, b, EmbeddingRole.QUERY, EmbeddingRole.DOCUMENT)[0, 0], 1.0
        )

    def test_mixed_str_and_array_input(self, client):
        c, oi = client
        oi.embeddings.create.return_value = _make_embedding_response([[1.0, 0.0]])
        b = np.array([[1.0, 0.0]])
        result = c.similarity("text", b, EmbeddingRole.QUERY, EmbeddingRole.DOCUMENT)
        assert oi.embeddings.create.call_count == 1
        assert result.shape == (1, 1)


# ---------------------------------------------------------------------------
# euclidean_distance()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEuclideanDistance:
    def test_same_embeddings_have_zero_distance(self, mock_openai):
        _, oi = mock_openai
        c = EmbeddingClient(model=OLLAMA_MODEL, api_base=OLLAMA_BASE, provider=OllamaProvider())
        oi.embeddings.create.return_value = _make_embedding_response([[1.0, 2.0, 3.0]])
        assert c.euclidean_distance("hello", "hello", EmbeddingRole.QUERY, EmbeddingRole.QUERY) == pytest.approx(0.0)

    def test_known_3_4_5_distance(self, mock_openai):
        _, oi = mock_openai
        c = EmbeddingClient(model=OLLAMA_MODEL, api_base=OLLAMA_BASE, provider=OllamaProvider())
        oi.embeddings.create.side_effect = [
            _make_embedding_response([[0.0, 0.0]]),
            _make_embedding_response([[3.0, 4.0]]),
        ]
        assert c.euclidean_distance("a", "b", EmbeddingRole.QUERY, EmbeddingRole.DOCUMENT) == pytest.approx(5.0)

    def test_returns_python_float(self, mock_openai):
        _, oi = mock_openai
        c = EmbeddingClient(model=OLLAMA_MODEL, api_base=OLLAMA_BASE, provider=OllamaProvider())
        oi.embeddings.create.return_value = _make_embedding_response([[1.0]])
        assert isinstance(c.euclidean_distance("x", "x", EmbeddingRole.QUERY, EmbeddingRole.QUERY), float)

    def test_distance_is_symmetric(self, mock_openai):
        _, oi = mock_openai
        c = EmbeddingClient(model=OLLAMA_MODEL, api_base=OLLAMA_BASE, provider=OllamaProvider())
        oi.embeddings.create.side_effect = [
            _make_embedding_response([[1.0, 0.0]]),
            _make_embedding_response([[0.0, 1.0]]),
            _make_embedding_response([[0.0, 1.0]]),
            _make_embedding_response([[1.0, 0.0]]),
        ]
        d_ab = c.euclidean_distance("a", "b", EmbeddingRole.QUERY, EmbeddingRole.DOCUMENT)
        d_ba = c.euclidean_distance("b", "a", EmbeddingRole.DOCUMENT, EmbeddingRole.QUERY)
        assert d_ab == pytest.approx(d_ba)


# ---------------------------------------------------------------------------
# EmbeddingClientError
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEmbeddingClientError:
    def test_is_subclass_of_runtime_error(self):
        assert issubclass(EmbeddingClientError, RuntimeError)

    def test_can_be_raised_and_caught_as_runtime_error(self):
        with pytest.raises(RuntimeError):
            raise EmbeddingClientError("something went wrong")

    def test_preserves_message(self):
        with pytest.raises(EmbeddingClientError, match="something went wrong"):
            raise EmbeddingClientError("something went wrong")


# ---------------------------------------------------------------------------
# load_embedding_prefixes(): env var loading and startup logging
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadEmbeddingPrefixes:
    def test_returns_empty_strings_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv(ENV_DOCUMENT_EMBEDDING_PREFIX, raising=False)
        monkeypatch.delenv(ENV_QUERY_EMBEDDING_PREFIX, raising=False)
        doc_prefix, query_prefix = EmbeddingClient.load_embedding_prefixes()
        assert doc_prefix == ""
        assert query_prefix == ""

    def test_returns_configured_prefixes(self, monkeypatch):
        monkeypatch.setenv(ENV_DOCUMENT_EMBEDDING_PREFIX, "search_document: ")
        monkeypatch.setenv(ENV_QUERY_EMBEDDING_PREFIX, "search_query: ")
        doc_prefix, query_prefix = EmbeddingClient.load_embedding_prefixes()
        assert doc_prefix == "search_document: "
        assert query_prefix == "search_query: "

    def test_logs_info_when_document_prefix_is_set(self, monkeypatch, caplog):
        monkeypatch.setenv(ENV_DOCUMENT_EMBEDDING_PREFIX, "passage: ")
        monkeypatch.delenv(ENV_QUERY_EMBEDDING_PREFIX, raising=False)
        import logging
        with caplog.at_level(logging.INFO, logger="omop_emb.embeddings.embedding_client"):
            EmbeddingClient.load_embedding_prefixes()
        assert any(
            "passage: " in r.message and r.levelname == "INFO"
            for r in caplog.records
        )

    def test_logs_info_when_query_prefix_is_set(self, monkeypatch, caplog):
        monkeypatch.delenv(ENV_DOCUMENT_EMBEDDING_PREFIX, raising=False)
        monkeypatch.setenv(ENV_QUERY_EMBEDDING_PREFIX, "search_query: ")
        import logging
        with caplog.at_level(logging.INFO, logger="omop_emb.embeddings.embedding_client"):
            EmbeddingClient.load_embedding_prefixes()
        assert any(
            "search_query: " in r.message and r.levelname == "INFO"
            for r in caplog.records
        )

    def test_logs_warning_when_document_prefix_not_set(self, monkeypatch, caplog):
        monkeypatch.delenv(ENV_DOCUMENT_EMBEDDING_PREFIX, raising=False)
        monkeypatch.delenv(ENV_QUERY_EMBEDDING_PREFIX, raising=False)
        import logging
        with caplog.at_level(logging.WARNING, logger="omop_emb.embeddings.embedding_client"):
            EmbeddingClient.load_embedding_prefixes()
        warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any(ENV_DOCUMENT_EMBEDDING_PREFIX in m for m in warning_messages)

    def test_logs_warning_when_query_prefix_not_set(self, monkeypatch, caplog):
        monkeypatch.delenv(ENV_DOCUMENT_EMBEDDING_PREFIX, raising=False)
        monkeypatch.delenv(ENV_QUERY_EMBEDDING_PREFIX, raising=False)
        import logging
        with caplog.at_level(logging.WARNING, logger="omop_emb.embeddings.embedding_client"):
            EmbeddingClient.load_embedding_prefixes()
        warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any(ENV_QUERY_EMBEDDING_PREFIX in m for m in warning_messages)


# ---------------------------------------------------------------------------
# _apply_embedding_prefix(): prefix application per role
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestApplyEmbeddingPrefix:
    @pytest.fixture
    def client_with_prefixes(self, monkeypatch, mock_openai):
        monkeypatch.setenv(ENV_DOCUMENT_EMBEDDING_PREFIX, "doc: ")
        monkeypatch.setenv(ENV_QUERY_EMBEDDING_PREFIX, "query: ")
        return EmbeddingClient(model=OLLAMA_MODEL, api_base=OLLAMA_BASE, provider=OllamaProvider())

    @pytest.fixture
    def client_no_prefixes(self, monkeypatch, mock_openai):
        monkeypatch.delenv(ENV_DOCUMENT_EMBEDDING_PREFIX, raising=False)
        monkeypatch.delenv(ENV_QUERY_EMBEDDING_PREFIX, raising=False)
        return EmbeddingClient(model=OLLAMA_MODEL, api_base=OLLAMA_BASE, provider=OllamaProvider())

    def test_document_prefix_prepended_to_string(self, client_with_prefixes):
        c = client_with_prefixes
        result = c._apply_embedding_prefix("Hypertension", text_role=EmbeddingRole.DOCUMENT)
        assert result == "doc: Hypertension"

    def test_query_prefix_prepended_to_string(self, client_with_prefixes):
        c = client_with_prefixes
        result = c._apply_embedding_prefix("high blood pressure", text_role=EmbeddingRole.QUERY)
        assert result == "query: high blood pressure"

    def test_document_and_query_produce_different_prefixed_texts(self, client_with_prefixes):
        c = client_with_prefixes
        doc = c._apply_embedding_prefix("Hypertension", text_role=EmbeddingRole.DOCUMENT)
        query = c._apply_embedding_prefix("Hypertension", text_role=EmbeddingRole.QUERY)
        assert doc != query

    def test_prefix_applied_to_all_items_in_tuple(self, client_with_prefixes):
        c = client_with_prefixes
        result = c._apply_embedding_prefix(("a", "b"), text_role=EmbeddingRole.DOCUMENT)
        assert result == ("doc: a", "doc: b")

    def test_prefix_applied_to_all_items_in_list(self, client_with_prefixes):
        c = client_with_prefixes
        result = c._apply_embedding_prefix(["a", "b"], text_role=EmbeddingRole.DOCUMENT)
        assert result == ["doc: a", "doc: b"]

    def test_no_prefix_returns_texts_unchanged_string(self, client_no_prefixes):
        c = client_no_prefixes
        result = c._apply_embedding_prefix("Hypertension", text_role=EmbeddingRole.DOCUMENT)
        assert result == "Hypertension"

    def test_no_prefix_returns_texts_unchanged_list(self, client_no_prefixes):
        c = client_no_prefixes
        result = c._apply_embedding_prefix(["a", "b"], text_role=EmbeddingRole.QUERY)
        assert result == ["a", "b"]

    def test_prefix_reflected_in_api_call(self, monkeypatch, mock_openai):
        """Verify the prefixed text reaches the OpenAI API call."""
        _, oi = mock_openai
        monkeypatch.setenv(ENV_DOCUMENT_EMBEDDING_PREFIX, "passage: ")
        monkeypatch.delenv(ENV_QUERY_EMBEDDING_PREFIX, raising=False)
        c = EmbeddingClient(model=OLLAMA_MODEL, api_base=OLLAMA_BASE, provider=OllamaProvider())
        oi.embeddings.create.return_value = _make_embedding_response([[0.1, 0.2]])
        c.embeddings("diabetes", embedding_role=EmbeddingRole.DOCUMENT)
        call_input = oi.embeddings.create.call_args.kwargs["input"]
        assert call_input == ("passage: diabetes",)

    def test_no_prefix_passes_text_verbatim_to_api(self, monkeypatch, mock_openai):
        _, oi = mock_openai
        monkeypatch.delenv(ENV_DOCUMENT_EMBEDDING_PREFIX, raising=False)
        monkeypatch.delenv(ENV_QUERY_EMBEDDING_PREFIX, raising=False)
        c = EmbeddingClient(model=OLLAMA_MODEL, api_base=OLLAMA_BASE, provider=OllamaProvider())
        oi.embeddings.create.return_value = _make_embedding_response([[0.1, 0.2]])
        c.embeddings("diabetes", embedding_role=EmbeddingRole.DOCUMENT)
        call_input = oi.embeddings.create.call_args.kwargs["input"]
        assert call_input == ("diabetes",)
