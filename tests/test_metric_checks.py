"""The metric check applies to nearest-neighbour queries only."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from omop_emb.backends.base_backend import EmbeddingBackend
from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.config import MetricType

from .conftest import CONCEPT_RECORDS, EMBEDDING_DIM, MODEL_NAME, PROVIDER_TYPE

_QUERY = np.array([[-1.0]], dtype=np.float32)
_VECTORS = np.array([[-10.0], [10.0]], dtype=np.float32)


def _populate(backend: EmbeddingBackend) -> None:
    backend.register_model(
        model_name=MODEL_NAME, provider_type=PROVIDER_TYPE, index_config=FlatIndexConfig(), dimensions=EMBEDDING_DIM,
    )
    backend.upsert_embeddings(model_name=MODEL_NAME, records=list(CONCEPT_RECORDS[:2]), embeddings=_VECTORS)


def _query(backend: EmbeddingBackend, metric_type: MetricType):
    return backend.get_nearest_concepts(model_name=MODEL_NAME, metric_type=metric_type, query_embeddings=_QUERY, k=1)


@pytest.mark.parametrize(
    "method",
    [
        "upsert_embeddings",
        "bulk_upsert_embeddings",
        "get_embeddings_by_concept_ids",
        "get_concept_filter_metadata",
        "get_stored_concept_ids",
        "has_any_embeddings",
        "get_embedding_count",
        "get_embedding_count_by_vocabulary",
    ],
)
def test_only_queries_take_a_metric(method):
    assert "metric_type" not in inspect.signature(getattr(EmbeddingBackend, method)).parameters
    assert "metric_type" in inspect.signature(EmbeddingBackend.get_nearest_concepts).parameters


@pytest.mark.parametrize("metric", [MetricType.L2, MetricType.COSINE, MetricType.L1])
def test_flat_accepts_every_supported_metric(svec_backend, metric):
    _populate(svec_backend)
    assert len(_query(svec_backend, metric)[0]) == 1


def test_flat_rejects_an_unsupported_metric(svec_backend):
    _populate(svec_backend)
    with pytest.raises(ValueError, match="not supported"):
        _query(svec_backend, MetricType.HAMMING)


@pytest.mark.pgvector
@pytest.mark.integration
def test_hnsw_accepts_only_its_index_metric(pg_backend):
    _populate(pg_backend)
    pg_backend.rebuild_index(model_name=MODEL_NAME, index_config=HNSWIndexConfig(metric_type=MetricType.L2))
    assert len(_query(pg_backend, MetricType.L2)[0]) == 1
    with pytest.raises(ValueError, match="indexed with metric 'l2'"):
        _query(pg_backend, MetricType.COSINE)
