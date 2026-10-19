"""The metric check applies to nearest-neighbour queries only."""

from __future__ import annotations

import inspect

import pytest

from omop_emb.backends.base_backend import EmbeddingBackend
from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.config import MetricType

from .conftest import (
    CONCEPT_EMBEDDINGS,
    CONCEPT_RECORDS,
    EMBEDDING_DIM,
    MODEL_NAME,
    PROVIDER_TYPE,
    QUERY_EMBEDDING,
    upsert_all,
)


def _query(backend: EmbeddingBackend, metric_type: MetricType):
    return backend.get_nearest_concepts(
        model_name=MODEL_NAME, metric_type=metric_type, query_embeddings=QUERY_EMBEDDING, k=1
    )


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


@pytest.mark.parametrize("metric", [MetricType.L2, MetricType.L1, MetricType.HAMMING])
def test_sqlitevec_model_accepts_only_its_registered_metric(svec_backend, metric):
    """The vec0 table declares the model's metric (cosine by default), so other metrics are refused."""
    upsert_all(svec_backend)
    with pytest.raises(ValueError, match="indexed with metric 'cosine'"):
        _query(svec_backend, metric)


@pytest.mark.pgvector
@pytest.mark.integration
@pytest.mark.parametrize("metric", [MetricType.L2, MetricType.COSINE, MetricType.L1])
def test_pgvector_flat_accepts_every_supported_metric(pg_backend, metric):
    upsert_all(pg_backend)
    assert len(_query(pg_backend, metric)[0]) == 1


@pytest.mark.pgvector
@pytest.mark.integration
def test_pgvector_flat_drops_a_given_metric_and_stays_free(pg_backend):
    record = pg_backend.register_model(
        model_name=MODEL_NAME,
        provider_type=PROVIDER_TYPE,
        index_config=FlatIndexConfig(metric_type=MetricType.L1),
        dimensions=EMBEDDING_DIM,
    )
    assert record.metric_type is None
    pg_backend.upsert_embeddings(model_name=MODEL_NAME, records=CONCEPT_RECORDS, embeddings=CONCEPT_EMBEDDINGS)
    for metric in (MetricType.L1, MetricType.COSINE):
        assert len(_query(pg_backend, metric)[0]) == 1
    rebuilt = pg_backend.rebuild_index(
        model_name=MODEL_NAME, index_config=FlatIndexConfig(metric_type=MetricType.L2)
    )
    assert rebuilt.metric_type is None


@pytest.mark.pgvector
@pytest.mark.integration
def test_pgvector_flat_rejects_an_unsupported_metric(pg_backend):
    upsert_all(pg_backend)
    with pytest.raises(ValueError, match="not supported"):
        _query(pg_backend, MetricType.HAMMING)


@pytest.mark.pgvector
@pytest.mark.integration
def test_hnsw_accepts_only_its_index_metric(pg_backend):
    upsert_all(pg_backend)
    pg_backend.rebuild_index(model_name=MODEL_NAME, index_config=HNSWIndexConfig(metric_type=MetricType.L2))
    assert len(_query(pg_backend, MetricType.L2)[0]) == 1
    with pytest.raises(ValueError, match="indexed with metric 'l2'"):
        _query(pg_backend, MetricType.COSINE)


@pytest.mark.pgvector
@pytest.mark.integration
def test_bundle_moves_between_sqlitevec_and_pgvector_without_carrying_the_lock(
    svec_backend, pg_backend, tmp_path
):
    """sqlite-vec locks its models to a metric; pgvector FLAT stays free, and the other way round."""
    pytest.importorskip("h5py")
    from omop_emb.storage import embedding_bundle

    svec_backend.register_model(
        model_name=MODEL_NAME,
        provider_type=PROVIDER_TYPE,
        index_config=FlatIndexConfig(metric_type=MetricType.L2),
        dimensions=EMBEDDING_DIM,
    )
    svec_backend.upsert_embeddings(model_name=MODEL_NAME, records=CONCEPT_RECORDS, embeddings=CONCEPT_EMBEDDINGS)
    _, bundle = embedding_bundle.export_bundle(svec_backend, MODEL_NAME, tmp_path / "from_svec")

    embedding_bundle.import_bundle(pg_backend, bundle, rebuild_index=True)

    assert pg_backend.get_registered_model(model_name=MODEL_NAME).metric_type is None
    for metric in (MetricType.L2, MetricType.COSINE, MetricType.L1):
        assert len(_query(pg_backend, metric)[0]) == 1

    _, back = embedding_bundle.export_bundle(pg_backend, MODEL_NAME, tmp_path / "from_pg")
    svec_backend.delete_model(model_name=MODEL_NAME)
    embedding_bundle.import_bundle(svec_backend, back, rebuild_index=True)

    assert svec_backend.get_registered_model(model_name=MODEL_NAME).metric_type == MetricType.COSINE
    assert len(_query(svec_backend, MetricType.COSINE)[0]) == 1
