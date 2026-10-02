"""pgvector HNSW index naming, operator classes and DDL (pure unit, no DB)."""

from __future__ import annotations

import pytest
import sqlalchemy as sa

pytest.importorskip(
    "pgvector", reason="omop-emb[pgvector] not installed: skipping pgvector tests"
)

from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.backends.pgvector.pg_sql import (
    hnsw_index_ddl,
    hnsw_index_name,
    hnsw_operator_class,
)
from omop_emb.config import IndexType, MetricType, VectorColumnType
from omop_emb.model_registry import EmbeddingModelRecord
from omop_emb.utils.embedding_utils import vector_column_type_for_dimensions

pytestmark = pytest.mark.unit


@pytest.fixture
def unconnected_pg_engine() -> sa.Engine:
    """A Postgres Engine that is never connected, giving qualified() a real dialect to quote against."""
    return sa.create_engine("postgresql+psycopg://unused:unused@localhost/unused").execution_options(
        schema_translate_map={"primary": "public"}
    )


def _record(dimensions: int = 4) -> EmbeddingModelRecord:
    return EmbeddingModelRecord(
        model_name="m",
        provider_type="ollama",
        index_config=FlatIndexConfig(),
        dimensions=dimensions,
        storage_identifier="my_table",
    )


def test_index_name_format():
    assert hnsw_index_name("my_table", MetricType.L2) == "idx_my_table_l2"
    assert hnsw_index_name("my_table", MetricType.COSINE) == "idx_my_table_cosine"


@pytest.mark.parametrize(
    ("metric", "dimensions", "expected"),
    [
        (MetricType.L2, 4, "vector_l2_ops"),
        (MetricType.COSINE, 4, "vector_cosine_ops"),
        (MetricType.L1, 4, "vector_l1_ops"),
        (MetricType.L2, 3000, "halfvec_l2_ops"),
        (MetricType.COSINE, 3000, "halfvec_cosine_ops"),
    ],
)
def test_operator_class_follows_metric_and_column_type(metric, dimensions, expected):
    assert hnsw_operator_class(metric, dimensions) == expected


def test_ddl_carries_name_ops_and_build_parameters(unconnected_pg_engine):
    config = HNSWIndexConfig(metric_type=MetricType.COSINE, num_neighbors=32, ef_search=64, ef_construction=128)
    ddl = hnsw_index_ddl(unconnected_pg_engine, _record(), config)
    assert ddl.startswith("CREATE INDEX idx_my_table_cosine ON ")
    assert "USING hnsw (embedding vector_cosine_ops)" in ddl
    assert "m = 32" in ddl
    assert "ef_construction = 128" in ddl


def test_ddl_differs_by_build_parameters(unconnected_pg_engine):
    a = hnsw_index_ddl(unconnected_pg_engine, _record(), HNSWIndexConfig(metric_type=MetricType.L2, num_neighbors=8))
    b = hnsw_index_ddl(unconnected_pg_engine, _record(), HNSWIndexConfig(metric_type=MetricType.L2, num_neighbors=64))
    assert "m = 8" in a
    assert "m = 64" in b


def test_vector_column_type_auto_selection():
    assert vector_column_type_for_dimensions(512) == VectorColumnType.VECTOR
    assert vector_column_type_for_dimensions(2000) == VectorColumnType.VECTOR
    assert vector_column_type_for_dimensions(2001) == VectorColumnType.HALFVEC
    assert vector_column_type_for_dimensions(4000) == VectorColumnType.HALFVEC


def test_vector_column_type_rejects_oversized():
    from omop_emb.config import PGVECTOR_HALFVEC_MAX_DIMENSIONS

    with pytest.raises(ValueError):
        vector_column_type_for_dimensions(PGVECTOR_HALFVEC_MAX_DIMENSIONS + 1)


@pytest.mark.parametrize("metric", [MetricType.HAMMING, MetricType.JACCARD])
def test_bit_metrics_not_in_pgvector_supported_metrics(metric):
    from omop_emb.config import BackendType, is_supported_index_metric_combination_for_backend

    for index in (IndexType.FLAT, IndexType.HNSW):
        assert not is_supported_index_metric_combination_for_backend(
            backend=BackendType.PGVECTOR, index=index, metric=metric,
        )


@pytest.mark.parametrize("metric", [MetricType.HAMMING, MetricType.JACCARD])
def test_get_distance_raises_for_bit_metrics(metric):
    from omop_emb.backends.pgvector.pg_sql import get_distance

    class _FakeTable:
        embedding = None

    with pytest.raises(ValueError, match="bit"):
        get_distance(_FakeTable, [], metric)


def test_get_similarity_raises_valueerror_for_hamming():
    from omop_emb.utils.embedding_utils import get_similarity_from_distance

    with pytest.raises(ValueError, match="HAMMING"):
        get_similarity_from_distance(0.5, MetricType.HAMMING)
