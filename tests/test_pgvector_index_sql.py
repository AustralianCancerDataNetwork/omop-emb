"""pgvector HNSW index naming, operator classes and DDL (pure unit, no DB)."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex

pytest.importorskip("pgvector", reason="omop-emb[pgvector] not installed: skipping pgvector tests")

from omop_emb.backends.db_utils import table_for
from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.backends.index_naming import IndexNames
from omop_emb.backends.pgvector.pg_sql import (
    column_type_for,
    default_storage_precision,
    hnsw_index_ddl,
    hnsw_operator_class,
    partial_hnsw_index,
    pg_embedding_table_descriptor,
    q_domain_ids,
)
from omop_emb.config import IndexType, MetricType, VectorColumnType, VectorPrecision
from omop_emb.model_registry import EmbeddingModelRecord, RegistryManager
from omop_emb.model_registry.model_registry_manager import (
    STORAGE_IDENTIFIER_READABLE_PREFIX_LENGTH,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def unconnected_pg_engine() -> sa.Engine:
    """A Postgres Engine that is never connected, giving qualified() a real dialect to quote against."""
    return sa.create_engine(
        "postgresql+psycopg://unused:unused@localhost/unused"
    ).execution_options(schema_translate_map={"primary": "public"})


def _record(dimensions: int = 4) -> EmbeddingModelRecord:
    return EmbeddingModelRecord(
        model_name="m",
        provider_type="ollama",
        index_config=FlatIndexConfig(),
        dimensions=dimensions,
        storage_identifier="my_table",
    )


def test_store_scoped_storage_names_are_deterministic_and_fit_every_index():
    model_name = "hf.co/second-state/multilingual-e5-large-instruct-GGUF:" + "x" * 80
    storage_id = RegistryManager.storage_name("store_a", model_name)

    assert storage_id == RegistryManager.storage_name("store_a", model_name)
    assert storage_id != RegistryManager.storage_name("store_b", model_name)
    assert len(storage_id.encode("ascii")) <= 63
    readable_prefix = storage_id[len("emb_") : storage_id.rfind("_")]
    assert len(readable_prefix) == STORAGE_IDENTIFIER_READABLE_PREFIX_LENGTH == 38
    assert all(char in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in readable_prefix)
    for metric in MetricType:
        index_name = IndexNames.hnsw(storage_id, metric)
        assert len(index_name.encode("ascii")) <= 63

    sanitized = RegistryManager.storage_name("store_a", " ..My__Model!! ")
    assert sanitized[4 : sanitized.rfind("_")] == "my_model"


@pytest.mark.parametrize(
    ("metric", "precision", "expected"),
    [
        (MetricType.L2, VectorPrecision.FLOAT32, "vector_l2_ops"),
        (MetricType.COSINE, VectorPrecision.FLOAT32, "vector_cosine_ops"),
        (MetricType.L1, VectorPrecision.FLOAT32, "vector_l1_ops"),
        (MetricType.L2, VectorPrecision.FLOAT16, "halfvec_l2_ops"),
        (MetricType.COSINE, VectorPrecision.FLOAT16, "halfvec_cosine_ops"),
    ],
)
def test_operator_class_follows_metric_and_index_precision(metric, precision, expected):
    assert hnsw_operator_class(metric, precision) == expected


def test_ddl_carries_name_ops_and_build_parameters(unconnected_pg_engine):
    config = HNSWIndexConfig(
        metric_type=MetricType.COSINE, num_neighbors=32, ef_search=64, ef_construction=128
    )
    ddl = hnsw_index_ddl(unconnected_pg_engine, _record(), config, VectorPrecision.FLOAT32)
    assert ddl.startswith("CREATE INDEX idx_my_table_cosine ON ")
    assert "USING hnsw (embedding vector_cosine_ops)" in ddl
    assert "m = 32" in ddl
    assert "ef_construction = 128" in ddl


def test_ddl_differs_by_build_parameters(unconnected_pg_engine):
    a = hnsw_index_ddl(
        unconnected_pg_engine,
        _record(),
        HNSWIndexConfig(metric_type=MetricType.L2, num_neighbors=8),
        VectorPrecision.FLOAT32,
    )
    b = hnsw_index_ddl(
        unconnected_pg_engine,
        _record(),
        HNSWIndexConfig(metric_type=MetricType.L2, num_neighbors=64),
        VectorPrecision.FLOAT32,
    )
    assert "m = 8" in a
    assert "m = 64" in b


def test_column_type_follows_precision():
    assert column_type_for(VectorPrecision.FLOAT32) == VectorColumnType.VECTOR
    assert column_type_for(VectorPrecision.FLOAT16) == VectorColumnType.HALFVEC


def test_default_storage_precision_switches_above_the_vector_index_limit():
    assert default_storage_precision(512) == VectorPrecision.FLOAT32
    assert default_storage_precision(2000) == VectorPrecision.FLOAT32
    assert default_storage_precision(2001) == VectorPrecision.FLOAT16


def test_column_type_rejects_a_precision_it_has_no_type_for():
    with pytest.raises(ValueError, match="no column type"):
        column_type_for(cast(VectorPrecision, "float8"))


def test_ddl_over_half_cast_indexes_the_cast_expression(unconnected_pg_engine):
    config = HNSWIndexConfig(metric_type=MetricType.COSINE, index_precision=VectorPrecision.FLOAT16)
    ddl = hnsw_index_ddl(unconnected_pg_engine, _record(3000), config, VectorPrecision.FLOAT32)
    assert "USING hnsw ((embedding::halfvec(3000)) halfvec_cosine_ops)" in ddl


def test_ddl_on_halfvec_column_indexes_the_column_directly(unconnected_pg_engine):
    config = HNSWIndexConfig(metric_type=MetricType.COSINE, index_precision=VectorPrecision.FLOAT16)
    ddl = hnsw_index_ddl(unconnected_pg_engine, _record(3000), config, VectorPrecision.FLOAT16)
    assert "USING hnsw (embedding halfvec_cosine_ops)" in ddl


def test_ddl_without_an_index_precision_follows_the_stored_column(unconnected_pg_engine):
    config = HNSWIndexConfig(metric_type=MetricType.COSINE)
    ddl = hnsw_index_ddl(unconnected_pg_engine, _record(3000), config, VectorPrecision.FLOAT16)
    assert "USING hnsw (embedding halfvec_cosine_ops)" in ddl


@pytest.mark.parametrize("metric", [MetricType.HAMMING, MetricType.JACCARD])
def test_bit_metrics_not_in_pgvector_supported_metrics(metric):
    from omop_emb.backends.pgvector import PGVectorEmbeddingBackend

    for index in (IndexType.FLAT, IndexType.HNSW):
        assert not PGVectorEmbeddingBackend.CAPABILITIES.supports_metric(index, metric)


@pytest.mark.parametrize("metric", [MetricType.HAMMING, MetricType.JACCARD])
def test_get_distance_raises_for_bit_metrics(metric):
    from omop_emb.backends.embedding_table import PGEmbeddingTable
    from omop_emb.backends.pgvector.pg_sql import get_distance

    class _FakeTable:
        embedding = None

    with pytest.raises(ValueError, match="bit"):
        get_distance(
            cast(type[PGEmbeddingTable], _FakeTable), np.empty(0, dtype=np.float32), metric
        )


def test_get_similarity_raises_valueerror_for_hamming():
    from omop_emb.utils.embedding_utils import get_similarity_from_distance

    with pytest.raises(ValueError, match="HAMMING"):
        get_similarity_from_distance(0.5, MetricType.HAMMING)


class TestPartialIndexes:
    @staticmethod
    def _ddl(config: HNSWIndexConfig, domain: str, precision=VectorPrecision.FLOAT32, dims=4):
        index = partial_hnsw_index(_record(dims), config, precision, domain)
        return str(CreateIndex(index).compile(dialect=postgresql.dialect()))

    def test_ddl_restricts_the_index_to_the_domain(self):
        config = HNSWIndexConfig(metric_type=MetricType.COSINE, partial_domain_indexes=True)
        ddl = self._ddl(config, "Condition")

        assert f"CREATE INDEX {IndexNames.partial('my_table', 'Condition')} ON " in ddl
        assert "USING hnsw (embedding vector_cosine_ops)" in ddl
        assert "m = 32, ef_construction = 200)" in ddl
        assert ddl.endswith(" WHERE domain_id = 'Condition'")

    def test_ddl_adds_the_valid_restriction_when_valid_only(self):
        config = HNSWIndexConfig(
            metric_type=MetricType.COSINE, partial_domain_indexes=True, partial_valid_only=True
        )
        assert self._ddl(config, "Condition").endswith(
            " WHERE domain_id = 'Condition' AND is_valid = true"
        )

    def test_ddl_keeps_the_half_precision_cast(self):
        config = HNSWIndexConfig(
            metric_type=MetricType.COSINE,
            index_precision=VectorPrecision.FLOAT16,
            partial_domain_indexes=True,
        )
        ddl = self._ddl(config, "Drug", dims=3000)
        assert "USING hnsw (CAST(embedding AS HALFVEC(3000)) halfvec_cosine_ops)" in ddl

    def test_ddl_quotes_a_domain_with_quotes_and_colons(self):
        config = HNSWIndexConfig(metric_type=MetricType.COSINE, partial_domain_indexes=True)
        assert self._ddl(config, "O'Neil:x").endswith(" WHERE domain_id = 'O''Neil:x'")

    def test_creating_the_index_leaves_the_table_without_indexes(self):
        config = HNSWIndexConfig(metric_type=MetricType.COSINE, partial_domain_indexes=True)
        table = table_for(pg_embedding_table_descriptor(_record(), VectorPrecision.FLOAT32))
        before = set(table.indexes)

        partial_hnsw_index(_record(), config, VectorPrecision.FLOAT32, "Condition")

        assert set(table.indexes) == before

    def test_global_ddl_has_no_restriction(self, unconnected_pg_engine):
        config = HNSWIndexConfig(metric_type=MetricType.COSINE, partial_domain_indexes=True)
        ddl = hnsw_index_ddl(unconnected_pg_engine, _record(), config, VectorPrecision.FLOAT32)
        assert "WHERE" not in ddl

    def test_domain_query_counts_only_valid_rows_when_valid_only(self):
        table = pg_embedding_table_descriptor(_record(), VectorPrecision.FLOAT32)

        assert "is_valid" not in str(q_domain_ids(table, valid_only=False))
        assert "is_valid" in str(q_domain_ids(table, valid_only=True))
