"""Tests for the per-backend capabilities object and its validation."""

from __future__ import annotations

import pytest

from omop_emb.backends.capabilities import BackendCapabilities
from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.backends.sqlitevec import SQLiteVecEmbeddingBackend
from omop_emb.config import BackendType, IndexType, MetricType
from omop_emb.utils.errors import UnsupportedIndexConfigError

_ALL_BACKENDS = [
    pytest.param(BackendType.SQLITEVEC, id="sqlitevec"),
    pytest.param(BackendType.PGVECTOR, id="pgvector"),
    pytest.param(BackendType.DUCKDBVSS, id="duckdbvss"),
]


def _declared(backend_type: BackendType) -> BackendCapabilities:
    if backend_type == BackendType.SQLITEVEC:
        return SQLiteVecEmbeddingBackend.CAPABILITIES
    if backend_type == BackendType.PGVECTOR:
        from omop_emb.backends.pgvector import PGVectorEmbeddingBackend

        return PGVectorEmbeddingBackend.CAPABILITIES
    from omop_emb.backends.duckdbvss import DuckDBVSSEmbeddingBackend

    return DuckDBVSSEmbeddingBackend.CAPABILITIES


@pytest.mark.unit
@pytest.mark.parametrize("backend_type", _ALL_BACKENDS)
def test_each_backend_declares_its_own_capabilities(backend_type):
    assert _declared(backend_type).backend_type == backend_type


@pytest.mark.unit
class TestDeclaredSupport:
    def test_sqlitevec_is_flat_only(self):
        capabilities = _declared(BackendType.SQLITEVEC)
        assert capabilities.index_types == (IndexType.FLAT,)
        assert not capabilities.max_scan_tuples

    def test_pgvector_supports_hnsw_and_scan_cap(self):
        capabilities = _declared(BackendType.PGVECTOR)
        assert capabilities.supports_index_type(IndexType.HNSW)
        assert capabilities.supports_metric(IndexType.HNSW, MetricType.L1)
        assert capabilities.max_scan_tuples

    def test_duckdb_has_no_l1_and_no_scan_cap(self):
        capabilities = _declared(BackendType.DUCKDBVSS)
        assert not capabilities.supports_metric(IndexType.HNSW, MetricType.L1)
        assert not capabilities.max_scan_tuples

    @pytest.mark.parametrize("backend_type", _ALL_BACKENDS)
    def test_bit_metrics_are_unsupported(self, backend_type):
        capabilities = _declared(backend_type)
        for index_type in IndexType:
            assert not capabilities.supports_metric(index_type, MetricType.HAMMING)


@pytest.mark.unit
class TestValidateIndexConfig:
    def test_accepts_supported_config(self):
        _declared(BackendType.PGVECTOR).validate_index_config(
            HNSWIndexConfig(metric_type=MetricType.COSINE, max_scan_tuples=100_000)
        )

    def test_rejects_unsupported_index_type_naming_backend(self):
        with pytest.raises(UnsupportedIndexConfigError, match="'hnsw'.*'sqlitevec'"):
            _declared(BackendType.SQLITEVEC).validate_index_config(
                HNSWIndexConfig(metric_type=MetricType.COSINE)
            )

    def test_rejects_unsupported_metric_naming_backend(self):
        with pytest.raises(UnsupportedIndexConfigError, match="'l1'.*'duckdbvss'"):
            _declared(BackendType.DUCKDBVSS).validate_index_config(
                HNSWIndexConfig(metric_type=MetricType.L1)
            )

    def test_flat_without_metric_is_accepted(self):
        _declared(BackendType.PGVECTOR).validate_index_config(FlatIndexConfig(metric_type=None))

    def test_rejects_scan_cap_where_unsupported(self):
        with pytest.raises(UnsupportedIndexConfigError, match="max_scan_tuples.*'duckdbvss'"):
            _declared(BackendType.DUCKDBVSS).validate_index_config(
                HNSWIndexConfig(metric_type=MetricType.COSINE, max_scan_tuples=100)
            )

    def test_unset_scan_cap_is_accepted_everywhere(self):
        _declared(BackendType.DUCKDBVSS).validate_index_config(
            HNSWIndexConfig(metric_type=MetricType.COSINE)
        )
