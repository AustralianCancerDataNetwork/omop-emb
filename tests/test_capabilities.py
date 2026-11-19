"""Tests for the per-backend capabilities object and its validation."""

from __future__ import annotations

import pytest

from omop_emb.backends.capabilities import BackendCapabilities
from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.backends.sqlitevec import SQLiteVecEmbeddingBackend
from omop_emb.config import BackendType, IndexType, MetricType, VectorPrecision
from omop_emb.utils.errors import UnsupportedIndexConfigError

_ALL_BACKENDS = [
    pytest.param(BackendType.SQLITEVEC, id=BackendType.SQLITEVEC.value),
    pytest.param(BackendType.PGVECTOR, id=BackendType.PGVECTOR.value),
    pytest.param(BackendType.DUCKDBVSS, id=BackendType.DUCKDBVSS.value),
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

    def test_pgvector_supports_hnsw_and_scan_cap(self):
        capabilities = _declared(BackendType.PGVECTOR)
        assert capabilities.supports_index_type(IndexType.HNSW)
        assert capabilities.supports_metric(IndexType.HNSW, MetricType.L1)
        assert capabilities.indices[IndexType.HNSW].max_scan_tuples
        assert not capabilities.indices[IndexType.FLAT].max_scan_tuples

    def test_duckdb_has_no_l1_and_no_scan_cap(self):
        capabilities = _declared(BackendType.DUCKDBVSS)
        assert not capabilities.supports_metric(IndexType.HNSW, MetricType.L1)
        assert not capabilities.indices[IndexType.HNSW].max_scan_tuples

    @pytest.mark.parametrize("backend_type", _ALL_BACKENDS)
    def test_bit_metrics_are_unsupported(self, backend_type):
        capabilities = _declared(backend_type)
        for index_type in IndexType:
            assert not capabilities.supports_metric(index_type, MetricType.HAMMING)


def _validate(
    backend_type,
    index_config,
    *,
    dimensions=768,
    storage_precision=VectorPrecision.FLOAT32,
):
    _declared(backend_type).validate_index_config(
        index_config, dimensions=dimensions, storage_precision=storage_precision
    )


def _quoted(*members) -> str:
    """Regex matching the given enum values, each single-quoted, in order."""
    return ".*".join(f"'{member.value}'" for member in members)


@pytest.mark.unit
class TestValidateIndexConfig:
    def test_accepts_supported_config(self):
        _validate(
            BackendType.PGVECTOR,
            HNSWIndexConfig(metric_type=MetricType.COSINE, max_scan_tuples=100_000),
        )

    def test_rejects_unsupported_index_type_naming_backend(self):
        with pytest.raises(
            UnsupportedIndexConfigError, match=_quoted(IndexType.HNSW, BackendType.SQLITEVEC)
        ):
            _validate(BackendType.SQLITEVEC, HNSWIndexConfig(metric_type=MetricType.COSINE))

    def test_rejects_unsupported_metric_naming_backend(self):
        with pytest.raises(
            UnsupportedIndexConfigError, match=_quoted(MetricType.L1, BackendType.DUCKDBVSS)
        ):
            _validate(BackendType.DUCKDBVSS, HNSWIndexConfig(metric_type=MetricType.L1))

    def test_flat_without_metric_is_accepted(self):
        _validate(BackendType.PGVECTOR, FlatIndexConfig(metric_type=None))

    def test_rejects_scan_cap_where_unsupported(self):
        with pytest.raises(
            UnsupportedIndexConfigError,
            match=f"max_scan_tuples.*{_quoted(BackendType.DUCKDBVSS)}",
        ):
            _validate(
                BackendType.DUCKDBVSS,
                HNSWIndexConfig(metric_type=MetricType.COSINE, max_scan_tuples=100),
            )

    def test_unset_scan_cap_is_accepted_everywhere(self):
        _validate(BackendType.DUCKDBVSS, HNSWIndexConfig(metric_type=MetricType.COSINE))


@pytest.mark.unit
class TestPrecision:
    def test_pgvector_supports_both_precisions(self):
        capabilities = _declared(BackendType.PGVECTOR)
        assert capabilities.storage.precisions == (
            VectorPrecision.FLOAT32,
            VectorPrecision.FLOAT16,
        )
        assert capabilities.indices[IndexType.HNSW].precisions == (
            VectorPrecision.FLOAT32,
            VectorPrecision.FLOAT16,
        )
        assert capabilities.indices[IndexType.FLAT].precisions == ()

    @pytest.mark.parametrize("backend_type", [BackendType.SQLITEVEC, BackendType.DUCKDBVSS])
    def test_other_backends_store_float32_only(self, backend_type):
        assert _declared(backend_type).storage.precisions == (VectorPrecision.FLOAT32,)

    def test_duckdb_indexes_float32_only(self):
        capabilities = _declared(BackendType.DUCKDBVSS)
        assert capabilities.indices[IndexType.HNSW].precisions == (VectorPrecision.FLOAT32,)
        assert capabilities.indices[IndexType.FLAT].precisions == ()

    def test_sqlitevec_accepts_no_explicit_index_precision(self):
        capabilities = _declared(BackendType.SQLITEVEC)
        assert all(index.precisions == () for index in capabilities.indices.values())

    @pytest.mark.parametrize("backend_type", [BackendType.SQLITEVEC, BackendType.DUCKDBVSS])
    def test_registration_rejects_half_storage_naming_backend(self, backend_type):
        with pytest.raises(
            UnsupportedIndexConfigError, match=_quoted(VectorPrecision.FLOAT16, backend_type)
        ):
            _declared(backend_type).validate_registration(
                dimensions=768, storage_precision=VectorPrecision.FLOAT16
            )

    def test_registration_rejects_too_many_dimensions(self):
        with pytest.raises(UnsupportedIndexConfigError, match="at most 4,000"):
            _declared(BackendType.PGVECTOR).validate_registration(
                dimensions=4001, storage_precision=VectorPrecision.FLOAT16
            )

    def test_registration_has_no_dimension_limit_without_a_declared_one(self):
        _declared(BackendType.DUCKDBVSS).validate_registration(
            dimensions=100_000, storage_precision=VectorPrecision.FLOAT32
        )

    def test_index_rejects_half_precision_where_unsupported(self):
        with pytest.raises(
            UnsupportedIndexConfigError,
            match=_quoted(VectorPrecision.FLOAT16, IndexType.HNSW, BackendType.DUCKDBVSS),
        ):
            _validate(
                BackendType.DUCKDBVSS,
                HNSWIndexConfig(
                    metric_type=MetricType.COSINE, index_precision=VectorPrecision.FLOAT16
                ),
            )

    @pytest.mark.parametrize("backend_type", [BackendType.SQLITEVEC, BackendType.PGVECTOR])
    def test_flat_index_rejects_an_explicit_precision(self, backend_type):
        with pytest.raises(
            UnsupportedIndexConfigError,
            match=_quoted(VectorPrecision.FLOAT16, IndexType.FLAT, backend_type),
        ):
            _validate(backend_type, FlatIndexConfig(index_precision=VectorPrecision.FLOAT16))

    def test_index_rejects_float32_over_a_half_column(self):
        with pytest.raises(UnsupportedIndexConfigError, match="needs a float32 column"):
            _validate(
                BackendType.PGVECTOR,
                HNSWIndexConfig(
                    metric_type=MetricType.COSINE, index_precision=VectorPrecision.FLOAT32
                ),
                storage_precision=VectorPrecision.FLOAT16,
            )

    def test_wide_float32_column_needs_a_half_index_and_the_error_says_so(self):
        config = HNSWIndexConfig(metric_type=MetricType.COSINE)
        with pytest.raises(
            UnsupportedIndexConfigError,
            match=f"at most 2,000.*index precision {_quoted(VectorPrecision.FLOAT16)}",
        ):
            _validate(BackendType.PGVECTOR, config, dimensions=3000)

    def test_wide_float32_column_accepts_a_half_index(self):
        _validate(
            BackendType.PGVECTOR,
            HNSWIndexConfig(metric_type=MetricType.COSINE, index_precision=VectorPrecision.FLOAT16),
            dimensions=3000,
        )

    def test_wide_half_column_indexes_without_an_option(self):
        _validate(
            BackendType.PGVECTOR,
            HNSWIndexConfig(metric_type=MetricType.COSINE),
            dimensions=3000,
            storage_precision=VectorPrecision.FLOAT16,
        )

    def test_index_beyond_every_precision_has_no_hint(self):
        with pytest.raises(UnsupportedIndexConfigError, match="at most 4,000") as excinfo:
            _validate(
                BackendType.PGVECTOR,
                HNSWIndexConfig(
                    metric_type=MetricType.COSINE, index_precision=VectorPrecision.FLOAT16
                ),
                dimensions=5000,
            )
        assert "Use index precision" not in str(excinfo.value)

    def test_flat_ignores_index_limits(self):
        _validate(BackendType.PGVECTOR, FlatIndexConfig(), dimensions=3500)


@pytest.mark.unit
class TestIndexLookup:
    def test_unsupported_index_type_is_rejected_naming_backend(self):
        with pytest.raises(
            UnsupportedIndexConfigError, match=_quoted(IndexType.HNSW, BackendType.SQLITEVEC)
        ):
            _declared(BackendType.SQLITEVEC).index(IndexType.HNSW)

    def test_scan_cap_requires_an_index_that_supports_it(self):
        _declared(BackendType.PGVECTOR).require_max_scan_tuples()
        with pytest.raises(UnsupportedIndexConfigError, match="max_scan_tuples"):
            _declared(BackendType.DUCKDBVSS).require_max_scan_tuples()
