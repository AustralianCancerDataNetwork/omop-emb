"""Tests for omop_emb.config's per-backend index/metric support accessors."""

from __future__ import annotations

import pytest

from omop_emb.config import (
    BackendType,
    IndexType,
    MetricType,
    SUPPORTED_INDICES_AND_METRICS_PER_BACKEND,
    get_supported_index_types_for_backend,
    get_supported_metrics_for_backend,
    is_index_type_supported_for_backend,
    is_supported_index_metric_combination_for_backend,
)


@pytest.mark.unit
class TestUnregisteredBackendRaises:
    """An unregistered backend must raise, not silently report 'supports
    nothing'. Simulated via monkeypatch since BackendType has no member
    left unregistered today."""

    def test_is_supported_index_metric_combination_for_backend(self, monkeypatch):
        monkeypatch.delitem(SUPPORTED_INDICES_AND_METRICS_PER_BACKEND, BackendType.SQLITEVEC)
        with pytest.raises(ValueError, match="Unsupported backend"):
            is_supported_index_metric_combination_for_backend(
                BackendType.SQLITEVEC, IndexType.FLAT, MetricType.L2
            )

    def test_is_index_type_supported_for_backend(self, monkeypatch):
        monkeypatch.delitem(SUPPORTED_INDICES_AND_METRICS_PER_BACKEND, BackendType.SQLITEVEC)
        with pytest.raises(ValueError, match="Unsupported backend"):
            is_index_type_supported_for_backend(BackendType.SQLITEVEC, IndexType.FLAT)

    def test_get_supported_index_types_for_backend(self, monkeypatch):
        monkeypatch.delitem(SUPPORTED_INDICES_AND_METRICS_PER_BACKEND, BackendType.SQLITEVEC)
        with pytest.raises(ValueError, match="Unsupported backend"):
            get_supported_index_types_for_backend(BackendType.SQLITEVEC)

    def test_get_supported_metrics_for_backend(self, monkeypatch):
        monkeypatch.delitem(SUPPORTED_INDICES_AND_METRICS_PER_BACKEND, BackendType.SQLITEVEC)
        with pytest.raises(ValueError, match="Unsupported backend"):
            get_supported_metrics_for_backend(BackendType.SQLITEVEC)
