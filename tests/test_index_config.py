"""Unit tests for IndexConfig.from_kwargs."""

from __future__ import annotations

import pytest

from omop_emb.backends.index_config import (
    FlatIndexConfig,
    HNSWIndexConfig,
    index_config_from_dict,
    index_config_from_index_type,
)
from omop_emb.config import IndexType, MetricType


def test_from_kwargs_ignores_unknown_keys():
    cfg = HNSWIndexConfig.from_kwargs(metric_type=MetricType.COSINE, not_a_real_field=123)
    assert cfg.metric_type == MetricType.COSINE


def test_from_kwargs_keeps_default_when_value_is_none():
    cfg = HNSWIndexConfig.from_kwargs(
        metric_type=MetricType.COSINE,
        num_neighbors=None,
        ef_search=None,
        ef_construction=None,
    )
    assert cfg.num_neighbors == 32
    assert cfg.ef_search == 200
    assert cfg.ef_construction == 200
    assert cfg.max_scan_tuples is None


def test_from_kwargs_applies_explicit_non_none_values():
    cfg = HNSWIndexConfig.from_kwargs(metric_type=MetricType.COSINE, num_neighbors=64, ef_search=32)
    assert cfg.num_neighbors == 64
    assert cfg.ef_search == 32
    assert cfg.ef_construction == 200


def test_flat_config_accepts_a_metric_or_none():
    assert FlatIndexConfig().metric_type is None
    assert FlatIndexConfig(metric_type=MetricType.L2).metric_type == MetricType.L2


def test_flat_config_metric_survives_a_dict_round_trip():
    for config in (FlatIndexConfig(), FlatIndexConfig(metric_type=MetricType.L1)):
        assert index_config_from_dict(IndexType.FLAT, config.to_dict()) == config


def test_index_config_from_index_type_passes_the_flat_metric():
    assert index_config_from_index_type(IndexType.FLAT) == FlatIndexConfig()
    assert index_config_from_index_type(
        IndexType.FLAT, metric_type=MetricType.L2
    ) == FlatIndexConfig(metric_type=MetricType.L2)


def test_hnsw_config_with_max_scan_tuples_survives_a_dict_round_trip():
    config = HNSWIndexConfig(metric_type=MetricType.COSINE, max_scan_tuples=100_000)
    assert index_config_from_dict(IndexType.HNSW, config.to_dict()) == config


def test_hnsw_record_stored_before_max_scan_tuples_existed_still_loads():
    stored = {
        "index_type": "hnsw",
        "metric_type": "cosine",
        "num_neighbors": 32,
        "ef_search": 16,
        "ef_construction": 64,
    }
    config = index_config_from_dict(IndexType.HNSW, stored)
    assert config == HNSWIndexConfig(
        metric_type=MetricType.COSINE, ef_search=16, ef_construction=64, max_scan_tuples=None
    )


def test_hnsw_record_missing_a_required_field_is_rejected():
    with pytest.raises(ValueError, match="metric_type"):
        index_config_from_dict(IndexType.HNSW, {"index_type": "hnsw", "ef_search": 16})


@pytest.mark.parametrize(
    "name", ["num_neighbors", "ef_search", "ef_construction", "max_scan_tuples"]
)
def test_hnsw_config_rejects_a_non_positive_value(name):
    with pytest.raises(ValueError, match=name):
        HNSWIndexConfig(metric_type=MetricType.COSINE, **{name: 0})
