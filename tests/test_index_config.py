"""Unit tests for IndexConfig.from_kwargs."""

from __future__ import annotations

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
    assert cfg.ef_search == 16
    assert cfg.ef_construction == 64


def test_from_kwargs_applies_explicit_non_none_values():
    cfg = HNSWIndexConfig.from_kwargs(metric_type=MetricType.COSINE, num_neighbors=64, ef_search=32)
    assert cfg.num_neighbors == 64
    assert cfg.ef_search == 32
    assert cfg.ef_construction == 64


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
