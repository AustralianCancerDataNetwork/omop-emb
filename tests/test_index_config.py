"""Unit tests for IndexConfig.from_kwargs."""

from __future__ import annotations

import json

import pytest

from omop_emb.backends.index_config import (
    FlatIndexConfig,
    HNSWIndexConfig,
    index_config_from_dict,
    index_config_from_index_type,
)
from omop_emb.config import IndexType, MetricType, VectorPrecision


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
    "name",
    ["num_neighbors", "ef_search", "ef_construction", "max_scan_tuples"],
)
def test_hnsw_config_rejects_a_non_positive_value(name):
    with pytest.raises(ValueError, match=name):
        HNSWIndexConfig.from_kwargs(metric_type=MetricType.COSINE, **{name: 0})


@pytest.mark.parametrize(
    "config", [FlatIndexConfig(), HNSWIndexConfig(metric_type=MetricType.COSINE)]
)
def test_index_precision_defaults_to_following_the_stored_column(config):
    assert config.index_precision is None


def test_index_precision_survives_a_json_round_trip():
    config = HNSWIndexConfig(metric_type=MetricType.COSINE, index_precision=VectorPrecision.FLOAT16)
    stored = json.loads(json.dumps(config.to_dict()))
    assert index_config_from_dict(IndexType.HNSW, stored) == config


def test_index_precision_is_read_from_its_stored_string_form():
    stored = {"index_type": "hnsw", "metric_type": "cosine", "index_precision": "float16"}
    config = index_config_from_dict(IndexType.HNSW, stored)
    assert config == HNSWIndexConfig(
        metric_type=MetricType.COSINE, index_precision=VectorPrecision.FLOAT16
    )


def test_flat_config_carries_an_index_precision_for_capabilities_to_reject():
    config = index_config_from_index_type(IndexType.FLAT, index_precision=VectorPrecision.FLOAT16)
    assert config.index_precision == VectorPrecision.FLOAT16


@pytest.mark.parametrize(
    ("storage", "index_precision", "effective", "casts"),
    [
        (VectorPrecision.FLOAT32, None, VectorPrecision.FLOAT32, False),
        (VectorPrecision.FLOAT16, None, VectorPrecision.FLOAT16, False),
        (VectorPrecision.FLOAT32, VectorPrecision.FLOAT32, VectorPrecision.FLOAT32, False),
        (VectorPrecision.FLOAT16, VectorPrecision.FLOAT16, VectorPrecision.FLOAT16, False),
        (VectorPrecision.FLOAT32, VectorPrecision.FLOAT16, VectorPrecision.FLOAT16, True),
    ],
)
def test_effective_precision_and_cast_follow_the_stored_column(
    storage, index_precision, effective, casts
):
    config = HNSWIndexConfig(metric_type=MetricType.COSINE, index_precision=index_precision)
    assert config.effective_precision(storage) == effective
    assert config.casts_storage(storage) is casts


def test_flat_config_never_casts():
    config = FlatIndexConfig()
    assert not config.casts_storage(VectorPrecision.FLOAT32)
    assert not config.casts_storage(VectorPrecision.FLOAT16)


def test_hnsw_record_stored_before_index_precision_existed_still_loads():
    stored = {
        "index_type": "hnsw",
        "metric_type": "cosine",
        "num_neighbors": 16,
        "ef_search": 40,
        "ef_construction": 64,
        "max_scan_tuples": None,
    }
    assert HNSWIndexConfig.from_dict(stored).index_precision is None


def _hnsw(**overrides):
    return HNSWIndexConfig(metric_type=MetricType.COSINE, **overrides)


def test_same_build_ignores_query_time_settings():
    stored = _hnsw(ef_search=300, max_scan_tuples=50_000)
    assert stored.same_build(_hnsw(), VectorPrecision.FLOAT32)


@pytest.mark.parametrize("change", [{"num_neighbors": 16}, {"ef_construction": 64}])
def test_same_build_sees_graph_settings(change):
    assert not _hnsw().same_build(_hnsw(**change), VectorPrecision.FLOAT32)


def test_same_build_sees_the_distance_metric_and_index_type():
    assert not _hnsw().same_build(
        HNSWIndexConfig(metric_type=MetricType.L2), VectorPrecision.FLOAT32
    )
    assert not _hnsw().same_build(FlatIndexConfig(), VectorPrecision.FLOAT32)


def test_same_build_compares_the_effective_index_precision():
    unset = _hnsw()
    assert unset.same_build(_hnsw(index_precision=VectorPrecision.FLOAT32), VectorPrecision.FLOAT32)
    assert not unset.same_build(
        _hnsw(index_precision=VectorPrecision.FLOAT16), VectorPrecision.FLOAT32
    )
    assert unset.same_build(_hnsw(index_precision=VectorPrecision.FLOAT16), VectorPrecision.FLOAT16)


def test_factory_keeps_precision_and_query_settings_of_the_current_hnsw_index():
    current = _hnsw(ef_search=300, max_scan_tuples=50_000, index_precision=VectorPrecision.FLOAT16)
    config = index_config_from_index_type(
        IndexType.HNSW, current=current, metric_type=MetricType.COSINE, ef_search=None
    )
    assert config == current


def test_factory_prefers_a_value_that_is_given():
    current = _hnsw(ef_search=300, index_precision=VectorPrecision.FLOAT16)
    config = index_config_from_index_type(
        IndexType.HNSW,
        current=current,
        metric_type=MetricType.COSINE,
        ef_search=100,
        index_precision=VectorPrecision.FLOAT32,
    )
    assert config == _hnsw(ef_search=100, index_precision=VectorPrecision.FLOAT32)


def test_factory_keeps_the_graph_settings_of_the_current_index():
    current = _hnsw(num_neighbors=16, ef_construction=64)
    config = index_config_from_index_type(
        IndexType.HNSW, current=current, metric_type=MetricType.COSINE
    )
    assert config == current


def test_factory_takes_the_defaults_without_a_current_hnsw_index():
    config = index_config_from_index_type(IndexType.HNSW, metric_type=MetricType.COSINE)
    assert config == _hnsw()


def test_factory_keeps_nothing_across_index_types():
    current = _hnsw(ef_search=300, index_precision=VectorPrecision.FLOAT16)
    flat = index_config_from_index_type(IndexType.FLAT, current=current)
    assert flat.index_precision is None
    hnsw = index_config_from_index_type(
        IndexType.HNSW, current=FlatIndexConfig(), metric_type=MetricType.COSINE
    )
    assert hnsw == _hnsw()


def test_partial_indexes_are_off_by_default():
    config = HNSWIndexConfig(metric_type=MetricType.COSINE)
    assert config.partial_domain_indexes is False
    assert config.partial_valid_only is False


def test_partial_index_settings_survive_a_dict_round_trip():
    config = HNSWIndexConfig(
        metric_type=MetricType.COSINE, partial_domain_indexes=True, partial_valid_only=True
    )
    assert index_config_from_dict(IndexType.HNSW, config.to_dict()) == config


def test_hnsw_record_stored_before_partial_indexes_existed_still_loads():
    stored = {
        "index_type": "hnsw",
        "metric_type": "cosine",
        "num_neighbors": 16,
        "ef_search": 40,
        "ef_construction": 64,
        "max_scan_tuples": None,
    }
    config = HNSWIndexConfig.from_dict(stored)
    assert config.partial_domain_indexes is False
    assert config.partial_valid_only is False


def test_valid_only_partial_indexes_need_partial_indexes():
    with pytest.raises(ValueError, match="partial_valid_only.*partial_domain_indexes"):
        HNSWIndexConfig(metric_type=MetricType.COSINE, partial_valid_only=True)
