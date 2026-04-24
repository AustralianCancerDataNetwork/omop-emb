"""Unit tests for FaissBaseIndexManager and its concrete subclasses.

These tests run entirely in-process with temporary directories — no database
or PostgreSQL connection required.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from omop_emb.config import IndexType, MetricType
from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.backends.faiss.faiss_index_manager import (
    FaissBaseIndexManager,
    FaissFlatIndexManager,
    FaissHNSWIndexManager,
)


DIM = 4
N = 10
K = 3

def _random_vecs(n: int = N, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((n, dim), dtype=np.float32)

def _random_ids(n: int = N) -> np.ndarray:
    return np.arange(1, n + 1, dtype=np.int64)


# ---------------------------------------------------------------------------
# FaissFlatIndexManager
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFaissFlatIndexManager:

    @pytest.fixture
    def manager(self, tmp_path: Path) -> FaissFlatIndexManager:
        return FaissFlatIndexManager(
            dimension=DIM,
            base_index_dir=tmp_path,
            index_config=FlatIndexConfig(),
        )

    def test_supported_index_type(self, manager):
        assert manager.supported_index_type == IndexType.FLAT

    def test_has_index_false_before_create(self, manager):
        assert not manager.has_index(MetricType.L2)
        assert not manager.has_index(MetricType.COSINE)

    def test_create_index_l2(self, manager):
        manager.create_index(MetricType.L2)
        assert manager.has_index(MetricType.L2)
        assert not manager.has_index(MetricType.COSINE)

    def test_create_index_is_idempotent(self, manager):
        manager.create_index(MetricType.L2)
        manager.create_index(MetricType.L2)  # must not raise

    def test_add_and_search_l2(self, manager):
        ids = _random_ids()
        vecs = _random_vecs()
        manager.create_index(MetricType.L2)
        manager.add(ids, vecs, MetricType.L2)

        query = vecs[:1]
        distances, returned_ids = manager.search(query, MetricType.L2, k=K)
        assert distances.shape == (1, K)
        assert returned_ids.shape == (1, K)
        assert returned_ids[0, 0] == ids[0]

    def test_add_and_search_cosine(self, manager):
        ids = _random_ids()
        vecs = _random_vecs()
        manager.create_index(MetricType.COSINE)
        manager.add(ids, vecs, MetricType.COSINE)

        query = vecs[:1]
        distances, returned_ids = manager.search(query, MetricType.COSINE, k=1)
        assert returned_ids[0, 0] == ids[0]

    def test_unsupported_metric_raises(self, manager):
        with pytest.raises(ValueError, match="Unsupported metric"):
            manager.create_index(MetricType.L1)

    def test_save_and_load_roundtrip(self, manager, tmp_path):
        ids = _random_ids()
        vecs = _random_vecs()
        manager.create_index(MetricType.L2)
        manager.add(ids, vecs, MetricType.L2)
        manager.save(MetricType.L2)

        assert manager.index_filepath(MetricType.L2).exists()

        fresh = FaissFlatIndexManager(
            dimension=DIM,
            base_index_dir=tmp_path,
            index_config=FlatIndexConfig(),
        )
        fresh.load(MetricType.L2)
        _, returned_ids = fresh.search(vecs[:1], MetricType.L2, k=1)
        assert returned_ids[0, 0] == ids[0]

    def test_drop_index_removes_file(self, manager, tmp_path):
        ids = _random_ids()
        vecs = _random_vecs()
        manager.create_index(MetricType.L2)
        manager.add(ids, vecs, MetricType.L2)
        manager.save(MetricType.L2)
        assert manager.has_index(MetricType.L2)

        manager.drop_index(MetricType.L2)
        assert not manager.has_index(MetricType.L2)
        assert not manager.index_filepath(MetricType.L2).exists()

    def test_drop_index_is_idempotent(self, manager):
        manager.drop_index(MetricType.L2)  # nothing exists yet — must not raise

    def test_load_or_create_builds_from_stream(self, manager):
        ids = _random_ids()
        vecs = _random_vecs()

        def stream():
            yield ids, vecs

        manager.load_or_create(MetricType.L2, data_stream=stream(), expected_count=N)
        assert manager.has_index(MetricType.L2)
        _, returned_ids = manager.search(vecs[:1], MetricType.L2, k=1)
        assert returned_ids[0, 0] == ids[0]

    def test_load_or_create_loads_existing(self, manager, tmp_path):
        ids = _random_ids()
        vecs = _random_vecs()
        manager.create_index(MetricType.L2)
        manager.add(ids, vecs, MetricType.L2)
        manager.save(MetricType.L2)

        fresh = FaissFlatIndexManager(
            dimension=DIM, base_index_dir=tmp_path, index_config=FlatIndexConfig()
        )
        
        fresh.load_or_create(MetricType.L2, data_stream=iter([]), expected_count=N)
        _, returned_ids = fresh.search(vecs[:1], MetricType.L2, k=1)
        assert returned_ids[0, 0] == ids[0]

    def test_load_or_create_no_stream_and_no_file_raises(self, manager):
        with pytest.raises(ValueError, match="data_stream"):
            manager.load_or_create(MetricType.L2)

    def test_rebuild_drops_and_repopulates(self, manager):
        ids = _random_ids()
        vecs = _random_vecs()
        manager.create_index(MetricType.L2)
        manager.add(ids, vecs, MetricType.L2)
        manager.save(MetricType.L2)

        new_ids = np.array([99, 100], dtype=np.int64)
        new_vecs = _random_vecs(2)

        def stream():
            yield new_ids, new_vecs

        manager.rebuild_index(MetricType.L2, data_stream=stream(), expected_count=2)
        _, returned_ids = manager.search(new_vecs[:1], MetricType.L2, k=1)
        assert returned_ids[0, 0] in new_ids

    def test_staleness_warning_emitted(self, manager, tmp_path, caplog):
        ids = _random_ids(5)
        vecs = _random_vecs(5)
        manager.create_index(MetricType.L2)
        manager.add(ids, vecs, MetricType.L2)
        manager.save(MetricType.L2)

        fresh = FaissFlatIndexManager(
            dimension=DIM, base_index_dir=tmp_path, index_config=FlatIndexConfig()
        )
        import logging
        with caplog.at_level(logging.WARNING):
            fresh.load_or_create(MetricType.L2, expected_count=999)  # wrong expected count
        assert "stale" in caplog.text.lower()

    def test_search_on_empty_index_raises(self, manager):
        manager.create_index(MetricType.L2)
        with pytest.raises(RuntimeError, match="no vectors"):
            manager.search(_random_vecs(1), MetricType.L2, k=1)

    def test_subset_concept_id_filter(self, manager):
        ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        vecs = _random_vecs(5)
        manager.create_index(MetricType.L2)
        manager.add(ids, vecs, MetricType.L2)

        allowed = np.array([3, 4, 5], dtype=np.int64)
        _, returned_ids = manager.search(vecs[:1], MetricType.L2, k=3, subset_concept_ids=allowed)
        valid = returned_ids[returned_ids != -1]
        assert all(cid in allowed for cid in valid)


# ---------------------------------------------------------------------------
# FaissHNSWIndexManager
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFaissHNSWIndexManager:

    @pytest.fixture
    def config(self) -> HNSWIndexConfig:
        return HNSWIndexConfig(num_neighbors=8, ef_search=16, ef_construction=32)

    @pytest.fixture
    def manager(self, tmp_path: Path, config: HNSWIndexConfig) -> FaissHNSWIndexManager:
        return FaissHNSWIndexManager(
            dimension=DIM,
            base_index_dir=tmp_path,
            index_config=config,
        )

    def test_supported_index_type(self, manager):
        assert manager.supported_index_type == IndexType.HNSW

    def test_wrong_index_config_raises(self, tmp_path):
        with pytest.raises(ValueError, match="index_type"):
            FaissHNSWIndexManager(
                dimension=DIM,
                base_index_dir=tmp_path,
                index_config=FlatIndexConfig(), # type: ignore
            )

    def test_add_and_search_l2(self, manager):
        ids = _random_ids()
        vecs = _random_vecs()
        manager.create_index(MetricType.L2)
        manager.add(ids, vecs, MetricType.L2)

        _, returned_ids = manager.search(vecs[:1], MetricType.L2, k=1)
        assert returned_ids[0, 0] == ids[0]

    def test_add_and_search_cosine(self, manager):
        ids = _random_ids()
        vecs = _random_vecs()
        manager.create_index(MetricType.COSINE)
        manager.add(ids, vecs, MetricType.COSINE)

        _, returned_ids = manager.search(vecs[:1], MetricType.COSINE, k=1)
        assert returned_ids[0, 0] == ids[0]

    def test_ef_search_applied_in_params(self, manager, config):
        params = manager._create_search_parameters()
        assert params.efSearch == config.ef_search

    def test_ef_construction_applied_on_index_creation(self, manager, config):
        import faiss
        manager.create_index(MetricType.L2)
        inner = manager._metric_to_index_cache[MetricType.L2]
        # IndexIDMap wraps the raw HNSW — unwrap to verify
        raw = faiss.downcast_index(inner.index if hasattr(inner, "index") else inner)
        assert raw.hnsw.efConstruction == config.ef_construction

    def test_num_neighbors_applied_on_index_creation(self, manager, config):
        import faiss
        manager.create_index(MetricType.L2)
        inner = manager._metric_to_index_cache[MetricType.L2]
        raw = faiss.downcast_index(inner.index if hasattr(inner, "index") else inner)
        assert raw.hnsw.nb_neighbors(1) == config.num_neighbors

    def test_unsupported_metric_raises(self, manager):
        with pytest.raises(ValueError, match="Unsupported metric"):
            manager.create_index(MetricType.L1)

    def test_save_load_roundtrip(self, manager, tmp_path, config):
        ids = _random_ids()
        vecs = _random_vecs()
        manager.create_index(MetricType.L2)
        manager.add(ids, vecs, MetricType.L2)
        manager.save(MetricType.L2)

        fresh = FaissHNSWIndexManager(
            dimension=DIM, base_index_dir=tmp_path, index_config=config
        )
        fresh.load(MetricType.L2)
        _, returned_ids = fresh.search(vecs[:1], MetricType.L2, k=1)
        assert returned_ids[0, 0] == ids[0]

    def test_subset_filter_with_hnsw(self, manager):
        ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        vecs = _random_vecs(5)
        manager.create_index(MetricType.L2)
        manager.add(ids, vecs, MetricType.L2)

        allowed = np.array([30, 40, 50], dtype=np.int64)
        _, returned_ids = manager.search(vecs[:1], MetricType.L2, k=3, subset_concept_ids=allowed)
        valid = returned_ids[returned_ids != -1]
        assert all(cid in allowed for cid in valid)

    def test_multi_metric_independence(self, manager):
        ids = _random_ids()
        vecs = _random_vecs()
        manager.create_index(MetricType.L2)
        manager.add(ids, vecs, MetricType.L2)
        assert manager.has_index(MetricType.L2)
        assert not manager.has_index(MetricType.COSINE)

    def test_rebuild_clears_and_repopulates(self, manager):
        ids = _random_ids()
        vecs = _random_vecs()
        manager.create_index(MetricType.L2)
        manager.add(ids, vecs, MetricType.L2)
        manager.save(MetricType.L2)

        new_ids = np.array([101, 102], dtype=np.int64)
        new_vecs = _random_vecs(2)

        def stream():
            yield new_ids, new_vecs

        manager.rebuild_index(MetricType.L2, data_stream=stream(), expected_count=2)
        _, returned_ids = manager.search(new_vecs[:1], MetricType.L2, k=1)
        assert returned_ids[0, 0] in new_ids
