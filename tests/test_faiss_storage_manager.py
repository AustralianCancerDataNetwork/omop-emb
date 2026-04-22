"""Unit tests for FAISS storage manager rebuild behavior."""

from __future__ import annotations

import pytest

from omop_emb.config import BackendType, IndexType, MetricType
from omop_emb.backends.faiss.storage_manager import EmbeddingStorageManager


@pytest.mark.unit
def test_rebuild_index_does_not_call_load_or_populate(monkeypatch, tmp_path):
    storage_manager = EmbeddingStorageManager(
        file_dir=tmp_path,
        dimensions=16,
        backend_type=BackendType.FAISS,
    )

    events: list[str] = []

    class FakeIndexManager:
        def load_or_populate(self, data_stream):
            events.append("load_or_populate")

        def rebuild_from_storage(self, data_stream):
            events.append("rebuild_from_storage")

    monkeypatch.setattr(
        storage_manager,
        "_instantiate_index_manager",
        lambda **kwargs: FakeIndexManager(),
    )

    storage_manager.rebuild_index(
        index_type=IndexType.HNSW,
        metric_type=MetricType.COSINE,
        batch_size=10,
    )

    assert events == ["rebuild_from_storage"]
