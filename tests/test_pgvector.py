"""Integration tests for the pgvector embedding backend.

Requires a running PostgreSQL instance with the pgvector extension.
Set TEST_DB_HOST and TEST_DB_PORT to enable these tests.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pgvector", reason="omop-emb[pgvector] not installed — skipping pgvector tests")

from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.backends.pgvector import PGVectorEmbeddingBackend
from omop_emb.config import IndexType, MetricType

from .conftest import (
    CONCEPT_EMBEDDINGS,
    CONCEPT_RECORDS,
    EMBEDDING_DIM,
    HYPERTENSION_ID,
    MODEL_NAME,
    PROVIDER_TYPE,
    QUERY_EMBEDDING,
)
from .shared_backend_tests import SharedBackendTests


@pytest.mark.pgvector
@pytest.mark.integration
class TestPGVectorBackend(SharedBackendTests):
    """Runs the full shared suite against a pgvector backend."""

    @pytest.fixture
    def backend(self, pg_backend: PGVectorEmbeddingBackend):
        return pg_backend


@pytest.mark.pgvector
@pytest.mark.integration
class TestPGVectorHNSWBackend:
    """pgvector-specific HNSW index behaviour."""

    HNSW_CONFIG = HNSWIndexConfig(
        metric_type=MetricType.L2, num_neighbors=4, ef_search=8, ef_construction=16
    )

    def _register_and_upsert(self, backend, *, metric_type=MetricType.L2):
        """Register with FLAT (required at ingestion time) then upsert data."""
        backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(),
            dimensions=EMBEDDING_DIM,
        )
        backend.upsert_embeddings(
            model_name=MODEL_NAME,
            metric_type=metric_type,
            records=list(CONCEPT_RECORDS),
            embeddings=CONCEPT_EMBEDDINGS,
        )

    def test_rebuild_index_keeps_storage_identifier(self, pg_backend: PGVectorEmbeddingBackend):
        """Rebuilding from FLAT to HNSW keeps the same physical table name."""
        r_flat = pg_backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(),
            dimensions=EMBEDDING_DIM,
        )
        r_hnsw = pg_backend.rebuild_index(
            model_name=MODEL_NAME,
            index_config=self.HNSW_CONFIG,
        )
        assert r_flat.storage_identifier == r_hnsw.storage_identifier
        assert r_hnsw.index_type == IndexType.HNSW
        assert r_hnsw.metric_type == MetricType.L2

    def test_hnsw_registration_creates_index_manager(self, pg_backend: PGVectorEmbeddingBackend):
        """Rebuilding to HNSW after FLAT registration yields an HNSW index manager."""
        pg_backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(),
            dimensions=EMBEDDING_DIM,
        )
        pg_backend.rebuild_index(
            model_name=MODEL_NAME,
            index_config=self.HNSW_CONFIG,
        )
        from omop_emb.backends.pgvector.pg_index_manager import PGVectorHNSWIndexManager
        record = pg_backend.get_registered_model(model_name=MODEL_NAME)
        assert record is not None, "Model record should exist after registration and rebuild"
        mgr = pg_backend.get_index_manager(record.storage_identifier)
        assert isinstance(mgr, PGVectorHNSWIndexManager), f"Index manager should be PGVectorHNSWIndexManager after rebuilding to HNSW. Got: {type(mgr)}"

    def test_hnsw_search_returns_correct_top1(self, pg_backend: PGVectorEmbeddingBackend):
        self._register_and_upsert(pg_backend)
        pg_backend.rebuild_index(
            model_name=MODEL_NAME,
            index_config=self.HNSW_CONFIG,
        )
        results = pg_backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.L2,
            query_embeddings=np.array([[-10.0]], dtype=np.float32),
            k=1,
        )
        assert results[0][0].concept_id == HYPERTENSION_ID

    def test_rebuild_index(self, pg_backend: PGVectorEmbeddingBackend):
        """FLAT → HNSW rebuild then search still returns the correct top-1."""
        self._register_and_upsert(pg_backend)
        pg_backend.rebuild_index(
            model_name=MODEL_NAME,
            index_config=self.HNSW_CONFIG,
        )
        results = pg_backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.L2,
            query_embeddings=np.array([[-10.0]], dtype=np.float32),
            k=1,
        )
        assert results[0][0].concept_id == HYPERTENSION_ID
