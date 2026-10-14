"""Tests for the DuckDB VSS embedding backend.

In-memory DuckDB with the vss extension: no external service required.
"""

from __future__ import annotations

import numpy as np
import pytest
from oa_configurator import physical_schema_of, qualified
from sqlalchemy import text

pytest.importorskip(
    "duckdb_sqlalchemy", reason="omop-emb[duckdbvss] not installed: skipping duckdbvss tests"
)

from omop_emb.backends.duckdbvss.duckdbvss_backend import DuckDBVSSEmbeddingBackend
from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.config import IndexType, MetricType

from .conftest import (
    CONCEPT_EMBEDDINGS,
    CONCEPT_RECORDS,
    EMBEDDING_DIM,
    HYPERTENSION_ID,
    MODEL_NAME,
    PROVIDER_TYPE,
)
from .shared_backend_tests import SharedBackendTests


@pytest.mark.duckdbvss
@pytest.mark.db_dialect
class TestDuckDBVSSEmbeddingBackend(SharedBackendTests):
    """Runs the full shared suite against an in-memory DuckDBVSSEmbeddingBackend."""

    @pytest.fixture
    def backend(self, duckdbvss_backend: DuckDBVSSEmbeddingBackend):
        return duckdbvss_backend


@pytest.mark.duckdbvss
@pytest.mark.db_dialect
class TestDuckDBVSSHNSWBackend:
    """DuckDB VSS-specific HNSW index behaviour."""

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
            records=list(CONCEPT_RECORDS),
            embeddings=CONCEPT_EMBEDDINGS,
        )

    def test_rebuild_index_keeps_storage_identifier(
        self, duckdbvss_backend: DuckDBVSSEmbeddingBackend
    ):
        """Rebuilding from FLAT to HNSW keeps the same physical table name."""
        r_flat = duckdbvss_backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(),
            dimensions=EMBEDDING_DIM,
        )
        r_hnsw = duckdbvss_backend.rebuild_index(
            model_name=MODEL_NAME,
            index_config=self.HNSW_CONFIG,
        )
        assert r_flat.storage_identifier == r_hnsw.storage_identifier
        assert r_hnsw.index_type == IndexType.HNSW
        assert r_hnsw.metric_type == MetricType.L2

    def test_rebuild_leaves_exactly_the_configured_index(
        self, duckdbvss_backend: DuckDBVSSEmbeddingBackend
    ):
        """Switching metric replaces the HNSW index; reverting to FLAT drops it."""
        self._register_and_upsert(duckdbvss_backend)
        table = duckdbvss_backend.get_registered_model(model_name=MODEL_NAME).storage_identifier

        duckdbvss_backend.rebuild_index(model_name=MODEL_NAME, index_config=HNSWIndexConfig(metric_type=MetricType.COSINE))
        assert duckdbvss_backend.physical_indexes(MODEL_NAME) == (f"idx_{table}_cosine",)

        duckdbvss_backend.rebuild_index(model_name=MODEL_NAME, index_config=HNSWIndexConfig(metric_type=MetricType.L2))
        assert duckdbvss_backend.physical_indexes(MODEL_NAME) == (f"idx_{table}_l2",)

        duckdbvss_backend.rebuild_index(model_name=MODEL_NAME, index_config=FlatIndexConfig())
        assert duckdbvss_backend.physical_indexes(MODEL_NAME) == ()

    def test_rebuild_rejects_a_metric_the_backend_cannot_index(
        self, duckdbvss_backend: DuckDBVSSEmbeddingBackend
    ):
        self._register_and_upsert(duckdbvss_backend)
        duckdbvss_backend.rebuild_index(model_name=MODEL_NAME, index_config=self.HNSW_CONFIG)
        before = duckdbvss_backend.physical_indexes(MODEL_NAME)
        with pytest.raises(ValueError, match="l1"):
            duckdbvss_backend.rebuild_index(
                model_name=MODEL_NAME, index_config=HNSWIndexConfig(metric_type=MetricType.L1)
            )
        assert duckdbvss_backend.physical_indexes(MODEL_NAME) == before
        assert duckdbvss_backend.get_registered_model(model_name=MODEL_NAME).index_config == self.HNSW_CONFIG

    def test_hnsw_search_returns_correct_top1(
        self, duckdbvss_backend: DuckDBVSSEmbeddingBackend
    ):
        self._register_and_upsert(duckdbvss_backend)
        duckdbvss_backend.rebuild_index(
            model_name=MODEL_NAME,
            index_config=self.HNSW_CONFIG,
        )
        results = duckdbvss_backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.L2,
            query_embeddings=np.array([[-10.0]], dtype=np.float32),
            k=1,
        )
        assert results[0][0].concept_id == HYPERTENSION_ID

    def _knn_uses_hnsw_index(self, duckdbvss_backend: DuckDBVSSEmbeddingBackend, *, table_ref: str) -> bool:
        """EXPLAIN a KNN query shaped like _build_knn_stmt's real one
        (ORDER BY array_distance(...) LIMIT k) and report whether DuckDB's
        planner actually used the HNSW index, rather than trusting that a
        CREATE INDEX call alone guarantees it's used at query time."""
        with duckdbvss_backend.emb_engine.connect() as conn:
            conn.execute(text("SET hnsw_enable_experimental_persistence = true"))
            plan_rows = conn.execute(
                text(f"EXPLAIN SELECT concept_id, array_distance(embedding, [-10.0]::FLOAT[1]) AS distance FROM {table_ref} ORDER BY distance LIMIT 1")
            ).fetchall()
        return any("HNSW Index:" in row[1] for row in plan_rows)

    def test_knn_query_actually_uses_the_hnsw_index(
        self, duckdbvss_backend: DuckDBVSSEmbeddingBackend
    ):
        """Regression guard for a silent fallback to a sequential scan: the
        HNSW index must be the thing DuckDB's own query planner picks, not
        merely present on the table."""
        self._register_and_upsert(duckdbvss_backend)
        record = duckdbvss_backend.get_registered_model(model_name=MODEL_NAME)
        table_ref = qualified(
            duckdbvss_backend.emb_engine, record.storage_identifier, physical_schema=physical_schema_of(duckdbvss_backend.emb_engine)
        )

        assert self._knn_uses_hnsw_index(duckdbvss_backend, table_ref=table_ref) is False

        duckdbvss_backend.rebuild_index(model_name=MODEL_NAME, index_config=self.HNSW_CONFIG)
        assert self._knn_uses_hnsw_index(duckdbvss_backend, table_ref=table_ref) is True

        duckdbvss_backend.rebuild_index(model_name=MODEL_NAME, index_config=FlatIndexConfig())
        assert self._knn_uses_hnsw_index(duckdbvss_backend, table_ref=table_ref) is False
