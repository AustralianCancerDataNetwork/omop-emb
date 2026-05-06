"""Tests for the SQLiteVec embedding backend.

In-memory SQLite — no external service required.
"""

from __future__ import annotations

import numpy as np
import pytest

from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.backends.sqlitevec import SQLiteVecBackend
from omop_emb.config import MetricType

from .conftest import (
    CONCEPT_EMBEDDINGS,
    CONCEPT_RECORDS,
    EMBEDDING_DIM,
    MODEL_NAME,
    PROVIDER_TYPE,
    QUERY_EMBEDDING,
)
from .shared_backend_tests import SharedBackendTests


@pytest.mark.unit
class TestSQLiteVecBackend(SharedBackendTests):
    """Runs the full shared suite against an in-memory SQLiteVecBackend."""

    @pytest.fixture
    def backend(self, svec_backend: SQLiteVecBackend):
        return svec_backend


@pytest.mark.unit
class TestSQLiteVecSpecific:
    """SQLiteVec-specific behaviour not covered by the shared suite."""

    def test_hnsw_registration_raises(self, svec_backend: SQLiteVecBackend):
        with pytest.raises(ValueError, match="Only FLAT index is allowed at registration"):
            svec_backend.register_model(
                model_name=MODEL_NAME,
                provider_type=PROVIDER_TYPE,
                index_config=HNSWIndexConfig(metric_type=MetricType.COSINE),
                dimensions=EMBEDDING_DIM,
            )

    def test_flat_registration_has_no_metric(self, svec_backend: SQLiteVecBackend):
        """FLAT-registered models have metric_type=None — metric is supplied at query time."""
        record = svec_backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(),
            dimensions=EMBEDDING_DIM,
        )
        assert record.metric_type is None

    def test_flat_model_accepts_cosine_metric_at_query_time(self, svec_backend: SQLiteVecBackend):
        """FLAT models accept any metric at query time; sqlite-vec uses L2 internally.

        The vec0 table is created with L2 (FLAT default). Querying with COSINE is
        valid per the decorator, but distances are L2-based. The ordering is still
        correct: [-10] is closer to query [-1] than [+10] under both metrics.
        """
        from omop_emb.backends.base_backend import ConceptEmbeddingRecord

        nonzero_records = [
            ConceptEmbeddingRecord(concept_id=1, domain_id="Condition", vocabulary_id="SNOMED", is_standard=True),
            ConceptEmbeddingRecord(concept_id=3, domain_id="Drug", vocabulary_id="RxNorm", is_standard=True),
        ]
        nonzero_embeddings = np.array([[-10.0], [10.0]], dtype=np.float32)

        svec_backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(),
            dimensions=EMBEDDING_DIM,
        )
        svec_backend.upsert_embeddings(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            metric_type=MetricType.COSINE,
            records=nonzero_records,
            embeddings=nonzero_embeddings,
        )
        results = svec_backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            metric_type=MetricType.COSINE,
            query_embeddings=QUERY_EMBEDDING,
            k=2,
        )
        concept_ids_in_order = [r.concept_id for r in results[0]]
        # [-10] is closer to query [-1] than [+10] under both L2 and cosine
        assert concept_ids_in_order[0] == 1   # Hypertension
        assert concept_ids_in_order[1] == 3   # Aspirin
        # Similarities are in valid range
        for match in results[0]:
            assert 0.0 <= match.similarity <= 1.0

    def test_one_table_per_model(self, svec_backend: SQLiteVecBackend):
        """One row and one physical table per model — metric is not part of the key."""
        r1 = svec_backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(),
            dimensions=EMBEDDING_DIM,
        )
        r2 = svec_backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(),
            dimensions=EMBEDDING_DIM,
        )
        assert r1.storage_identifier == r2.storage_identifier
        assert len(svec_backend.get_registered_models(model_name=MODEL_NAME)) == 1

    def test_from_path_constructor(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        backend = SQLiteVecBackend.from_path(db_file)
        record = backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(),
            dimensions=EMBEDDING_DIM,
        )
        assert record.model_name == MODEL_NAME
        backend.emb_engine.dispose()
