"""Tests for the SQLiteVec embedding backend.

In-memory SQLite: no external service required.
"""

from __future__ import annotations

import numpy as np
import pytest
import sqlalchemy as sa

from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.backends.sqlitevec import SQLiteVecEmbeddingBackend
from omop_emb.backends.sqlitevec.sqlitevec_backend import _load_sqlite_vec
from omop_emb.config import MetricType
from omop_emb.model_registry import RegistryManager, registry_writer_engine
from omop_emb.model_registry.model_registry_orm import ModelRegistry

from .conftest import (
    CONCEPT_RECORDS,
    EMBEDDING_DIM,
    MODEL_NAME,
    PROVIDER_TYPE,
    QUERY_EMBEDDING,
    sqlite_resolved_database,
)
from .shared_backend_tests import SharedBackendTests


@pytest.mark.unit
class TestSQLiteVecEmbeddingBackend(SharedBackendTests):
    """Runs the full shared suite against an in-memory SQLiteVecEmbeddingBackend."""

    @pytest.fixture
    def backend(self, svec_backend: SQLiteVecEmbeddingBackend):
        return svec_backend


@pytest.mark.unit
class TestSQLiteVecSpecific:
    """SQLiteVec-specific behaviour not covered by the shared suite."""

    def test_hnsw_registration_raises(self, svec_backend: SQLiteVecEmbeddingBackend):
        with pytest.raises(
            ValueError, match="Only FLAT index is allowed at registration"
        ):
            svec_backend.register_model(
                model_name=MODEL_NAME,
                provider_type=PROVIDER_TYPE,
                index_config=HNSWIndexConfig(metric_type=MetricType.COSINE),
                dimensions=EMBEDDING_DIM,
            )

    def test_flat_registration_has_no_metric(
        self, svec_backend: SQLiteVecEmbeddingBackend
    ):
        """FLAT-registered models have metric_type=None: metric is supplied at query time."""
        record = svec_backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(),
            dimensions=EMBEDDING_DIM,
        )
        assert record.metric_type is None

    def test_flat_model_accepts_cosine_metric_at_query_time(
        self, svec_backend: SQLiteVecEmbeddingBackend
    ):
        """FLAT models accept any metric at query time; sqlite-vec uses L2 internally.

        The vec0 table is created with L2 (FLAT default). Querying with COSINE is
        valid per the decorator, but distances are L2-based. The ordering is still
        correct: [-10] is closer to query [-1] than [+10] under both metrics.
        """
        from omop_emb.backends.base_backend import ConceptEmbeddingRecord

        nonzero_records = [
            ConceptEmbeddingRecord(
                concept_id=1,
                domain_id="Condition",
                vocabulary_id="SNOMED",
                is_standard=True,
                is_valid=True,
            ),
            ConceptEmbeddingRecord(
                concept_id=3,
                domain_id="Drug",
                vocabulary_id="RxNorm",
                is_standard=True,
                is_valid=True,
            ),
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
            records=nonzero_records,
            embeddings=nonzero_embeddings,
        )
        results = svec_backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.COSINE,
            query_embeddings=QUERY_EMBEDDING,
            k=2,
        )
        concept_ids_in_order = [r.concept_id for r in results[0]]
        # [-10] is closer to query [-1] than [+10] under both L2 and cosine
        assert concept_ids_in_order[0] == 1  # Hypertension
        assert concept_ids_in_order[1] == 3  # Aspirin
        # Similarities are in valid range
        for match in results[0]:
            assert 0.0 <= match.similarity <= 1.0

    def test_one_table_per_model(self, svec_backend: SQLiteVecEmbeddingBackend):
        """One row and one physical table per model: metric is not part of the key."""
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

    def test_existing_old_style_storage_identifier_keeps_working(
        self, svec_backend: SQLiteVecEmbeddingBackend, monkeypatch
    ):
        model_name = "old-style-model"
        storage_identifier = "emb_old_style_model"
        vector = np.linspace(0.0, 1.0, EMBEDDING_DIM, dtype=np.float32).reshape(1, -1)
        with svec_backend._registry.emb_session_factory.begin() as session:
            session.add(
                ModelRegistry(
                    database_config_name=svec_backend._registry._database_config_name,
                    model_name=model_name,
                    provider_type=PROVIDER_TYPE,
                    storage_identifier=storage_identifier,
                    dimensions=EMBEDDING_DIM,
                    index_config=FlatIndexConfig(),
                    details={},
                )
            )

        record = svec_backend.get_registered_model(model_name=model_name)
        assert record is not None
        assert record.storage_identifier == storage_identifier
        svec_backend._ensure_storage_table(record)
        monkeypatch.setattr(
            RegistryManager,
            "storage_name",
            staticmethod(lambda *_args: pytest.fail("existing row identifier was recomputed")),
        )
        same_record = svec_backend.register_model(
            model_name=model_name,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(),
            dimensions=EMBEDDING_DIM,
        )
        assert same_record.storage_identifier == storage_identifier
        svec_backend.upsert_embeddings(
            model_name=model_name, records=CONCEPT_RECORDS[:1], embeddings=vector
        )
        found = svec_backend.get_embeddings_by_concept_ids(
            model_name=model_name, concept_ids=[CONCEPT_RECORDS[0].concept_id]
        )
        assert np.allclose(found[CONCEPT_RECORDS[0].concept_id], vector[0])

        svec_backend.delete_model(model_name=model_name)
        assert svec_backend.get_registered_model(model_name=model_name) is None
        assert not sa.inspect(svec_backend.emb_engine).has_table(storage_identifier)

    def test_file_backed_engine_constructor(self, tmp_path):
        """A real file-backed (not just in-memory) engine works end to end."""
        db_file = str(tmp_path / "test.db")
        engine = registry_writer_engine(
            sqlite_resolved_database(db_file), extensions=[_load_sqlite_vec]
        )
        backend = SQLiteVecEmbeddingBackend(emb_engine=engine)
        record = backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(),
            dimensions=EMBEDDING_DIM,
        )
        assert record.model_name == MODEL_NAME
        backend.emb_engine.dispose()
