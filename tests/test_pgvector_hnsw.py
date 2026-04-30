"""Integration tests for pgvector HNSW backend.

Requires a running PostgreSQL instance with the pgvector extension.
"""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import Engine

from omop_emb.config import IndexType, MetricType, ProviderType
from omop_emb.backends.pgvector import PGVectorEmbeddingBackend
from omop_emb.backends.index_config import HNSWIndexConfig, FlatIndexConfig
from omop_emb.backends.pgvector.pgvector_index_manager import PGVectorHNSWIndexManager
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter
from .conftest import CONCEPTS, MODEL_NAME, EMBEDDING_DIM, PROVIDER_TYPE


HNSW_CONFIG = HNSWIndexConfig(num_neighbors=4, ef_search=8, ef_construction=16)


@pytest.fixture
def hnsw_pgvector_backend(engine: Engine, temp_storage_dir) -> PGVectorEmbeddingBackend:
    backend = PGVectorEmbeddingBackend(
        omop_cdm_engine=engine,
        storage_base_dir=temp_storage_dir
    )
    return backend

def _register_and_upsert(backend, session, *, index_config=HNSW_CONFIG, metric_type=MetricType.L2):
    backend.register_model(
        engine=session.bind,
        model_name=MODEL_NAME,
        provider_type=PROVIDER_TYPE,
        index_config=index_config,
        dimensions=EMBEDDING_DIM,
    )
    ids = [c.concept_id for c in CONCEPTS.values()]
    vecs = np.vstack([c.embeddings for c in CONCEPTS.values()]).astype(np.float32)
    backend.upsert_embeddings(
        session=session,
        model_name=MODEL_NAME,
        provider_type=PROVIDER_TYPE,
        index_type=index_config.index_type,
        concept_ids=ids,
        embeddings=vecs,
    )
    return ids, vecs


@pytest.mark.pgvector
@pytest.mark.integration
class TestPGVectorHNSWBackend:

    def test_register_creates_manager_not_sql_index(self, session, hnsw_pgvector_backend):
        hnsw_pgvector_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=HNSW_CONFIG,
            dimensions=EMBEDDING_DIM,
        )
        manager = hnsw_pgvector_backend.get_index_manager(MODEL_NAME)
        assert isinstance(manager, PGVectorHNSWIndexManager)
        # SQL index must NOT exist yet — it is created only via initialise_indexes
        assert not manager.has_index(MetricType.L2)
        assert not manager.has_index(MetricType.COSINE)

    def test_initialise_indexes_creates_sql_index(self, session, hnsw_pgvector_backend):
        _register_and_upsert(hnsw_pgvector_backend, session)
        hnsw_pgvector_backend.initialise_indexes(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            metric_types=[MetricType.L2],
        )
        manager = hnsw_pgvector_backend.get_index_manager(MODEL_NAME)
        assert manager.has_index(MetricType.L2)
        assert not manager.has_index(MetricType.COSINE)

    def test_initialise_indexes_is_idempotent(self, session, hnsw_pgvector_backend):
        _register_and_upsert(hnsw_pgvector_backend, session)
        hnsw_pgvector_backend.initialise_indexes(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            metric_types=[MetricType.L2],
        )
        hnsw_pgvector_backend.initialise_indexes(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            metric_types=[MetricType.L2],
        )

    def test_search_works_without_hnsw_index_flat_fallback(self, session, hnsw_pgvector_backend):
        _register_and_upsert(hnsw_pgvector_backend, session)
        # HNSW SQL index not created — pgvector falls back to sequential scan
        results = hnsw_pgvector_backend.get_nearest_concepts(
            session=session,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            query_embeddings=CONCEPTS["Hypertension"].embeddings,
            metric_type=MetricType.L2,
            concept_filter=EmbeddingConceptFilter(limit=1),
        )
        assert len(results) == 1
        assert results[0][0].concept_id == CONCEPTS["Hypertension"].concept_id

    def test_search_l2_with_hnsw_index(self, session, hnsw_pgvector_backend):
        _register_and_upsert(hnsw_pgvector_backend, session)
        hnsw_pgvector_backend.initialise_indexes(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            metric_types=[MetricType.L2],
        )
        results = hnsw_pgvector_backend.get_nearest_concepts(
            session=session,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            query_embeddings=CONCEPTS["Hypertension"].embeddings,
            metric_type=MetricType.L2,
            concept_filter=EmbeddingConceptFilter(limit=1),
        )
        assert len(results) == 1
        assert results[0][0].concept_id == CONCEPTS["Hypertension"].concept_id

    def test_search_cosine_with_hnsw_index(self, session, hnsw_pgvector_backend):
        _register_and_upsert(hnsw_pgvector_backend, session)
        hnsw_pgvector_backend.initialise_indexes(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            metric_types=[MetricType.COSINE],
        )
        results = hnsw_pgvector_backend.get_nearest_concepts(
            session=session,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            query_embeddings=CONCEPTS["Hypertension"].embeddings,
            metric_type=MetricType.COSINE,
            concept_filter=EmbeddingConceptFilter(limit=1),
        )
        assert len(results) == 1
        assert results[0][0].concept_id == CONCEPTS["Hypertension"].concept_id

    def test_rebuild_indexes_drops_and_recreates(self, session, hnsw_pgvector_backend):
        _register_and_upsert(hnsw_pgvector_backend, session)
        hnsw_pgvector_backend.initialise_indexes(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            metric_types=[MetricType.L2],
        )
        manager = hnsw_pgvector_backend.get_index_manager(MODEL_NAME)
        assert manager.has_index(MetricType.L2)

        hnsw_pgvector_backend.rebuild_model_indexes(
            MODEL_NAME,
            PROVIDER_TYPE,
            IndexType.HNSW,
            engine=session.bind,
            metric_types=[MetricType.L2],
        )
        assert manager.has_index(MetricType.L2)

        # Data must still be searchable
        results = hnsw_pgvector_backend.get_nearest_concepts(
            session=session,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            query_embeddings=CONCEPTS["Hypertension"].embeddings,
            metric_type=MetricType.L2,
            concept_filter=EmbeddingConceptFilter(limit=1),
        )
        assert results[0][0].concept_id == CONCEPTS["Hypertension"].concept_id

    def test_update_config_and_rebuild_uses_new_params(self, session, hnsw_pgvector_backend):
        _register_and_upsert(hnsw_pgvector_backend, session)
        hnsw_pgvector_backend.initialise_indexes(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            metric_types=[MetricType.L2],
        )

        new_config = HNSWIndexConfig(num_neighbors=8, ef_search=32, ef_construction=64)
        updated_record = hnsw_pgvector_backend.update_model_index_configuration(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            index_config=new_config,
        )
        # Manager was evicted — rebuild must re-register with new config
        assert MODEL_NAME not in hnsw_pgvector_backend._pgvector_index_managers

        hnsw_pgvector_backend.rebuild_model_indexes(
            updated_record.model_name,
            updated_record.provider_type,
            updated_record.index_type,
            engine=session.bind,
            metric_types=[MetricType.L2],
        )

        new_manager = hnsw_pgvector_backend.get_index_manager(MODEL_NAME)
        assert isinstance(new_manager, PGVectorHNSWIndexManager)
        assert new_manager.index_config.num_neighbors == 8
        assert new_manager.index_config.ef_construction == 64
        assert new_manager.has_index(MetricType.L2)

        # DDL should contain new params
        ddl = new_manager._create_index_ddl(MetricType.L2)
        assert "m = 8" in ddl
        assert "ef_construction = 64" in ddl

    def test_ef_search_set_per_session(self, session, hnsw_pgvector_backend):
        _register_and_upsert(hnsw_pgvector_backend, session)
        hnsw_pgvector_backend.initialise_indexes(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            metric_types=[MetricType.L2],
        )
        # After get_nearest_concepts the session should have ef_search set
        hnsw_pgvector_backend.get_nearest_concepts(
            session=session,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            query_embeddings=CONCEPTS["Hypertension"].embeddings,
            metric_type=MetricType.L2,
            concept_filter=EmbeddingConceptFilter(limit=1),
        )
        row = session.execute(
            __import__("sqlalchemy").text("SHOW hnsw.ef_search")
        ).scalar()
        assert str(HNSW_CONFIG.ef_search) in str(row)

    def test_flat_backend_has_no_sql_index(self, session, hnsw_pgvector_backend):
        hnsw_pgvector_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(),
            dimensions=EMBEDDING_DIM,
        )
        manager = hnsw_pgvector_backend.get_index_manager(MODEL_NAME)
        # Flat manager always returns True (sequential scan is implicit)
        assert manager.has_index(MetricType.L2) is True
        assert manager.has_index(MetricType.COSINE) is True

    def test_delete_model_evicts_index_manager(self, session, hnsw_pgvector_backend):
        _register_and_upsert(hnsw_pgvector_backend, session)
        assert MODEL_NAME in hnsw_pgvector_backend._pgvector_index_managers

        hnsw_pgvector_backend.delete_model(
            session.bind,
            MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
        )
        assert MODEL_NAME not in hnsw_pgvector_backend._pgvector_index_managers
