"""Integration tests for pgvector HNSW backend."""

from __future__ import annotations

import numpy as np
import pytest
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from omop_emb.config import IndexType, MetricType, ProviderType
from omop_emb.storage import PGVectorEmbeddingBackend
from omop_emb.storage.index_config import HNSWIndexConfig, FlatIndexConfig
from omop_emb.storage.postgres.pg_index_manager import PGVectorHNSWIndexManager
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter
from .conftest import CONCEPTS, MODEL_NAME, EMBEDDING_DIM, PROVIDER_TYPE


HNSW_CONFIG = HNSWIndexConfig(num_neighbors=4, ef_search=8, ef_construction=16)


@pytest.fixture
def hnsw_pgvector_backend(emb_engine: Engine, cdm_engine: Engine) -> PGVectorEmbeddingBackend:
    return PGVectorEmbeddingBackend(
        emb_engine=emb_engine,
        omop_cdm_engine=cdm_engine,
    )


def _register_and_upsert(backend, session, *, index_config=HNSW_CONFIG):
    backend.register_model(
        model_name=MODEL_NAME,
        provider_type=PROVIDER_TYPE,
        index_config=index_config,
        dimensions=EMBEDDING_DIM,
    )
    ids = [c.concept_id for c in CONCEPTS.values()]
    vecs = np.vstack([c.embeddings for c in CONCEPTS.values()]).astype(np.float32)
    backend.upsert_embeddings(
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
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=HNSW_CONFIG,
            dimensions=EMBEDDING_DIM,
        )
        manager = hnsw_pgvector_backend.get_index_manager(MODEL_NAME)
        assert isinstance(manager, PGVectorHNSWIndexManager)
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
        results = hnsw_pgvector_backend.get_nearest_concepts(
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

        record = hnsw_pgvector_backend.get_registered_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
        )
        hnsw_pgvector_backend.rebuild_model_indexes(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            metric_types=[MetricType.L2],
            _model_record=record,
        )
        assert manager.has_index(MetricType.L2)

        results = hnsw_pgvector_backend.get_nearest_concepts(
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
        assert MODEL_NAME not in hnsw_pgvector_backend._pgvector_index_managers

        hnsw_pgvector_backend.rebuild_model_indexes(
            model_name=updated_record.model_name,
            provider_type=ProviderType(updated_record.provider_type),
            index_type=IndexType(updated_record.index_type),
            metric_types=[MetricType.L2],
            _model_record=updated_record,
        )

        new_manager = hnsw_pgvector_backend.get_index_manager(MODEL_NAME)
        assert isinstance(new_manager, PGVectorHNSWIndexManager)
        assert new_manager.index_config.num_neighbors == 8
        assert new_manager.index_config.ef_construction == 64
        assert new_manager.has_index(MetricType.L2)

        ddl = new_manager._create_index_ddl(MetricType.L2)
        assert "m = 8" in ddl
        assert "ef_construction = 64" in ddl

    def test_flat_backend_has_no_sql_index(self, session, hnsw_pgvector_backend):
        hnsw_pgvector_backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(),
            dimensions=EMBEDDING_DIM,
        )
        manager = hnsw_pgvector_backend.get_index_manager(MODEL_NAME)
        assert manager.has_index(MetricType.L2) is True
        assert manager.has_index(MetricType.COSINE) is True

    def test_delete_model_evicts_index_manager(self, session, hnsw_pgvector_backend):
        _register_and_upsert(hnsw_pgvector_backend, session)
        assert MODEL_NAME in hnsw_pgvector_backend._pgvector_index_managers

        hnsw_pgvector_backend.delete_model(
            MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
        )
        assert MODEL_NAME not in hnsw_pgvector_backend._pgvector_index_managers
