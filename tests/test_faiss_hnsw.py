"""Integration tests for FAISS HNSW backend.

Requires a running PostgreSQL instance (for the FAISS SQL registry table).
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from sqlalchemy import Engine

from omop_emb.config import IndexType, MetricType, ProviderType
from omop_emb.backends.faiss import FaissEmbeddingBackend
from omop_emb.backends.index_config import HNSWIndexConfig, FlatIndexConfig
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter
from .conftest import CONCEPTS, MODEL_NAME, EMBEDDING_DIM, PROVIDER_TYPE


HNSW_CONFIG = HNSWIndexConfig(num_neighbors=4, ef_search=8, ef_construction=16)


@pytest.fixture
def hnsw_backend(engine: Engine, temp_storage_dir) -> FaissEmbeddingBackend:
    backend = FaissEmbeddingBackend(
        omop_cdm_engine=engine,
        storage_base_dir=temp_storage_dir
    )
    return backend


@pytest.mark.faiss
@pytest.mark.integration
class TestFaissHNSWBackend:

    def test_register_model_with_hnsw_config(self, session, hnsw_backend):
        record = hnsw_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=HNSW_CONFIG,
            dimensions=EMBEDDING_DIM,
        )
        assert record.index_type == IndexType.HNSW
        assert record.metadata["index_config"]["num_neighbors"] == HNSW_CONFIG.num_neighbors
        assert record.metadata["index_config"]["ef_construction"] == HNSW_CONFIG.ef_construction

    def test_hnsw_config_roundtrips_from_metadata(self, session, hnsw_backend):
        hnsw_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=HNSW_CONFIG,
            dimensions=EMBEDDING_DIM,
        )
        retrieved = hnsw_backend.get_registered_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
        )
        assert retrieved is not None
        cfg = HNSWIndexConfig.from_metadata(retrieved.metadata)
        assert cfg.num_neighbors == HNSW_CONFIG.num_neighbors
        assert cfg.ef_search == HNSW_CONFIG.ef_search
        assert cfg.ef_construction == HNSW_CONFIG.ef_construction

    def test_upsert_with_metric_updates_index(self, session, hnsw_backend):
        hnsw_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=HNSW_CONFIG,
            dimensions=EMBEDDING_DIM,
        )
        concept = CONCEPTS["Hypertension"]
        ids = (concept.concept_id,)
        vecs = concept.embeddings

        hnsw_backend.upsert_embeddings(
            session=session,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            concept_ids=ids,
            embeddings=vecs,
            metric_type=MetricType.L2,
        )
        assert hnsw_backend.has_any_embeddings(
            session=session, model_name=MODEL_NAME,
            index_type=IndexType.HNSW, provider_type=PROVIDER_TYPE,
        )

    def test_nearest_neighbor_search_l2(self, session, hnsw_backend):
        hnsw_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=HNSW_CONFIG,
            dimensions=EMBEDDING_DIM,
        )
        all_concepts = list(CONCEPTS.values())
        ids = [c.concept_id for c in all_concepts]
        vecs = np.vstack([c.embeddings for c in all_concepts]).astype(np.float32)

        hnsw_backend.upsert_embeddings(
            session=session,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            concept_ids=ids,
            embeddings=vecs,
            metric_type=MetricType.L2,
        )

        query = CONCEPTS["Hypertension"].embeddings
        results = hnsw_backend.get_nearest_concepts(
            session=session,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            query_embeddings=query,
            metric_type=MetricType.L2,
            concept_filter=EmbeddingConceptFilter(limit=1),
        )
        assert len(results) == 1
        assert results[0][0].concept_id == CONCEPTS["Hypertension"].concept_id

    def test_nearest_neighbor_search_cosine(self, session, hnsw_backend):
        hnsw_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=HNSW_CONFIG,
            dimensions=EMBEDDING_DIM,
        )
        all_concepts = list(CONCEPTS.values())
        ids = [c.concept_id for c in all_concepts]
        vecs = np.vstack([c.embeddings for c in all_concepts]).astype(np.float32)

        hnsw_backend.upsert_embeddings(
            session=session,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            concept_ids=ids,
            embeddings=vecs,
            metric_type=MetricType.COSINE,
        )

        query = CONCEPTS["Hypertension"].embeddings
        results = hnsw_backend.get_nearest_concepts(
            session=session,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            query_embeddings=query,
            metric_type=MetricType.COSINE,
            concept_filter=EmbeddingConceptFilter(limit=1),
        )
        assert len(results) == 1
        assert results[0][0].concept_id == CONCEPTS["Hypertension"].concept_id

    def test_empty_storage_returns_correct_shape(self, session, hnsw_backend):
        hnsw_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=HNSW_CONFIG,
            dimensions=EMBEDDING_DIM,
        )
        query = np.array([[-1.0]], dtype=np.float32)
        results = hnsw_backend.get_nearest_concepts(
            session=session,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            query_embeddings=query,
            metric_type=MetricType.L2,
            concept_filter=EmbeddingConceptFilter(limit=3),
        )
        # One empty tuple per query vector — not a 0-tuple
        assert len(results) == 1
        assert len(results[0]) == 0

    def test_initialise_indexes_cold_start(self, engine: Engine, temp_storage_dir):
        backend_a = FaissEmbeddingBackend(
            omop_cdm_engine=engine,
            storage_base_dir=temp_storage_dir
        )
        backend_a.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=HNSW_CONFIG,
            dimensions=EMBEDDING_DIM,
        )
        ids = [c.concept_id for c in CONCEPTS.values()]
        vecs = np.vstack([c.embeddings for c in CONCEPTS.values()]).astype(np.float32)
        backend_a.upsert_embeddings(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            concept_ids=ids,
            embeddings=vecs,
            metric_type=MetricType.L2,
        )

        # Simulate cold start with a fresh backend pointing at same storage
        backend_b = FaissEmbeddingBackend(
            omop_cdm_engine=engine,
            storage_base_dir=temp_storage_dir
        )
        backend_b.initialise_indexes(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            metric_types=[MetricType.L2],
        )

        results = backend_b.get_nearest_concepts(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            query_embeddings=CONCEPTS["Hypertension"].embeddings,
            metric_type=MetricType.L2,
            concept_filter=EmbeddingConceptFilter(limit=1),
        )
        assert results[0][0].concept_id == CONCEPTS["Hypertension"].concept_id

    def test_rebuild_after_config_change(self, session, hnsw_backend):
        hnsw_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=HNSW_CONFIG,
            dimensions=EMBEDDING_DIM,
        )
        ids = [c.concept_id for c in CONCEPTS.values()]
        vecs = np.vstack([c.embeddings for c in CONCEPTS.values()]).astype(np.float32)
        hnsw_backend.upsert_embeddings(
            session=session,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            concept_ids=ids,
            embeddings=vecs,
            metric_type=MetricType.L2,
        )

        new_config = HNSWIndexConfig(num_neighbors=8, ef_search=32, ef_construction=64)
        updated_record = hnsw_backend.update_model_index_configuration(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            index_config=new_config,
        )
        hnsw_backend.rebuild_model_indexes(
            updated_record.model_name,
            updated_record.provider_type,
            updated_record.index_type,
            engine=session.bind,
            metric_types=[MetricType.L2],
        )

        # Verify search still works after rebuild
        results = hnsw_backend.get_nearest_concepts(
            session=session,
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=IndexType.HNSW,
            query_embeddings=CONCEPTS["Hypertension"].embeddings,
            metric_type=MetricType.L2,
            concept_filter=EmbeddingConceptFilter(limit=1),
        )
        assert results[0][0].concept_id == CONCEPTS["Hypertension"].concept_id
