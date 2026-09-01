"""Integration tests for the pgvector embedding backend.

Requires a running PostgreSQL instance with the pgvector extension.
Set TEST_DB_HOST and TEST_DB_PORT to enable these tests.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip(
    "pgvector", reason="omop-emb[pgvector] not installed: skipping pgvector tests"
)

from oa_configurator import schema_inspect
from oa_configurator.testing import isolated_test_schema

from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.backends.pgvector import PGVectorEmbeddingBackend
from omop_emb.config import IndexType, MetricType
from omop_emb.model_registry import RegistryManager

from .conftest import (
    CONCEPT_EMBEDDINGS,
    CONCEPT_RECORDS,
    EMBEDDING_DIM,
    HYPERTENSION_ID,
    MODEL_NAME,
    PROVIDER_TYPE,
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

    def test_rebuild_index_keeps_storage_identifier(
        self, pg_backend: PGVectorEmbeddingBackend
    ):
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

    def test_hnsw_registration_creates_index_manager(
        self, pg_backend: PGVectorEmbeddingBackend
    ):
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
        assert record is not None, (
            "Model record should exist after registration and rebuild"
        )
        mgr = pg_backend.get_index_manager(record.storage_identifier)
        assert isinstance(mgr, PGVectorHNSWIndexManager), (
            f"Index manager should be PGVectorHNSWIndexManager after rebuilding to HNSW. Got: {type(mgr)}"
        )

    def test_hnsw_search_returns_correct_top1(
        self, pg_backend: PGVectorEmbeddingBackend
    ):
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


@pytest.mark.pgvector
@pytest.mark.integration
class TestPGVectorNonDefaultSchema:
    """Every method here defaulted to the public schema in existing coverage,
    so a bug that silently ignored schema_translate_map would still pass
    every other test in this file. This is what actually catches that."""

    HNSW_CONFIG = HNSWIndexConfig(
        metric_type=MetricType.L2, num_neighbors=4, ef_search=8, ef_construction=16
    )

    @pytest.fixture
    def scoped_backend(self, pg_engine):
        with isolated_test_schema(pg_engine, prefix="emb_schema") as schema:
            scoped_engine = pg_engine.execution_options(
                schema_translate_map={None: schema}
            )
            backend = PGVectorEmbeddingBackend(emb_engine=scoped_engine)
            yield backend, schema

    def test_table_and_index_lifecycle_stays_in_the_configured_schema(
        self, scoped_backend, pg_engine
    ):
        backend, schema = scoped_backend
        record = backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(),
            dimensions=EMBEDDING_DIM,
        )
        backend.upsert_embeddings(
            model_name=MODEL_NAME,
            metric_type=MetricType.L2,
            records=list(CONCEPT_RECORDS),
            embeddings=CONCEPT_EMBEDDINGS,
        )

        # table_exists() sees it in the configured schema...
        assert backend._storage_table_exists(record) is True
        # ...and a bare inspector scoped to "public" doesn't.
        assert schema_inspect(pg_engine, schema="public").has_table(
            record.storage_identifier
        ) is False

        # get_indexes()/drop_index(): rebuild to HNSW, confirm the index lands
        # in the configured schema, then drop it.
        backend.rebuild_index(model_name=MODEL_NAME, index_config=self.HNSW_CONFIG)
        manager = backend.get_index_manager(record.storage_identifier)
        assert manager.has_index(MetricType.L2) is True
        indexes_in_schema = schema_inspect(pg_engine, schema=schema).get_indexes(
            record.storage_identifier
        )
        assert any(
            idx["name"] == manager._index_name(MetricType.L2) for idx in indexes_in_schema
        )
        manager.drop_index(MetricType.L2)
        assert manager.has_index(MetricType.L2) is False

        # drop_pg_embedding_table(): drops from the configured schema, not public.
        backend.delete_model(model_name=MODEL_NAME)
        assert backend._storage_table_exists(record) is False

    def test_model_registry_lookup_stays_in_the_configured_schema(
        self, scoped_backend, pg_engine
    ):
        backend, schema = scoped_backend
        backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(),
            dimensions=EMBEDDING_DIM,
        )

        scoped_engine = pg_engine.execution_options(schema_translate_map={None: schema})
        registry = RegistryManager.read_only(scoped_engine)
        assert registry.registry_available is True
        assert len(registry.get_registered_models(model_name=MODEL_NAME)) == 1

        # A registry pointed at "public" must not see it: proves the lookup is
        # genuinely schema-scoped, not incidentally finding it via search_path.
        public_engine = pg_engine.execution_options(schema_translate_map={None: "public"})
        public_registry = RegistryManager.read_only(public_engine)
        found_in_public = (
            public_registry.get_registered_models(model_name=MODEL_NAME)
            if public_registry.registry_available
            else ()
        )
        assert found_in_public == ()
