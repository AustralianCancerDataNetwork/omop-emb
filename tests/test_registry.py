"""Unit tests for RegistryManager using in-memory SQLite."""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.config import BackendType, IndexType, MetricType, ProviderType
from omop_emb.model_registry import RegistryManager
from omop_emb.utils.errors import ModelRegistrationConflictError

from .conftest import EMBEDDING_DIM, MODEL_NAME, PROVIDER_TYPE


@pytest.fixture
def registry(svec_engine) -> RegistryManager:
    """RegistryManager backed by an in-memory SQLite engine."""
    return RegistryManager(svec_engine)


BACKEND = BackendType.SQLITEVEC
METRIC = MetricType.L2
FLAT = FlatIndexConfig()
HNSW = HNSWIndexConfig(metric_type=MetricType.COSINE)


@pytest.mark.unit
class TestRegistryManager:

    def test_register_and_retrieve(self, registry: RegistryManager):
        record = registry.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            backend_type=BACKEND,
            index_config=FLAT,
            dimensions=EMBEDDING_DIM,
        )
        assert record.model_name == MODEL_NAME
        assert record.dimensions == EMBEDDING_DIM
        assert record.index_type == IndexType.FLAT

    def test_register_idempotent(self, registry: RegistryManager):
        r1 = registry.register_model(
            model_name=MODEL_NAME, provider_type=PROVIDER_TYPE, backend_type=BACKEND,
            index_config=FLAT, dimensions=EMBEDDING_DIM,
        )
        r2 = registry.register_model(
            model_name=MODEL_NAME, provider_type=PROVIDER_TYPE, backend_type=BACKEND,
            index_config=FLAT, dimensions=EMBEDDING_DIM,
        )
        assert r1.storage_identifier == r2.storage_identifier

    def test_dimension_conflict_raises(self, registry: RegistryManager):
        registry.register_model(
            model_name=MODEL_NAME, provider_type=PROVIDER_TYPE, backend_type=BACKEND,
            index_config=FLAT, dimensions=EMBEDDING_DIM,
        )
        with pytest.raises(ModelRegistrationConflictError, match="dimensions"):
            registry.register_model(
                model_name=MODEL_NAME, provider_type=PROVIDER_TYPE, backend_type=BACKEND,
                index_config=FLAT, dimensions=EMBEDDING_DIM + 1,
            )

    def test_update_index_config_keeps_storage_identifier(self, registry: RegistryManager):
        """Rebuilding from FLAT to HNSW keeps the same physical table name."""
        r_flat = registry.register_model(
            model_name=MODEL_NAME, provider_type=PROVIDER_TYPE,
            backend_type=BackendType.PGVECTOR,
            index_config=FLAT, dimensions=EMBEDDING_DIM,
        )
        r_hnsw = registry.update_index_config(
            model_name=MODEL_NAME,
            backend_type=BackendType.PGVECTOR,
            index_config=HNSW,
        )
        assert r_flat.storage_identifier == r_hnsw.storage_identifier
        assert r_hnsw.index_type == IndexType.HNSW
        assert r_hnsw.metric_type == MetricType.COSINE

    def test_storage_name_excludes_index_type(self, registry: RegistryManager):
        record = registry.register_model(
            model_name=MODEL_NAME, 
            provider_type=PROVIDER_TYPE, 
            backend_type=BACKEND,
            index_config=FLAT, 
            dimensions=EMBEDDING_DIM,
        )
        assert "flat" not in record.storage_identifier
        assert "hnsw" not in record.storage_identifier

    def test_get_model_exact_match(self, registry: RegistryManager):
        registry.register_model(
            model_name=MODEL_NAME, 
            provider_type=PROVIDER_TYPE, 
            backend_type=BACKEND,
            index_config=FLAT, 
            dimensions=EMBEDDING_DIM,
        )
        records = registry.get_registered_models(
            model_name=MODEL_NAME, provider_type=PROVIDER_TYPE, backend_type=BACKEND,
        )
        assert len(records) == 1
        assert records[0].index_type == IndexType.FLAT

    def test_get_model_returns_none_for_missing(self, registry: RegistryManager):
        records = registry.get_registered_models(
            model_name="nonexistent", provider_type=PROVIDER_TYPE, backend_type=BACKEND,
        )
        assert len(records) == 0

    def test_get_registered_models_filters_by_model(self, registry: RegistryManager):
        registry.register_model(
            model_name=MODEL_NAME, 
            provider_type=PROVIDER_TYPE, 
            backend_type=BACKEND,
            index_config=FLAT, 
            dimensions=EMBEDDING_DIM,
        )
        registry.register_model(
            model_name="other-model", 
            provider_type=PROVIDER_TYPE, 
            backend_type=BACKEND,
            index_config=FLAT, 
            dimensions=EMBEDDING_DIM,
        )
        records = registry.get_registered_models(
            backend_type=BACKEND, model_name=MODEL_NAME
        )
        assert all(r.model_name == MODEL_NAME for r in records)

    def test_delete_model(self, registry: RegistryManager):
        registry.register_model(
            model_name=MODEL_NAME, 
            provider_type=PROVIDER_TYPE, 
            backend_type=BACKEND,
            index_config=FLAT, 
            dimensions=EMBEDDING_DIM,
        )
        registry.delete_model(
            model_name=MODEL_NAME, backend_type=BACKEND,
        )
        records = registry.get_registered_models(
            model_name=MODEL_NAME, provider_type=PROVIDER_TYPE, backend_type=BACKEND,
        )
        assert len(records) == 0

    def test_update_metadata(self, registry: RegistryManager):
        registry.register_model(
            model_name=MODEL_NAME, 
            provider_type=PROVIDER_TYPE, 
            backend_type=BACKEND,
            index_config=FLAT, 
            dimensions=EMBEDDING_DIM,
        )
        updated = registry.update_metadata(
            model_name=MODEL_NAME, backend_type=BACKEND,
            metadata={"custom": "value"},
        )
        assert updated.metadata.get("custom") == "value"

    def test_index_config_round_trips_in_metadata(self, registry: RegistryManager):
        """index_config serialised to details and deserialised back on retrieval."""
        hnsw = HNSWIndexConfig(metric_type=MetricType.COSINE, num_neighbors=32, ef_search=64, ef_construction=128)
        registry.register_model(
            model_name=MODEL_NAME, 
            provider_type=PROVIDER_TYPE,
            backend_type=BackendType.PGVECTOR,
            index_config=hnsw, 
            dimensions=EMBEDDING_DIM,
        )
        records = registry.get_registered_models(
            model_name=MODEL_NAME, provider_type=PROVIDER_TYPE,
            backend_type=BackendType.PGVECTOR,
        )
        from omop_emb.backends.index_config import HNSWIndexConfig as HNSWCfg
        assert isinstance(records[0].index_config, HNSWCfg)
        assert records[0].index_config.num_neighbors == 32

    def test_safe_model_name_normalisation(self):
        assert RegistryManager.safe_model_name("MyModel:v1") == "mymodel_v1"
        assert RegistryManager.safe_model_name("  spaces  ") == "spaces"
        assert RegistryManager.safe_model_name("a__b") == "a_b"

    def test_storage_name_format(self):
        name = RegistryManager.storage_name(
            safe_model_name="mymodel_v1",
            backend_type=BackendType.SQLITEVEC,
        )
        assert name == "sqlitevec_mymodel_v1"
