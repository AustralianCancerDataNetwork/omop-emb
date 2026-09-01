"""Tests for RegistryManager and registry schema upgrades."""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.config import IndexType, MetricType
from omop_emb.model_registry import RegistryManager, ensure_registry_schema
from omop_emb.utils.errors import ModelRegistrationConflictError

from .conftest import EMBEDDING_DIM, MODEL_NAME, PROVIDER_TYPE


@pytest.fixture
def registry(svec_engine) -> RegistryManager:
    """RegistryManager backed by an in-memory SQLite engine."""
    return RegistryManager(svec_engine)


BACKEND_PREFIX = "sqlitevec"
METRIC = MetricType.L2
FLAT = FlatIndexConfig()
HNSW = HNSWIndexConfig(metric_type=MetricType.COSINE)

_SAFE = RegistryManager.safe_model_name(MODEL_NAME)
_PG_STORAGE_ID = RegistryManager.storage_name(_SAFE, "pgvector")


@pytest.mark.unit
class TestRegistryManager:
    def test_register_and_retrieve(self, registry: RegistryManager):
        record = registry.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FLAT,
            dimensions=EMBEDDING_DIM,
        )
        assert record.model_name == MODEL_NAME
        assert record.dimensions == EMBEDDING_DIM
        assert record.index_type == IndexType.FLAT

    def test_register_idempotent(self, registry: RegistryManager):
        r1 = registry.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FLAT,
            dimensions=EMBEDDING_DIM,
        )
        r2 = registry.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FLAT,
            dimensions=EMBEDDING_DIM,
        )
        assert r1.storage_identifier == r2.storage_identifier

    def test_dimension_conflict_raises(self, registry: RegistryManager):
        registry.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FLAT,
            dimensions=EMBEDDING_DIM,
        )
        with pytest.raises(ModelRegistrationConflictError, match="dimensions"):
            registry.register_model(
                model_name=MODEL_NAME,
                provider_type=PROVIDER_TYPE,
                index_config=FLAT,
                dimensions=EMBEDDING_DIM + 1,
            )

    def test_update_index_config_keeps_storage_identifier(
        self, registry: RegistryManager
    ):
        """Rebuilding from FLAT to HNSW keeps the same physical table name."""
        r_flat = registry.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FLAT,
            dimensions=EMBEDDING_DIM,
        )
        r_hnsw = registry.update_index_config(
            model_name=MODEL_NAME,
            index_config=HNSW,
        )
        assert r_flat.storage_identifier == r_hnsw.storage_identifier
        assert r_hnsw.index_type == IndexType.HNSW
        assert r_hnsw.metric_type == MetricType.COSINE

    def test_storage_name_excludes_index_type(self, registry: RegistryManager):
        record = registry.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FLAT,
            dimensions=EMBEDDING_DIM,
        )
        assert "flat" not in record.storage_identifier
        assert "hnsw" not in record.storage_identifier

    def test_get_model_exact_match(self, registry: RegistryManager):
        registry.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FLAT,
            dimensions=EMBEDDING_DIM,
        )
        records = registry.get_registered_models(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
        )
        assert len(records) == 1
        assert records[0].index_type == IndexType.FLAT

    def test_get_model_returns_none_for_missing(self, registry: RegistryManager):
        records = registry.get_registered_models(
            model_name="nonexistent",
            provider_type=PROVIDER_TYPE,
        )
        assert len(records) == 0

    def test_get_registered_models_filters_by_model(self, registry: RegistryManager):
        registry.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FLAT,
            dimensions=EMBEDDING_DIM,
        )
        registry.register_model(
            model_name="other-model",
            provider_type=PROVIDER_TYPE,
            index_config=FLAT,
            dimensions=EMBEDDING_DIM,
        )
        records = registry.get_registered_models(model_name=MODEL_NAME)
        assert all(r.model_name == MODEL_NAME for r in records)

    def test_delete_model(self, registry: RegistryManager):
        registry.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FLAT,
            dimensions=EMBEDDING_DIM,
        )
        registry.delete_model(model_name=MODEL_NAME)
        records = registry.get_registered_models(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
        )
        assert len(records) == 0

    def test_update_metadata(self, registry: RegistryManager):
        registry.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FLAT,
            dimensions=EMBEDDING_DIM,
        )
        updated = registry.update_metadata(
            model_name=MODEL_NAME,
            metadata={"custom": "value"},
        )
        assert updated.metadata.get("custom") == "value"

    def test_index_config_round_trips_in_metadata(self, registry: RegistryManager):
        """index_config serialised to details and deserialised back on retrieval."""
        hnsw = HNSWIndexConfig(
            metric_type=MetricType.COSINE,
            num_neighbors=32,
            ef_search=64,
            ef_construction=128,
        )
        registry.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=hnsw,
            dimensions=EMBEDDING_DIM,
        )
        records = registry.get_registered_models(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
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
        )
        assert name == "emb_mymodel_v1"


@pytest.mark.unit
class TestProviderTypeValidation:
    """provider_type is checked live against omop_llm.supported_providers()."""

    def test_unsupported_provider_type_raises(self, registry: RegistryManager):
        with pytest.raises(ValueError, match="Unsupported provider type"):
            registry.register_model(
                model_name=MODEL_NAME,
                provider_type="not-a-real-provider",
                index_config=FLAT,
                dimensions=EMBEDDING_DIM,
            )

    def test_every_supported_provider_is_accepted(self, registry: RegistryManager):
        from omop_llm import supported_providers

        for i, provider in enumerate(supported_providers()):
            record = registry.register_model(
                model_name=f"{MODEL_NAME}-{i}",
                provider_type=provider,
                index_config=FLAT,
                dimensions=EMBEDDING_DIM,
            )
            assert record.provider_type == provider


@pytest.mark.unit
def test_legacy_provider_name_is_normalized_in_sqlite(svec_engine):
    with svec_engine.begin() as connection:
        connection.execute(
            sa.text("CREATE TABLE model_registry (provider_type VARCHAR(6))")
        )
        connection.execute(
            sa.text("INSERT INTO model_registry (provider_type) VALUES ('OLLAMA')")
        )

    RegistryManager(svec_engine)

    with svec_engine.connect() as connection:
        assert connection.scalar(
            sa.text("SELECT provider_type FROM model_registry")
        ) == "ollama"


@pytest.mark.pgvector
@pytest.mark.integration
def test_legacy_provider_column_is_widened_in_postgres(pg_engine):
    try:
        with pg_engine.begin() as connection:
            connection.execute(sa.text("DROP TABLE IF EXISTS model_registry CASCADE"))
            connection.execute(
                sa.text("CREATE TABLE model_registry (provider_type VARCHAR(6))")
            )
            connection.execute(
                sa.text("INSERT INTO model_registry (provider_type) VALUES ('OLLAMA')")
            )

        RegistryManager(pg_engine)

        provider_column = next(
            column
            for column in sa.inspect(pg_engine).get_columns("model_registry")
            if column["name"] == "provider_type"
        )
        assert getattr(provider_column["type"], "length", None) is None

        with pg_engine.begin() as connection:
            assert connection.scalar(
                sa.text("SELECT provider_type FROM model_registry")
            ) == "ollama"
            connection.execute(
                sa.text("INSERT INTO model_registry (provider_type) VALUES ('anthropic')")
            )
    finally:
        with pg_engine.begin() as connection:
            connection.execute(sa.text("DROP TABLE IF EXISTS model_registry CASCADE"))
        ensure_registry_schema(pg_engine)
