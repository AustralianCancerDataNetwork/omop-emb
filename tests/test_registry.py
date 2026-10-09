"""Tests for RegistryManager and registry schema upgrades."""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from oa_configurator import ensure_schema

from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.backends.sqlitevec.sqlitevec_backend import _load_sqlite_vec
from omop_emb.config import MODEL_REGISTRY_SCHEMA, IndexType, MetricType
from omop_emb.model_registry import (
    REGISTRY_SCHEMA_KEY,
    RegistryManager,
    ensure_registry_table,
    registry_reader_engine,
)
from omop_emb.utils.errors import LegacyRegistryError, ModelRegistrationConflictError

from .conftest import EMBEDDING_DIM, MODEL_NAME, PROVIDER_TYPE, sqlite_resolved_database


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


def _create_pre_scoping_registry(connection: sa.Connection, *, schema: str | None = None) -> None:
    """Create a model_registry table in the layout every pre-2.2 release
    shipped: the three signature columns and no database_config_name.

    The provider_type width reproduces v1.x, where the column was declared
    Enum(ProviderType, native_enum=False) and so rendered VARCHAR(6) holding
    uppercase member names. v2.0/v2.1 widened it to an unbounded string, which
    changes nothing about the detection this exercises.
    """
    qualified = "model_registry" if schema is None else f"{schema}.model_registry"
    connection.execute(sa.text(f"""
        CREATE TABLE {qualified} (
            model_name VARCHAR PRIMARY KEY,
            provider_type VARCHAR(6),
            storage_identifier VARCHAR NOT NULL UNIQUE,
            dimensions INTEGER NOT NULL,
            index_type VARCHAR,
            metric_type VARCHAR,
            index_config JSON,
            details JSON
        )
    """))
    connection.execute(sa.text(
        f"INSERT INTO {qualified} (model_name, provider_type, storage_identifier, dimensions) "
        "VALUES ('nomic-embed-text', 'OLLAMA', 'emb_nomic_embed_text', 768)"
    ))


@pytest.mark.unit
def test_pre_scoping_registry_is_rejected_rather_than_migrated():
    """A registry predating per-store row scoping is refused outright. No
    reader path upgrades it, so nothing runs DDL against a table a concurrent
    writer may be using; migrations/to_v2.py does that offline instead.
    """
    engine = registry_reader_engine(sqlite_resolved_database(), extensions=[_load_sqlite_vec])
    with engine.begin() as connection:
        _create_pre_scoping_registry(connection)

    with pytest.raises(LegacyRegistryError, match="database_config_name"):
        ensure_registry_table(engine)


@pytest.mark.unit
def test_current_registry_layout_is_not_mistaken_for_a_legacy_one():
    """The rejection keys on the absence of database_config_name, so a
    registry this version created must survive repeated opens."""
    engine = registry_reader_engine(sqlite_resolved_database(), extensions=[_load_sqlite_vec])
    ensure_registry_table(engine)
    ensure_registry_table(engine)

    with engine.connect() as connection:
        columns = {c["name"] for c in sa.inspect(connection).get_columns("model_registry")}
    assert "database_config_name" in columns


@pytest.mark.pgvector
@pytest.mark.integration
def test_pre_scoping_registry_in_another_schema_is_rejected(pg_engine):
    """The staggered-upgrade case: the current registry already exists here,
    created by another store, while this store's rows are still in their old
    schema, where they would read back as an empty catalogue rather than an
    error. Detection must therefore run even when the current registry is
    present, not only when one is missing.
    """
    stale_schema = "legacy_emb_registry"
    ensure_registry_table(pg_engine)
    with pg_engine.begin() as connection:
        ensure_schema(connection, stale_schema)
        _create_pre_scoping_registry(connection, schema=stale_schema)
    try:
        with pytest.raises(LegacyRegistryError, match=stale_schema):
            ensure_registry_table(pg_engine)
    finally:
        with pg_engine.begin() as connection:
            connection.execute(sa.text(f"DROP SCHEMA IF EXISTS {stale_schema} CASCADE"))


@pytest.mark.pgvector
@pytest.mark.integration
def test_registry_reader_engine_maps_the_registry_schema(pg_db):
    engine = registry_reader_engine(pg_db.resolved)
    try:
        translate_map = engine.get_execution_options()["schema_translate_map"]
        assert translate_map[REGISTRY_SCHEMA_KEY] == MODEL_REGISTRY_SCHEMA
    finally:
        engine.dispose()
