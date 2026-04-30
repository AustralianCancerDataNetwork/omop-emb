"""Tests for PGVectorBaseIndexManager and its subclasses.

DDL-generation tests are pure unit tests (no DB).
Lifecycle tests (create/drop/has/rebuild) are integration tests and require
a running PostgreSQL instance with the vector extension installed.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy import inspect

from omop_emb.config import IndexType, MetricType, ProviderType
from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.backends.pgvector.pgvector_index_manager import (
    PGVectorFlatIndexManager,
    PGVectorHNSWIndexManager,
)
from omop_emb.backends.pgvector import PGVectorEmbeddingBackend


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

TABLENAME = "test_emb_idx_mgr"
EMBEDDING_COL = "embedding"
DEFAULT_HNSW_CONFIG = HNSWIndexConfig(num_neighbors=16, ef_search=64, ef_construction=128)


@pytest.fixture(scope="module")
def hnsw_table(pg_engine):
    """Create a minimal table with a vector column for index manager tests."""
    with pg_engine.begin() as conn:
        conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector CASCADE"))
        conn.execute(sa.text(
            f"CREATE TABLE IF NOT EXISTS {TABLENAME} "
            f"(concept_id INT PRIMARY KEY, {EMBEDDING_COL} vector(4))"
        ))
    yield
    with pg_engine.begin() as conn:
        conn.execute(sa.text(f"DROP TABLE IF EXISTS {TABLENAME} CASCADE"))


@pytest.fixture
def hnsw_manager(pg_engine, hnsw_table) -> PGVectorHNSWIndexManager:
    mgr = PGVectorHNSWIndexManager(
        omop_cdm_engine=pg_engine,
        tablename=TABLENAME,
        embedding_column=EMBEDDING_COL,
        index_config=DEFAULT_HNSW_CONFIG,
    )
    # Always start with a clean slate
    for metric in MetricType:
        mgr.drop_index(metric)
    return mgr


# ---------------------------------------------------------------------------
# PGVectorFlatIndexManager — unit tests (no DB needed for these)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPGVectorFlatIndexManagerUnit:

    def test_supported_index_type(self):
        mgr = PGVectorFlatIndexManager.__new__(PGVectorFlatIndexManager)
        mgr._index_config = FlatIndexConfig()
        assert mgr.supported_index_type == IndexType.FLAT

    def test_has_index_always_true(self):
        mgr = PGVectorFlatIndexManager(omop_cdm_engine=None, tablename="t", embedding_column="e", index_config=FlatIndexConfig())  # type: ignore[arg-type]
        assert mgr.has_index(MetricType.L2) is True
        assert mgr.has_index(MetricType.COSINE) is True

    def test_create_index_noop(self):
        mgr = PGVectorFlatIndexManager(omop_cdm_engine=None, tablename="t", embedding_column="e", index_config=FlatIndexConfig())  # type: ignore[arg-type]
        mgr.create_index(MetricType.L2)  # must not raise or create anything

    def test_drop_index_noop(self):
        mgr = PGVectorFlatIndexManager(omop_cdm_engine=None, tablename="t", embedding_column="e", index_config=FlatIndexConfig())  # type: ignore[arg-type]
        mgr.drop_index(MetricType.L2)  # must not raise or drop anything

    def test_create_index_ddl_returns_none(self):
        mgr = PGVectorFlatIndexManager(omop_cdm_engine=None, tablename="t", embedding_column="e", index_config=FlatIndexConfig())  # type: ignore[arg-type]
        assert mgr._create_index_ddl(MetricType.L2) is None

    def test_wrong_index_config_raises(self):
        with pytest.raises(ValueError, match="index_type"):
            PGVectorFlatIndexManager(
                omop_cdm_engine=None,  # type: ignore
                tablename="t",
                embedding_column="e",
                index_config=HNSWIndexConfig() # type: ignore
            )


# ---------------------------------------------------------------------------
# PGVectorHNSWIndexManager — DDL generation (pure unit, no DB)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPGVectorHNSWIndexManagerDDL:

    @pytest.fixture
    def mgr(self) -> PGVectorHNSWIndexManager:
        # engine not needed for DDL-string tests
        m = PGVectorHNSWIndexManager.__new__(PGVectorHNSWIndexManager)
        m._omop_cdm_engine = None  # type: ignore
        m._tablename = "my_table"
        m._embedding_column = "embedding"
        m._index_config = HNSWIndexConfig(num_neighbors=32, ef_search=64, ef_construction=128)
        return m

    def test_index_name_format(self, mgr):
        assert mgr._index_name(MetricType.L2) == "idx_my_table_l2"
        assert mgr._index_name(MetricType.COSINE) == "idx_my_table_cosine"

    def test_ddl_l2_contains_correct_ops(self, mgr):
        ddl = mgr._create_index_ddl(MetricType.L2)
        assert "vector_l2_ops" in ddl
        assert "USING hnsw" in ddl
        assert "m = 32" in ddl
        assert "ef_construction = 128" in ddl

    def test_ddl_cosine_contains_correct_ops(self, mgr):
        ddl = mgr._create_index_ddl(MetricType.COSINE)
        assert "vector_cosine_ops" in ddl

    def test_ddl_l1_contains_correct_ops(self, mgr):
        ddl = mgr._create_index_ddl(MetricType.L1)
        assert "vector_l1_ops" in ddl

    def test_ddl_hamming_generates_bit_ops_ddl(self, mgr):
        # HAMMING/JACCARD only valid for bit vectors — our embedding column is vector.
        # _ops_for_metric still generates DDL with bit_hamming_ops; failure is at DB level.
        ddl = mgr._create_index_ddl(MetricType.HAMMING)
        assert ddl is not None
        assert "bit_hamming_ops" in ddl

    def test_supported_index_type(self, mgr):
        assert mgr.supported_index_type == IndexType.HNSW

    def test_wrong_config_type_raises(self):
        with pytest.raises(ValueError, match="index_type"):
            PGVectorHNSWIndexManager(
                omop_cdm_engine=None,  # type: ignore
                tablename="t",
                embedding_column="e",
                index_config=FlatIndexConfig() # type: ignore
            )


# ---------------------------------------------------------------------------
# PGVectorHNSWIndexManager — integration tests (need DB)
# ---------------------------------------------------------------------------

@pytest.mark.pgvector
@pytest.mark.integration
class TestPGVectorHNSWIndexManagerIntegration:

    def test_has_index_false_before_creation(self, hnsw_manager):
        assert hnsw_manager.has_index(MetricType.L2) is False

    def test_create_index_l2(self, hnsw_manager, pg_engine):
        hnsw_manager.create_index(MetricType.L2)
        assert hnsw_manager.has_index(MetricType.L2)

        with pg_engine.connect() as conn:
            names = {i["name"] for i in inspect(conn).get_indexes(TABLENAME)}
        assert hnsw_manager._index_name(MetricType.L2) in names

    def test_create_index_is_idempotent(self, hnsw_manager):
        hnsw_manager.create_index(MetricType.L2)
        hnsw_manager.create_index(MetricType.L2)  # must not raise

    def test_create_index_cosine_independently(self, hnsw_manager):
        hnsw_manager.create_index(MetricType.COSINE)
        assert hnsw_manager.has_index(MetricType.COSINE)
        assert not hnsw_manager.has_index(MetricType.L2)

    def test_drop_index_removes_from_db(self, hnsw_manager, pg_engine):
        hnsw_manager.create_index(MetricType.L2)
        assert hnsw_manager.has_index(MetricType.L2)

        hnsw_manager.drop_index(MetricType.L2)
        assert not hnsw_manager.has_index(MetricType.L2)

        with pg_engine.connect() as conn:
            names = {i["name"] for i in inspect(conn).get_indexes(TABLENAME)}
        assert hnsw_manager._index_name(MetricType.L2) not in names

    def test_drop_index_is_idempotent(self, hnsw_manager):
        hnsw_manager.drop_index(MetricType.L2)  # nothing there yet
        hnsw_manager.drop_index(MetricType.L2)  # still nothing — must not raise

    def test_rebuild_index_drops_and_recreates(self, hnsw_manager):
        hnsw_manager.create_index(MetricType.L2)
        assert hnsw_manager.has_index(MetricType.L2)

        hnsw_manager.rebuild_index(MetricType.L2)
        assert hnsw_manager.has_index(MetricType.L2)

    def test_load_or_create_creates_if_missing(self, hnsw_manager):
        assert not hnsw_manager.has_index(MetricType.L2)
        hnsw_manager.load_or_create(MetricType.L2)
        assert hnsw_manager.has_index(MetricType.L2)

    def test_load_or_create_is_idempotent_when_exists(self, hnsw_manager):
        hnsw_manager.create_index(MetricType.L2)
        hnsw_manager.load_or_create(MetricType.L2)  # already exists — no-op

    def test_new_config_produces_different_ddl(self, pg_engine, hnsw_table):
        mgr_a = PGVectorHNSWIndexManager(
            omop_cdm_engine=pg_engine, tablename=TABLENAME, embedding_column=EMBEDDING_COL,
            index_config=HNSWIndexConfig(num_neighbors=8, ef_search=16, ef_construction=32),
        )
        mgr_b = PGVectorHNSWIndexManager(
            omop_cdm_engine=pg_engine, tablename=TABLENAME, embedding_column=EMBEDDING_COL,
            index_config=HNSWIndexConfig(num_neighbors=64, ef_search=128, ef_construction=256),
        )
        ddl_a = mgr_a._create_index_ddl(MetricType.L2)
        ddl_b = mgr_b._create_index_ddl(MetricType.L2)
        assert "m = 8" in ddl_a
        assert "m = 64" in ddl_b
        assert ddl_a != ddl_b


# ---------------------------------------------------------------------------
# PGVectorEmbeddingBackend — dimension validation (pure unit, no DB)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPGVectorDimensionValidation:

    def test_register_model_rejects_oversized_dimensions(self, tmp_path):
        backend = PGVectorEmbeddingBackend(storage_base_dir=tmp_path)
        with pytest.raises(ValueError, match="2,000"):
            backend.register_model(
                engine=None,  # type: ignore[arg-type]
                model_name="big-model:v1",
                dimensions=2001,
                provider_type=ProviderType.OLLAMA,
                index_config=FlatIndexConfig(),
            )

    def test_register_model_error_mentions_alternatives(self, tmp_path):
        backend = PGVectorEmbeddingBackend(storage_base_dir=tmp_path)
        with pytest.raises(ValueError, match="halfvec"):
            backend.register_model(
                engine=None,  # type: ignore[arg-type]
                model_name="big-model:v1",
                dimensions=4096,
                provider_type=ProviderType.OLLAMA,
                index_config=FlatIndexConfig(),
            )
