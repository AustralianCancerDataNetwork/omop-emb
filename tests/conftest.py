"""Test configuration and shared fixtures."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest
import sqlalchemy as sa

from omop_emb.backends.base_backend import ConceptEmbeddingRecord
from omop_emb.backends.sqlitevec import (
    SQLiteVecEmbeddingBackend,
    create_sqlitevec_engine,
)
from omop_emb.config import OmopEmbConfig, ProviderType


@pytest.fixture(autouse=True, scope="session")
def _test_stack_config():
    """When no config.toml exists (CI), write a minimal one to the default config path.

    Uses StackConfig.for_session() + save_stack_config() so the file is written
    exactly as the CLI would produce it and load_stack_config() reads it normally.
    On a local machine where config.toml already exists, this is a no-op.
    test_emb_db is intentionally absent so pgvector tests skip via resolve_test_resource.
    """
    from oa_configurator.io import save_stack_config
    from oa_configurator.loader import CONFIG_PATH
    from oa_configurator.models import DatabaseConfig, ResourceConfig, StackConfig, ToolConfig

    if CONFIG_PATH.exists():
        yield
        return

    stack = StackConfig.for_session(
        databases={"cdm": DatabaseConfig(dialect="sqlite", database_name=":memory:")},
        resources={"cdm_db": ResourceConfig(database="cdm", cdm_schema="main")},
        tools={"omop_emb": ToolConfig()},
    )
    save_stack_config(stack, CONFIG_PATH)
    CONFIG_PATH.chmod(0o600)
    yield
    CONFIG_PATH.unlink()

# ---------------------------------------------------------------------------
# Test data constants
# ---------------------------------------------------------------------------

MODEL_NAME = "test-model:v1"
PROVIDER_TYPE = ProviderType.OLLAMA
EMBEDDING_DIM = 1

# Fixed 1-D embeddings: Hypertension=-10, Diabetes=0, Aspirin=+10
# This makes L2 and cosine tests fully deterministic.
CONCEPT_RECORDS: tuple[ConceptEmbeddingRecord, ...] = (
    ConceptEmbeddingRecord(
        concept_id=1, domain_id="Condition", vocabulary_id="SNOMED", is_standard=True
    ),
    ConceptEmbeddingRecord(
        concept_id=2, domain_id="Condition", vocabulary_id="SNOMED", is_standard=True
    ),
    ConceptEmbeddingRecord(
        concept_id=3, domain_id="Drug", vocabulary_id="RxNorm", is_standard=True
    ),
    ConceptEmbeddingRecord(
        concept_id=4, domain_id="Drug", vocabulary_id="RxNorm", is_standard=False
    ),
)

CONCEPT_EMBEDDINGS = np.array([[-10.0], [0.0], [10.0], [20.0]], dtype=np.float32)

# Kept for backward compat with tests that reference by name
HYPERTENSION_ID = 1
DIABETES_ID = 2
ASPIRIN_ID = 3
NON_STANDARD_ID = 4

# Query vector used for similarity math tests: [-1.0]
# L2 distances from [-1]: Hypertension=9, Diabetes=1, Aspirin=11, NonStandard=21
# L2 similarities:        0.1,            0.5,        ~0.083,      ~0.045
QUERY_EMBEDDING = np.array([[-1.0]], dtype=np.float32)


# ---------------------------------------------------------------------------
# PostgreSQL config (integration tests only)
#
# Resolved via OA_Configurator resource 'test_emb_db' in ~/.config/omop/config.toml.
# Run: omop-config configure omop_emb (answer Y when asked to configure test database).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fixtures — SQLiteVec (in-memory, function-scoped)
# ---------------------------------------------------------------------------


@pytest.fixture
def svec_engine():
    """In-memory SQLiteVec engine, fresh per test."""
    engine = create_sqlitevec_engine(":memory:")
    yield engine
    engine.dispose()


@pytest.fixture
def svec_backend(svec_engine) -> SQLiteVecEmbeddingBackend:
    """In-memory SQLiteVecEmbeddingBackend, fresh per test."""
    return SQLiteVecEmbeddingBackend(emb_engine=svec_engine)


# ---------------------------------------------------------------------------
# Fixtures — pgvector (session-scoped engine, function-scoped backend)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pg_engine() -> Iterator[sa.Engine]:
    """Session-scoped PostgreSQL engine. Skipped when test_emb_db is not configured."""
    from oa_configurator.pytest_plugin import (
        create_fresh_test_db,
        drop_test_db,
        ensure_test_user_exists,
        require_pg_extension,
        resolve_test_resource,
    )

    raw_url = resolve_test_resource(OmopEmbConfig.TEST_DB)
    ensure_test_user_exists(raw_url)
    url = create_fresh_test_db(raw_url, extensions=["vector"])
    require_pg_extension(url, "vector")  # defensive: verify installation succeeded
    engine = sa.create_engine(url, echo=False, future=True)
    try:
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        yield engine
    finally:
        engine.dispose()
        drop_test_db(raw_url)


@pytest.fixture
def pg_backend(pg_engine: sa.Engine):
    """Function-scoped PGVectorEmbeddingBackend with a clean registry per test."""
    from omop_emb.backends.pgvector import PGVectorEmbeddingBackend
    from omop_emb.backends.embedding_table import EmbeddingTableBase

    backend = PGVectorEmbeddingBackend(emb_engine=pg_engine)

    yield backend

    # Tear down: remove all models registered during the test
    for record in backend.get_registered_models():
        try:
            backend.delete_model(model_name=record.model_name)
        except Exception:
            pass

    # Remove the tables from the ORM cache
    EmbeddingTableBase.metadata.clear()
    EmbeddingTableBase.registry._class_registry.clear()
