"""Test configuration and shared fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import sqlalchemy as sa

from omop_emb.backends.base_backend import ConceptEmbeddingRecord
from omop_emb.backends.sqlitevec import (
    SQLiteVecEmbeddingBackend,
    create_sqlitevec_engine,
)
from omop_emb.config import OmopEmbConfig


# ---------------------------------------------------------------------------
# Test data constants
# ---------------------------------------------------------------------------

MODEL_NAME = "test-model:v1"
PROVIDER_TYPE = "ollama"
EMBEDDING_DIM = 1

# Fixed 1-D embeddings: Hypertension=-10, Diabetes=0, Aspirin=+10
# This makes L2 and cosine tests fully deterministic.
CONCEPT_RECORDS: tuple[ConceptEmbeddingRecord, ...] = (
    ConceptEmbeddingRecord(
        concept_id=1, domain_id="Condition", vocabulary_id="SNOMED", is_standard=True, is_valid=True
    ),
    ConceptEmbeddingRecord(
        concept_id=2, domain_id="Condition", vocabulary_id="SNOMED", is_standard=True, is_valid=True
    ),
    ConceptEmbeddingRecord(
        concept_id=3, domain_id="Drug", vocabulary_id="RxNorm", is_standard=True, is_valid=True
    ),
    ConceptEmbeddingRecord(
        concept_id=4, domain_id="Drug", vocabulary_id="RxNorm", is_standard=False, is_valid=True
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
# Resolved via OA_Configurator resource 'test_emb_db_pg' in ~/.config/omop/config.toml.
# Run: omop-config configure omop_emb (answer Y when asked to configure test database).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fixtures: SQLiteVec (in-memory, function-scoped)
# ---------------------------------------------------------------------------


@pytest.fixture
def svec_engine(request):
    """Fresh SQLiteVec engine per test, via oa-configurator's canonical
    dialect-agnostic test-database entrypoint rather than a hand-built
    ``sa.create_engine()``."""
    from oa_configurator.testing import isolated_test_database

    with isolated_test_database(
        OmopEmbConfig, "test_emb_db_sqlite", dialect="sqlite", request=request
    ) as db:
        yield create_sqlitevec_engine(db.connection.engine)


@pytest.fixture
def svec_backend(svec_engine) -> SQLiteVecEmbeddingBackend:
    """In-memory SQLiteVecEmbeddingBackend, fresh per test."""
    return SQLiteVecEmbeddingBackend(emb_engine=svec_engine)


# ---------------------------------------------------------------------------
# Fixtures: pgvector (session-scoped engine, function-scoped backend)
# ---------------------------------------------------------------------------


@pytest.fixture
def pg_db(request):
    """Canonical isolated PostgreSQL test database (Phase 0 of the
    schema_translate_map fix)."""
    from oa_configurator.testing import isolated_test_database

    with isolated_test_database(
        OmopEmbConfig, "test_emb_db_pg", extensions=["vector"], request=request
    ) as db:
        yield db


@pytest.fixture
def pg_engine(pg_db) -> sa.Engine:
    """Real, committing engine for ``PGVectorEmbeddingBackend`` (needs
    ``.begin()``/``.connect()`` semantics a bare ``Connection`` can't give).
    A thin shim over ``pg_db.connection.engine``; isolation comes from
    ``pg_backend``'s teardown (drops each model's table), not a rollback.
    """
    return pg_db.connection.engine


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
