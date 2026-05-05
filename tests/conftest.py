"""Test configuration and shared fixtures."""

from __future__ import annotations

import os
import time
from typing import Generator

import numpy as np
import pytest
import sqlalchemy as sa

from omop_emb.backends.base_backend import ConceptEmbeddingRecord
from omop_emb.backends.sqlitevec import SQLiteVecBackend, create_sqlitevec_engine
from omop_emb.config import MetricType, ProviderType


# ---------------------------------------------------------------------------
# Test data constants
# ---------------------------------------------------------------------------

MODEL_NAME = "test-model:v1"
PROVIDER_TYPE = ProviderType.OLLAMA
EMBEDDING_DIM = 1

# Fixed 1-D embeddings: Hypertension=-10, Diabetes=0, Aspirin=+10
# This makes L2 and cosine tests fully deterministic.
CONCEPT_RECORDS: tuple[ConceptEmbeddingRecord, ...] = (
    ConceptEmbeddingRecord(concept_id=1, domain_id="Condition", vocabulary_id="SNOMED", is_standard=True),
    ConceptEmbeddingRecord(concept_id=2, domain_id="Condition", vocabulary_id="SNOMED", is_standard=True),
    ConceptEmbeddingRecord(concept_id=3, domain_id="Drug", vocabulary_id="RxNorm", is_standard=True),
    ConceptEmbeddingRecord(concept_id=4, domain_id="Drug", vocabulary_id="RxNorm", is_standard=False),
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
# ---------------------------------------------------------------------------

_DB_HOST = os.getenv("TEST_DB_HOST")
_DB_PORT = os.getenv("TEST_DB_PORT")
_DB_NAME = os.getenv("TEST_DATABASE_NAME", "test_omop_emb")
_DB_USER = os.getenv("TEST_DB_USERNAME", "test")
_DB_PASS = os.getenv("TEST_DB_PASSWORD", "test")
_DB_ADMIN_USER = os.getenv("POSTGRES_USER", "postgres")
_DB_ADMIN_PASS = os.getenv("POSTGRES_PASSWORD", "postgres")
_DB_DRIVER = os.getenv("TEST_DB_DRIVER", "postgresql+psycopg2")

_pg_available = _DB_HOST is not None and (_DB_PORT is not None and _DB_PORT.isdigit())


def _test_db_url() -> sa.URL:
    return sa.URL.create(
        drivername=_DB_DRIVER,
        username=_DB_USER,
        password=_DB_PASS,
        host=_DB_HOST,
        port=int(_DB_PORT),
        database=_DB_NAME,
    )


def _admin_db_url() -> sa.URL:
    return sa.URL.create(
        drivername=_DB_DRIVER,
        username=_DB_ADMIN_USER,
        password=_DB_ADMIN_PASS,
        host=_DB_HOST,
        port=int(_DB_PORT),
        database="postgres",
    )


def _create_test_db() -> sa.URL:
    admin_engine = sa.create_engine(_admin_db_url(), future=True)
    try:
        with admin_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(sa.text(f'DROP DATABASE IF EXISTS "{_DB_NAME}"'))
            conn.execute(sa.text(f"DROP ROLE IF EXISTS {_DB_USER}"))
            conn.execute(sa.text(f"CREATE USER {_DB_USER} WITH PASSWORD '{_DB_PASS}' SUPERUSER"))
            conn.execute(sa.text(f'CREATE DATABASE "{_DB_NAME}" OWNER {_DB_USER}'))
    finally:
        admin_engine.dispose()
    return _test_db_url()


def _drop_test_db() -> None:
    admin_engine = sa.create_engine(_admin_db_url(), future=True)
    try:
        with admin_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(
                sa.text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :db AND pid <> pg_backend_pid()"
                ),
                {"db": _DB_NAME},
            )
            conn.execute(sa.text(f'DROP DATABASE IF EXISTS "{_DB_NAME}"'))
            conn.execute(sa.text(f"DROP ROLE IF EXISTS {_DB_USER}"))
    finally:
        admin_engine.dispose()


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
def svec_backend(svec_engine) -> SQLiteVecBackend:
    """In-memory SQLiteVecBackend, fresh per test."""
    return SQLiteVecBackend(emb_engine=svec_engine)


# ---------------------------------------------------------------------------
# Fixtures — pgvector (session-scoped engine, function-scoped backend)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pg_engine() -> Generator[sa.Engine, None, None]:
    """Session-scoped PostgreSQL engine.  Skipped when TEST_DB_HOST is unset."""
    if not _pg_available:
        pytest.skip("PostgreSQL not configured (set TEST_DB_HOST and TEST_DB_PORT)")

    url = _create_test_db()
    max_retries = 20
    for attempt in range(max_retries):
        try:
            engine = sa.create_engine(url, echo=False, future=True)
            with engine.connect() as conn:
                conn.execute(sa.text("SELECT 1"))
            yield engine
            engine.dispose()
            _drop_test_db()
            return
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                _drop_test_db()
                raise RuntimeError(
                    f"PostgreSQL never became available after {max_retries} attempts: {exc}"
                )


@pytest.fixture
def pg_backend(pg_engine: sa.Engine):
    """Function-scoped PGVectorEmbeddingBackend with a clean registry per test."""
    from omop_emb.backends.pgvector import PGVectorEmbeddingBackend

    backend = PGVectorEmbeddingBackend(emb_engine=pg_engine)

    yield backend

    # Tear down: remove all models registered during the test
    for record in backend.get_registered_models():
        try:
            backend.delete_model(
                model_name=record.model_name,
                provider_type=record.provider_type,
            )
        except Exception:
            pass
