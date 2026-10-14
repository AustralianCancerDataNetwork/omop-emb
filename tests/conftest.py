"""Test configuration and shared fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import sqlalchemy as sa
from oa_configurator import (
    CDMDatabaseConfig,
    ConnectionConfig,
    Dialect,
    GenericDatabaseConfig,
    ResolvedCDMDatabase,
    ResolvedDatabase,
    ResolvedVectorStore,
    Resolver,
    SchemaClaim,
    StackConfig,
)

from omop_emb.backends.base_backend import ConceptEmbeddingRecord
from omop_emb.backends.sqlitevec import SQLiteVecEmbeddingBackend
from omop_emb.backends.sqlitevec.sqlitevec_backend import _load_sqlite_vec
from omop_emb.config import MODEL_REGISTRY_SCHEMA, REGISTRY_SCHEMA_KEY, OmopEmbConfig
from omop_emb.model_registry import ensure_registry_table


def sqlite_resolved_database(database_name: str = ":memory:") -> ResolvedDatabase:
    """A plain oa-configurator ``ResolvedDatabase`` for a SQLite target, for
    test code that needs ``registry_writer_engine()``/``registry_reader_engine()``
    against a bare SQLite file or ``:memory:`` target with no config file
    involved."""
    cfg = StackConfig.for_session(
        connections={"db": ConnectionConfig(dialect=Dialect.SQLITE, database_name=database_name)},
        databases={"default": GenericDatabaseConfig(connection="db")},
    )
    return Resolver(cfg).resolve_database("default")


def sqlite_cdm_database(database_name: str = ":memory:") -> ResolvedCDMDatabase:
    """A SQLite CDM database entry, for test code reading concepts."""
    cfg = StackConfig.for_session(
        connections={"db": ConnectionConfig(dialect=Dialect.SQLITE, database_name=database_name)},
        databases={"cdm": CDMDatabaseConfig(connection="db")},
    )
    resolved = Resolver(cfg).resolve_database("cdm")
    assert isinstance(resolved, ResolvedCDMDatabase)
    return resolved


def sqlite_resolved_vector_store(database_name: str = ":memory:") -> ResolvedVectorStore:
    """A sqlite-vec ``ResolvedVectorStore`` on ``sqlite_resolved_database(database_name)``."""
    return ResolvedVectorStore(
        name="default",
        backend_type="sqlitevec",
        database=sqlite_resolved_database(database_name),
        faiss_cache_dir=None,
        configuration={},
    )


def duckdbvss_resolved_vector_store(database_name: str) -> ResolvedVectorStore:
    """A duckdbvss ``ResolvedVectorStore`` on a plain, non-``test_only`` file
    target, mirroring ``sqlite_resolved_vector_store``.

    Deliberately not ``test_only`` to skip the  schema-claim drift check that 
    would hide a real bug in the reader's own nonexistent-file handling.
    """
    cfg = StackConfig.for_session(
        connections={"db": ConnectionConfig(dialect=Dialect.DUCKDB, database_name=database_name)},
        databases={"default": GenericDatabaseConfig(connection="db")},
    )
    return ResolvedVectorStore(
        name="default",
        backend_type="duckdbvss",
        database=Resolver(cfg).resolve_database("default"),
        faiss_cache_dir=None,
        configuration={},
    )


def duckdb_test_resolver(database_name: str = ":memory:") -> Resolver:
    """A ``Resolver`` with a ``test_emb_db_duckdb`` entry, for
    ``isolated_test_database()``'s ``resolver=`` override.

    ``DuckDBTestStrategy.resolve_without_config()`` (the fallback
    ``isolated_test_database(..., dialect="duckdb")`` would otherwise use
    when nothing configures the field) fabricates a ``ResolvedCDMDatabase``,
    built for omop-alchemy's CDM fixtures. omop-emb's embedding store is a
    plain database, not a CDM one, so this builds a real, resolvable
    ``GenericDatabaseConfig`` entry instead of relying on that fallback.
    """
    cfg = StackConfig.for_session(
        connections={"db": ConnectionConfig(dialect=Dialect.DUCKDB, database_name=database_name, test_only=True)},
        databases={"test_emb_db_duckdb": GenericDatabaseConfig(connection="db")},
    )
    return Resolver(cfg)


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
    ``sa.create_engine()``. Claims the registry schema and ensures the
    registry table exists, matching what ``registry_writer_engine()``
    does for a real engine built straight off a ``ResolvedDatabase``."""
    from oa_configurator.testing import isolated_test_database

    with isolated_test_database(
        OmopEmbConfig, "test_emb_db_sqlite", dialect="sqlite", request=request,
        schema_claims=[SchemaClaim(
            schema_tag=REGISTRY_SCHEMA_KEY, physical_schema=MODEL_REGISTRY_SCHEMA,
            reserved=True, owner="omop_emb",
        )],
        extensions=[_load_sqlite_vec],
    ) as db:
        engine = db.connection.engine
        ensure_registry_table(engine)
        yield engine


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
    from oa_configurator.testing import (
        install_postgres_extension,
        isolated_test_database,
    )

    with isolated_test_database(
        OmopEmbConfig, "test_emb_db_pg",
        extensions=[install_postgres_extension("vector")],
        request=request,
    ) as db:
        yield db


@pytest.fixture
def pg_backend(pg_db):
    """Function-scoped PGVectorEmbeddingBackend with a clean registry per test,
    built through ``_open_writer()`` so its engine claims the registry
    schema and installs pgvector exactly as production does."""
    from omop_emb.backends.base_backend import _open_writer
    from omop_emb.backends.embedding_table import EmbeddingTableBase

    backend = _open_writer("pgvector", database=pg_db.resolved)

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
    backend.emb_engine.dispose()


@pytest.fixture
def pg_engine(pg_backend) -> sa.Engine:
    """``pg_backend``'s own committing engine: registry schema claimed and
    pgvector installed. Isolation comes from ``pg_backend``'s teardown."""
    return pg_backend.emb_engine


# ---------------------------------------------------------------------------
# Fixtures: DuckDB VSS (in-memory, function-scoped)
# ---------------------------------------------------------------------------


@pytest.fixture
def duckdbvss_db(request):
    """Canonical isolated DuckDB test database, VSS extension loaded."""
    from oa_configurator.testing import isolated_test_database

    from omop_emb.backends.duckdbvss.duckdbvss_backend import _install_vss

    with isolated_test_database(
        OmopEmbConfig, "test_emb_db_duckdb",
        resolver=duckdb_test_resolver(),
        extensions=[_install_vss],
        request=request,
    ) as db:
        yield db


@pytest.fixture
def duckdbvss_backend(duckdbvss_db):
    """Function-scoped DuckDBVSSEmbeddingBackend with a clean registry per
    test, built through ``_open_writer()`` so its engine claims the registry
    schema and loads the VSS extension exactly as production does."""
    from omop_emb.backends.base_backend import _open_writer
    from omop_emb.backends.embedding_table import EmbeddingTableBase

    backend = _open_writer("duckdbvss", database=duckdbvss_db.resolved)

    yield backend

    for record in backend.get_registered_models():
        try:
            backend.delete_model(model_name=record.model_name)
        except Exception:
            pass

    EmbeddingTableBase.metadata.clear()
    EmbeddingTableBase.registry._class_registry.clear()
    backend.emb_engine.dispose()


@pytest.fixture
def duckdbvss_engine(duckdbvss_backend) -> sa.Engine:
    """``duckdbvss_backend``'s own engine: registry schema claimed and VSS loaded."""
    return duckdbvss_backend.emb_engine
