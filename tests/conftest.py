"""Test configuration and shared fixtures."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Self

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

from omop_emb.backends.base_backend import (
    ConceptEmbeddingRecord,
    EmbeddingBackend,
    EmbeddingStoreReader,
)
from omop_emb.backends.index_config import FlatIndexConfig
from omop_emb.backends.sqlitevec import SQLiteVecEmbeddingBackend
from omop_emb.backends.sqlitevec.sqlitevec_backend import _load_sqlite_vec
from omop_emb.config import (
    MODEL_REGISTRY_SCHEMA,
    REGISTRY_SCHEMA_KEY,
    BackendType,
    MetricType,
    OmopEmbConfig,
)
from omop_emb.model_registry import ensure_registry_table, registry_writer_engine
from omop_emb.model_registry.model_registry_types import EmbeddingModelRecord
from omop_emb.utils.embedding_utils import (
    EmbeddingArray,
    EmbeddingConceptFilter,
    NearestConceptMatch,
)


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


# ---------------------------------------------------------------------------
# Test data constants
# ---------------------------------------------------------------------------

MODEL_NAME = "test-model:v1"
PROVIDER_TYPE = "ollama"
EMBEDDING_DIM = 2

HYPERTENSION_ID = 1
DIABETES_ID = 2
ASPIRIN_ID = 3
NON_STANDARD_ID = 4

CONCEPT_RECORDS: tuple[ConceptEmbeddingRecord, ...] = (
    ConceptEmbeddingRecord(
        concept_id=HYPERTENSION_ID,
        domain_id="Condition",
        vocabulary_id="SNOMED",
        is_standard=True,
        is_valid=True,
    ),
    ConceptEmbeddingRecord(
        concept_id=DIABETES_ID,
        domain_id="Condition",
        vocabulary_id="SNOMED",
        is_standard=True,
        is_valid=True,
    ),
    ConceptEmbeddingRecord(
        concept_id=ASPIRIN_ID,
        domain_id="Drug",
        vocabulary_id="RxNorm",
        is_standard=True,
        is_valid=True,
    ),
    ConceptEmbeddingRecord(
        concept_id=NON_STANDARD_ID,
        domain_id="Drug",
        vocabulary_id="RxNorm",
        is_standard=False,
        is_valid=True,
    ),
)

# Unit vectors at known angles from QUERY_EMBEDDING.
# On unit vectors cosine and L2 rank identically
CONCEPT_ANGLES_DEGREES = {
    HYPERTENSION_ID: 40.0,
    DIABETES_ID: 10.0,
    ASPIRIN_ID: 80.0,
    NON_STANDARD_ID: 120.0,
}


def _unit_vector(degrees: float) -> list[float]:
    angle = np.radians(degrees)
    return [float(np.cos(angle)), float(np.sin(angle))]


CONCEPT_EMBEDDINGS = np.array(
    [_unit_vector(CONCEPT_ANGLES_DEGREES[r.concept_id]) for r in CONCEPT_RECORDS], dtype=np.float32
)
QUERY_EMBEDDING = np.array([[1.0, 0.0]], dtype=np.float32)

# Similarity of QUERY_EMBEDDING to each concept as omop-emb reports it.
# cosine: 1 - distance / 2 = (1 + cos(angle)) / 2. L2: 1 / (1 + distance), distance = 2 sin(angle / 2).
COSINE_SIMILARITY = {
    concept_id: float((1.0 + np.cos(np.radians(angle))) / 2.0)
    for concept_id, angle in CONCEPT_ANGLES_DEGREES.items()
}
L2_SIMILARITY = {
    concept_id: float(1.0 / (1.0 + 2.0 * np.sin(np.radians(angle) / 2.0)))
    for concept_id, angle in CONCEPT_ANGLES_DEGREES.items()
}


def register_flat(
    backend: EmbeddingBackend, dimensions: int = EMBEDDING_DIM, model_name: str = MODEL_NAME
):
    """Register *model_name* on *backend* with a FLAT index."""
    return backend.register_model(
        model_name=model_name,
        provider_type=PROVIDER_TYPE,
        index_config=FlatIndexConfig(),
        dimensions=dimensions,
    )


def populate(
    backend: EmbeddingBackend,
    records: Sequence[ConceptEmbeddingRecord],
    vectors: np.ndarray,
    model_name: str = MODEL_NAME,
) -> None:
    """Register *model_name* with the dimension of *vectors* and store *records* with *vectors*."""
    register_flat(backend, dimensions=vectors.shape[1], model_name=model_name)
    backend.upsert_embeddings(model_name=model_name, records=list(records), embeddings=vectors)


def upsert_all(backend: EmbeddingBackend) -> None:
    """Register MODEL_NAME and store CONCEPT_RECORDS with CONCEPT_EMBEDDINGS."""
    populate(backend, CONCEPT_RECORDS, CONCEPT_EMBEDDINGS)


def make_svec_backend() -> SQLiteVecEmbeddingBackend:
    """A fresh in-memory sqlite-vec backend, for tests that need more than one."""
    engine = registry_writer_engine(sqlite_resolved_database(), extensions=[_load_sqlite_vec])
    return SQLiteVecEmbeddingBackend(emb_engine=engine)


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
        OmopEmbConfig,
        "test_emb_db_sqlite",
        dialect="sqlite",
        request=request,
        schema_claims=[
            SchemaClaim(
                schema_tag=REGISTRY_SCHEMA_KEY,
                physical_schema=MODEL_REGISTRY_SCHEMA,
                reserved=True,
                owner="omop_emb",
            )
        ],
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
    from oa_configurator.testing import install_postgres_extension, isolated_test_database

    with isolated_test_database(
        OmopEmbConfig,
        "test_emb_db_pg",
        extensions=[install_postgres_extension("vector")],
        request=request,
    ) as db:
        yield db


class FakeStoreReader(EmbeddingStoreReader):
    """An initialized store serving fixed identity records.

    Only what population planning reads is implemented; every other member of the
    reader interface raises ``NotImplementedError``.

    Parameters
    ----------
    records : Iterable[ConceptEmbeddingRecord]
        Records returned by ``iter_stored_embeddings``.
    expected_batch_size : int, optional
        When given, ``iter_stored_embeddings`` asserts it receives this batch size.
    """

    def __init__(
        self,
        records: Iterable[ConceptEmbeddingRecord] = (),
        *,
        expected_batch_size: int | None = None,
    ) -> None:
        self._records = tuple(records)
        self._expected_batch_size = expected_batch_size

    @property
    def initialized(self) -> bool:
        return True

    def iter_stored_embeddings(
        self, model_name: str, *, batch_size: int = 10_000
    ) -> Iterator[ConceptEmbeddingRecord]:
        if self._expected_batch_size is not None:
            assert batch_size == self._expected_batch_size
        return iter(self._records)

    @property
    def backend_type(self) -> BackendType:
        raise NotImplementedError

    def get_registered_model(self, *, model_name: str) -> EmbeddingModelRecord | None:
        raise NotImplementedError

    def get_registered_models(
        self, *, model_name: str | None = None, provider_type: str | None = None
    ) -> tuple[EmbeddingModelRecord, ...]:
        raise NotImplementedError

    def has_any_embeddings(self, *, model_name: str) -> bool:
        raise NotImplementedError

    def get_stored_concept_ids(
        self, *, model_name: str, concept_filter: EmbeddingConceptFilter | None = None
    ) -> set[int]:
        raise NotImplementedError

    def get_embeddings_by_concept_ids(
        self, *, model_name: str, concept_ids: Sequence[int]
    ) -> Mapping[int, np.ndarray]:
        raise NotImplementedError

    def get_concept_filter_metadata(
        self, *, model_name: str, concept_ids: Sequence[int]
    ) -> Mapping[int, ConceptEmbeddingRecord]:
        raise NotImplementedError

    def get_embedding_count(self, *, model_name: str) -> int:
        raise NotImplementedError

    def get_embedding_count_by_vocabulary(self, *, model_name: str) -> Mapping[str, int]:
        raise NotImplementedError

    def get_nearest_concepts(
        self,
        *,
        model_name: str,
        metric_type: MetricType,
        query_embeddings: EmbeddingArray,
        concept_filter: EmbeddingConceptFilter | None = None,
        k: int = 10,
    ) -> tuple[tuple[NearestConceptMatch, ...], ...]:
        raise NotImplementedError

    def physical_indexes(self, model_name: str) -> tuple[str, ...]:
        raise NotImplementedError

    def drop_index_sql(self, model_name: str) -> tuple[str, ...]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> Self:
        raise NotImplementedError

    def __exit__(self, *_exc_info: object) -> None:
        raise NotImplementedError


def require_cdm(resolved: ResolvedDatabase | None) -> ResolvedCDMDatabase:
    """Narrow a resolved database to the CDM kind the CDM session factory requires."""
    assert isinstance(resolved, ResolvedCDMDatabase)
    return resolved


def table_of(model: Any) -> sa.Table:
    """The Table a mapped class is bound to, narrowed from the declarative base's FromClause."""
    table = model.__table__
    assert isinstance(table, sa.Table)
    return table


def registered_model(store: Any, model_name: str) -> EmbeddingModelRecord:
    """The record registered for *model_name* on *store*, which must exist."""
    record = store.get_registered_model(model_name=model_name)
    assert record is not None, f"{model_name!r} is not registered"
    return record


def drop_all_models(backend: EmbeddingBackend[Any]) -> None:
    """Delete every model registered on *backend*, dropping its embedding table."""
    for record in backend.get_registered_models():
        backend.delete_model(model_name=record.model_name)


@pytest.fixture
def pg_backend(pg_db):
    """Function-scoped PGVectorEmbeddingBackend with a clean registry per test,
    built through ``_open_writer()`` so its engine claims the registry
    schema and installs pgvector exactly as production does."""
    from omop_emb.backends.base_backend import _open_writer
    from omop_emb.backends.embedding_table import EmbeddingTableBase

    backend = _open_writer("pgvector", database=pg_db.resolved)

    yield backend

    try:
        drop_all_models(backend)
    finally:
        # Remove the tables from the ORM cache
        EmbeddingTableBase.metadata.clear()
        EmbeddingTableBase.registry._class_registry.clear()
    backend.emb_engine.dispose()


@pytest.fixture
def pg_engine(pg_backend) -> sa.Engine:
    """``pg_backend``'s own committing engine: registry schema claimed and
    pgvector installed. Isolation comes from ``pg_backend``'s teardown."""
    return pg_backend.emb_engine
