"""Integration tests for the pgvector embedding backend.

Requires a running PostgreSQL instance with the pgvector extension.
Set TEST_DB_HOST and TEST_DB_PORT to enable these tests.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import numpy as np
import pytest

from tests.conftest import registered_model

pytest.importorskip("pgvector", reason="omop-emb[pgvector] not installed: skipping pgvector tests")

import sqlalchemy as sa
from oa_configurator import Role, physical_schema_of
from oa_configurator.testing import resolve_with_role_schemas, scoped_test_schema
from sqlalchemy import event as sa_event

from omop_emb.backends.base_backend import ConceptEmbeddingRecord, _open_writer
from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.backends.pgvector import PGVectorEmbeddingBackend
from omop_emb.config import IndexType, MetricType
from omop_emb.model_registry import RegistryManager, registry_reader_engine
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter

from .conftest import (
    CONCEPT_EMBEDDINGS,
    CONCEPT_RECORDS,
    EMBEDDING_DIM,
    HYPERTENSION_ID,
    L2_SIMILARITY,
    MODEL_NAME,
    PROVIDER_TYPE,
    QUERY_EMBEDDING,
    populate,
    register_flat,
    upsert_all,
)
from .shared_backend_tests import SharedBackendTests


@pytest.mark.pgvector
@pytest.mark.integration
class TestPGVectorBackend(SharedBackendTests):
    """Runs the full shared suite against a pgvector backend."""

    @pytest.fixture
    def backend(self, pg_backend: PGVectorEmbeddingBackend):
        return pg_backend

    def test_l2_similarity_values(self, backend: PGVectorEmbeddingBackend):
        """L2 sim = 1 / (1 + dist), checked on the shared unit-vector dataset."""
        upsert_all(backend)
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.L2,
            query_embeddings=QUERY_EMBEDDING,
            k=len(CONCEPT_RECORDS),
        )
        assert {m.concept_id: m.similarity for m in results[0]} == pytest.approx(
            L2_SIMILARITY, abs=1e-5
        )

    def test_query_vectors_run_as_separate_statements(self, backend: PGVectorEmbeddingBackend):
        """Each query vector gets its own ordered statement and the same result as a lone query."""
        upsert_all(backend)
        queries = CONCEPT_EMBEDDINGS[[0, 1, 2]]
        statements: list[str] = []

        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement.lower())

        sa_event.listen(backend.emb_engine, "before_cursor_execute", capture)
        try:
            batched = backend.get_nearest_concepts(
                model_name=MODEL_NAME, metric_type=MetricType.COSINE, query_embeddings=queries, k=3
            )
        finally:
            sa_event.remove(backend.emb_engine, "before_cursor_execute", capture)

        selects = [s for s in statements if "order by" in s]
        assert len(selects) == len(queries)
        assert not any("lateral" in s for s in selects)
        for i, query in enumerate(queries):
            alone = backend.get_nearest_concepts(
                model_name=MODEL_NAME,
                metric_type=MetricType.COSINE,
                query_embeddings=query[None, :],
                k=3,
            )
            assert batched[i] == alone[0]


@contextmanager
def captured_statements(engine):
    statements: list[str] = []

    def capture(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.strip().lower())

    sa_event.listen(engine, "before_cursor_execute", capture)
    try:
        yield statements
    finally:
        sa_event.remove(engine, "before_cursor_execute", capture)


@pytest.mark.pgvector
@pytest.mark.integration
class TestPGVectorHNSWBackend:
    """pgvector-specific HNSW index behaviour."""

    HNSW_CONFIG = HNSWIndexConfig(
        metric_type=MetricType.L2, num_neighbors=4, ef_search=8, ef_construction=16
    )

    def test_rebuild_index_keeps_storage_identifier(self, pg_backend: PGVectorEmbeddingBackend):
        """Rebuilding from FLAT to HNSW keeps the same physical table name."""
        r_flat = register_flat(pg_backend)
        r_hnsw = pg_backend.rebuild_index(
            model_name=MODEL_NAME,
            index_config=self.HNSW_CONFIG,
        )
        assert r_flat.storage_identifier == r_hnsw.storage_identifier
        assert r_hnsw.index_type == IndexType.HNSW
        assert r_hnsw.metric_type == MetricType.L2

    def test_rebuild_leaves_exactly_the_configured_index(
        self, pg_backend: PGVectorEmbeddingBackend
    ):
        """Switching metric replaces the HNSW index; reverting to FLAT drops it."""
        upsert_all(pg_backend)
        table = registered_model(pg_backend, MODEL_NAME).storage_identifier

        pg_backend.rebuild_index(
            model_name=MODEL_NAME, index_config=HNSWIndexConfig(metric_type=MetricType.COSINE)
        )
        assert pg_backend.physical_indexes(MODEL_NAME) == (f"idx_{table}_cosine",)

        pg_backend.rebuild_index(
            model_name=MODEL_NAME, index_config=HNSWIndexConfig(metric_type=MetricType.L2)
        )
        assert pg_backend.physical_indexes(MODEL_NAME) == (f"idx_{table}_l2",)

        pg_backend.rebuild_index(model_name=MODEL_NAME, index_config=FlatIndexConfig())
        assert pg_backend.physical_indexes(MODEL_NAME) == ()

    def test_rebuild_with_same_metric_applies_new_build_parameters(
        self, pg_backend: PGVectorEmbeddingBackend, pg_engine
    ):
        upsert_all(pg_backend)
        table = registered_model(pg_backend, MODEL_NAME).storage_identifier
        for m in (4, 8):
            pg_backend.rebuild_index(
                model_name=MODEL_NAME,
                index_config=HNSWIndexConfig(metric_type=MetricType.L2, num_neighbors=m),
            )
        with pg_engine.connect() as conn:
            indexdef = conn.execute(
                sa.text("SELECT indexdef FROM pg_indexes WHERE indexname = :n"),
                {"n": f"idx_{table}_l2"},
            ).scalar_one()
        assert "m='8'" in indexdef

    def test_rebuild_rejects_a_metric_the_backend_cannot_index(
        self, pg_backend: PGVectorEmbeddingBackend
    ):
        upsert_all(pg_backend)
        pg_backend.rebuild_index(model_name=MODEL_NAME, index_config=self.HNSW_CONFIG)
        before = pg_backend.physical_indexes(MODEL_NAME)
        with pytest.raises(ValueError, match="hamming"):
            pg_backend.rebuild_index(
                model_name=MODEL_NAME, index_config=HNSWIndexConfig(metric_type=MetricType.HAMMING)
            )
        assert pg_backend.physical_indexes(MODEL_NAME) == before
        assert registered_model(pg_backend, MODEL_NAME).index_config == self.HNSW_CONFIG

    @pytest.mark.parametrize("vacuum,expected", [(True, 1), (False, 0)])
    def test_rebuild_index_vacuums_the_table_unless_told_not_to(
        self, pg_backend: PGVectorEmbeddingBackend, vacuum: bool, expected: int
    ):
        upsert_all(pg_backend)
        statements: list[str] = []

        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement.strip().lower())

        sa_event.listen(pg_backend.emb_engine, "before_cursor_execute", capture)
        try:
            pg_backend.rebuild_index(
                model_name=MODEL_NAME, index_config=self.HNSW_CONFIG, vacuum=vacuum
            )
        finally:
            sa_event.remove(pg_backend.emb_engine, "before_cursor_execute", capture)

        assert sum(s.startswith("vacuum (analyze)") for s in statements) == expected
        assert registered_model(pg_backend, MODEL_NAME).index_config == self.HNSW_CONFIG

    def test_compact_storage_shrinks_the_table_after_deletes(
        self, pg_backend: PGVectorEmbeddingBackend
    ):
        """Plain VACUUM leaves the table size alone, VACUUM FULL gives the deleted rows' space back."""
        ids = list(range(1, 3001))
        populate(
            pg_backend,
            [ConceptEmbeddingRecord(i, "Condition", "SNOMED", True, True) for i in ids],
            np.random.default_rng(0).normal(size=(len(ids), EMBEDDING_DIM)).astype(np.float32),
        )
        pg_backend.delete_embeddings(model_name=MODEL_NAME, concept_ids=ids[:2700])

        with captured_statements(pg_backend.emb_engine) as statements:
            result = pg_backend.compact_storage(model_name=MODEL_NAME)

        assert any(s.startswith("vacuum (full)") for s in statements)
        assert result.size_after < result.size_before
        assert pg_backend.get_embedding_count(model_name=MODEL_NAME) == 300

    def test_query_applies_registry_ef_search_to_its_transaction_only(
        self, pg_backend: PGVectorEmbeddingBackend, pg_db
    ):
        """A store that never ran rebuild_index() still applies the registry's ef_search, scoped with SET LOCAL."""
        upsert_all(pg_backend)
        pg_backend.rebuild_index(model_name=MODEL_NAME, index_config=self.HNSW_CONFIG)

        statements: list[str] = []

        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement.strip().lower())

        fresh = _open_writer("pgvector", database=pg_db.resolved)
        try:
            sa_event.listen(fresh.emb_engine, "before_cursor_execute", capture)
            fresh.get_nearest_concepts(
                model_name=MODEL_NAME,
                metric_type=MetricType.L2,
                query_embeddings=CONCEPT_EMBEDDINGS[[HYPERTENSION_ID - 1]],
                k=1,
            )
            with fresh.emb_engine.connect() as conn:
                ef_search_after = conn.execute(sa.text("SHOW hnsw.ef_search")).scalar_one()
        finally:
            fresh.emb_engine.dispose()

        assert f"set local hnsw.ef_search = {self.HNSW_CONFIG.ef_search}" in statements
        assert ef_search_after == "40"

    def _hnsw_query(self, backend: PGVectorEmbeddingBackend, *, k: int, filtered: bool):
        return backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.L2,
            query_embeddings=QUERY_EMBEDDING,
            concept_filter=EmbeddingConceptFilter(domains=("Condition",)) if filtered else None,
            k=k,
        )[0]

    @pytest.mark.parametrize("filtered", [False, True])
    def test_iterative_scan_is_relaxed_and_results_are_sorted(
        self, pg_backend: PGVectorEmbeddingBackend, filtered: bool
    ):
        """Where pgvector supports it, the HNSW scan is iterative and rows come back ordered by similarity."""
        if not pg_backend._supports_iterative_scan:
            pytest.skip("pgvector below 0.8.0")
        upsert_all(pg_backend)
        pg_backend.rebuild_index(model_name=MODEL_NAME, index_config=self.HNSW_CONFIG)

        with captured_statements(pg_backend.emb_engine) as statements:
            matches = self._hnsw_query(pg_backend, k=2, filtered=filtered)

        assert "set local hnsw.iterative_scan = relaxed_order" in statements
        assert "set local hnsw.iterative_scan = strict_order" not in statements
        similarities = [m.similarity for m in matches]
        assert len(matches) == 2
        assert similarities == sorted(similarities, reverse=True)

    def test_filtered_query_returns_k_rows_after_a_smaller_one(
        self, pg_backend: PGVectorEmbeddingBackend
    ):
        """The effective ef_search does not stay at the value of an earlier, smaller k."""
        upsert_all(pg_backend)
        pg_backend.rebuild_index(model_name=MODEL_NAME, index_config=self.HNSW_CONFIG)

        assert len(self._hnsw_query(pg_backend, k=1, filtered=True)) == 1
        assert len(self._hnsw_query(pg_backend, k=2, filtered=True)) == 2

    @pytest.mark.parametrize("k,expected_ef_search", [(1, 8), (50, 50), (5000, 1000)])
    def test_without_iterative_scan_ef_search_covers_k(
        self, pg_backend: PGVectorEmbeddingBackend, monkeypatch, k: int, expected_ef_search: int
    ):
        """On pgvector below 0.8.0, ef_search is max(configured, k) capped at 1000, with no iterative scan."""
        upsert_all(pg_backend)
        pg_backend.rebuild_index(model_name=MODEL_NAME, index_config=self.HNSW_CONFIG)
        monkeypatch.setattr(pg_backend, "_supports_iterative_scan", False)

        with captured_statements(pg_backend.emb_engine) as statements:
            self._hnsw_query(pg_backend, k=k, filtered=False)

        assert f"set local hnsw.ef_search = {expected_ef_search}" in statements
        assert not any("iterative_scan" in s for s in statements)

    def test_without_iterative_scan_filtered_queries_warn_once(
        self, pg_backend: PGVectorEmbeddingBackend, monkeypatch, caplog
    ):
        upsert_all(pg_backend)
        pg_backend.rebuild_index(model_name=MODEL_NAME, index_config=self.HNSW_CONFIG)
        monkeypatch.setattr(pg_backend, "_supports_iterative_scan", False)
        monkeypatch.setattr(pg_backend, "_filtered_hnsw_warned", False)

        with caplog.at_level(logging.WARNING, logger="omop_emb.backends.pgvector.pg_backend"):
            self._hnsw_query(pg_backend, k=1, filtered=False)
            assert not caplog.records
            self._hnsw_query(pg_backend, k=1, filtered=True)
            self._hnsw_query(pg_backend, k=1, filtered=True)

        assert len(caplog.records) == 1
        assert "pgvector below 0.8.0" in caplog.records[0].getMessage()

    def test_hnsw_search_returns_correct_top1(self, pg_backend: PGVectorEmbeddingBackend):
        upsert_all(pg_backend)
        pg_backend.rebuild_index(
            model_name=MODEL_NAME,
            index_config=self.HNSW_CONFIG,
        )
        results = pg_backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.L2,
            query_embeddings=CONCEPT_EMBEDDINGS[[HYPERTENSION_ID - 1]],
            k=1,
        )
        assert results[0][0].concept_id == HYPERTENSION_ID


@pytest.mark.pgvector
@pytest.mark.integration
def test_long_provider_model_variants_have_independent_tables_and_indexes(
    pg_backend: PGVectorEmbeddingBackend,
):
    variants = (
        "hf.co/second-state/multilingual-e5-large-instruct-GGUF:Q8_0",
        "hf.co/second-state/multilingual-e5-large-instruct-GGUF:Q5_0",
        "hf.co/second-state/multilingual-e5-large-instruct-GGUF:Q4_K_M",
    )
    records = {}
    try:
        for model_name in variants:
            records[model_name] = pg_backend.register_model(
                model_name=model_name,
                provider_type=PROVIDER_TYPE,
                index_config=FlatIndexConfig(),
                dimensions=EMBEDDING_DIM,
            )
        table_names = {record.storage_identifier for record in records.values()}
        assert len(table_names) == len(variants)
        inspector = sa.inspect(pg_backend.emb_engine)
        schema = physical_schema_of(pg_backend.emb_engine)
        assert all(inspector.has_table(table, schema=schema) for table in table_names)

        for model_name in variants:
            pg_backend.rebuild_index(
                model_name=model_name,
                index_config=HNSWIndexConfig(metric_type=MetricType.L2),
            )
        initial_indexes = {name: pg_backend.physical_indexes(name) for name in variants}
        all_indexes = {indexes[0] for indexes in initial_indexes.values()}
        assert len(all_indexes) == len(variants)
        assert all(
            indexes == (f"idx_{records[name].storage_identifier}_l2",)
            for name, indexes in initial_indexes.items()
        )

        first, *other_variants = variants
        pg_backend.rebuild_index(
            model_name=first,
            index_config=HNSWIndexConfig(metric_type=MetricType.COSINE),
        )
        assert pg_backend.physical_indexes(first) == (
            f"idx_{records[first].storage_identifier}_cosine",
        )
        assert all(
            pg_backend.physical_indexes(name) == initial_indexes[name] for name in other_variants
        )

        pg_backend.rebuild_index(model_name=first, index_config=FlatIndexConfig())
        assert pg_backend.physical_indexes(first) == ()
        assert all(
            pg_backend.physical_indexes(name) == initial_indexes[name] for name in other_variants
        )
    finally:
        for model_name in records:
            if pg_backend.get_registered_model(model_name=model_name) is not None:
                pg_backend.delete_model(model_name=model_name)


@pytest.mark.pgvector
@pytest.mark.integration
class TestPGVectorNonDefaultSchema:
    """Every method here defaulted to the public schema in existing coverage,
    so a bug that silently ignored schema_translate_map would still pass
    every other test in this file. This is what actually catches that.
    """

    HNSW_CONFIG = HNSWIndexConfig(
        metric_type=MetricType.L2, num_neighbors=4, ef_search=8, ef_construction=16
    )

    @pytest.fixture
    def scoped_backend(self, pg_db):
        with scoped_test_schema(pg_db.resolved, prefix="emb_schema") as scoped:
            backend = _open_writer("pgvector", database=scoped.resolved)
            try:
                yield backend, scoped.schemas[Role.PRIMARY]
            finally:
                backend.emb_engine.dispose()

    def test_table_and_index_lifecycle_stays_in_the_configured_schema(self, scoped_backend, pg_db):
        pg_engine = pg_db.committing_engine
        backend, schema = scoped_backend
        record = register_flat(backend)
        backend.upsert_embeddings(
            model_name=MODEL_NAME,
            records=list(CONCEPT_RECORDS),
            embeddings=CONCEPT_EMBEDDINGS,
        )

        # table_exists() sees it in the configured schema...
        assert backend._storage_table_exists(record) is True
        # ...and a bare inspector scoped to "public" doesn't.
        assert sa.inspect(pg_engine).has_table(record.storage_identifier, schema="public") is False

        # Rebuild to HNSW, confirm the index lands in the configured schema,
        # then revert to FLAT, which drops it from that schema.
        backend.rebuild_index(model_name=MODEL_NAME, index_config=self.HNSW_CONFIG)
        index_name = f"idx_{record.storage_identifier}_l2"
        assert backend.physical_indexes(MODEL_NAME) == (index_name,)
        indexes_in_schema = sa.inspect(pg_engine).get_indexes(
            record.storage_identifier, schema=schema
        )
        assert any(idx["name"] == index_name for idx in indexes_in_schema)
        backend.rebuild_index(model_name=MODEL_NAME, index_config=FlatIndexConfig())
        assert backend.physical_indexes(MODEL_NAME) == ()

        # drop_pg_embedding_table(): drops from the configured schema, not public.
        backend.delete_model(model_name=MODEL_NAME)
        assert backend._storage_table_exists(record) is False

    def test_model_registry_lives_in_its_own_reserved_schema(self, scoped_backend, pg_db):
        pg_engine = pg_db.committing_engine
        backend, schema = scoped_backend
        register_flat(backend)

        registry = RegistryManager(backend.emb_engine)
        assert registry.registry_available is True
        assert len(registry.get_registered_models(model_name=MODEL_NAME)) == 1

        # A registry built with a completely different primary-tagged schema
        # sees the exact same row: the registry is decoupled from it entirely.
        public_resolved = resolve_with_role_schemas(
            pg_db.resolved, {role: "public" for role in pg_db.resolved.schema_tags()}
        )
        public_engine = registry_reader_engine(public_resolved)
        try:
            public_registry = RegistryManager(public_engine)
            assert public_registry.registry_available is True
            assert len(public_registry.get_registered_models(model_name=MODEL_NAME)) == 1
        finally:
            public_engine.dispose()

        # And genuinely never created under the storage schema's own name.
        assert sa.inspect(pg_engine).has_table("model_registry", schema=schema) is False


@pytest.mark.pgvector
@pytest.mark.integration
def test_single_upsert_call_exceeds_the_bind_parameter_limit(pg_backend: PGVectorEmbeddingBackend):
    """12,000 rows x 6 columns is past PostgreSQL's 65,535 bind parameters for one statement."""
    from omop_emb.backends.embedding_table import ConceptEmbeddingRecord

    n = 12_000
    register_flat(pg_backend)
    records = [ConceptEmbeddingRecord(i, "Drug", "RxNorm", True, True) for i in range(n)]
    pg_backend.upsert_embeddings(
        model_name=MODEL_NAME,
        records=records,
        embeddings=np.ones((n, EMBEDDING_DIM), dtype=np.float32),
    )
    updated = [ConceptEmbeddingRecord(i, "Condition", "SNOMED", False, True) for i in range(n)]
    pg_backend.upsert_embeddings(
        model_name=MODEL_NAME,
        records=updated,
        embeddings=np.full((n, EMBEDDING_DIM), 2.0, dtype=np.float32),
    )
    assert pg_backend.get_embedding_count(model_name=MODEL_NAME) == n
    assert (
        pg_backend.get_concept_filter_metadata(model_name=MODEL_NAME, concept_ids=[n - 1])[n - 1]
        == updated[-1]
    )
    assert np.array_equal(
        pg_backend.get_embeddings_by_concept_ids(model_name=MODEL_NAME, concept_ids=[0])[0],
        np.full(EMBEDDING_DIM, 2.0, dtype=np.float32),
    )
