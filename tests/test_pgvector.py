"""Integration tests for the pgvector embedding backend.

Requires a running PostgreSQL instance with the pgvector extension.
Set TEST_DB_HOST and TEST_DB_PORT to enable these tests.
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
import pytest
import sqlalchemy as sa
from oa_configurator import Role, physical_schema_of, qualified
from oa_configurator.testing import resolve_with_role_schemas, scoped_test_schema
from sqlalchemy import event as sa_event

from omop_emb.backends.base_backend import ConceptEmbeddingRecord, _open_writer
from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.backends.index_naming import IndexNames
from omop_emb.backends.pgvector import PGVectorEmbeddingBackend
from omop_emb.config import BackendType, IndexType, MetricType, VectorColumnType, VectorPrecision
from omop_emb.model_registry import ModelRegistry, RegistryManager, registry_reader_engine
from omop_emb.model_registry.model_registry_orm import resolve_registry_physical_schema
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter
from omop_emb.utils.errors import UnsupportedIndexConfigError
from tests.conftest import registered_model

from .conftest import (
    CONCEPT_EMBEDDINGS,
    CONCEPT_RECORDS,
    EMBEDDING_DIM,
    HYPERTENSION_ID,
    L2_SIMILARITY,
    MODEL_NAME,
    PROVIDER_TYPE,
    QUERY_EMBEDDING,
    drop_all_models,
    populate,
    register_flat,
    upsert_all,
)
from .shared_backend_tests import SharedBackendTests


@pytest.mark.pgvector
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
        queries = CONCEPT_EMBEDDINGS[:3]
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
        for i in range(len(queries)):
            alone = backend.get_nearest_concepts(
                model_name=MODEL_NAME,
                metric_type=MetricType.COSINE,
                query_embeddings=queries[i : i + 1],
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
        with pytest.raises(ValueError, match=MetricType.HAMMING.value):
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

        fresh = _open_writer(BackendType.PGVECTOR, database=pg_db.resolved)
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
        assert pg_backend._supports_iterative_scan, "this test needs pgvector 0.8.0 or later"
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

    def test_max_scan_tuples_is_set_per_transaction_only_when_configured(
        self, pg_backend: PGVectorEmbeddingBackend
    ):
        """A configured scan cap is applied with SET LOCAL, and an unset one leaves the server default."""
        assert pg_backend._supports_iterative_scan, "this test needs pgvector 0.8.0 or later"
        upsert_all(pg_backend)
        pg_backend.rebuild_index(model_name=MODEL_NAME, index_config=self.HNSW_CONFIG)

        with captured_statements(pg_backend.emb_engine) as statements:
            self._hnsw_query(pg_backend, k=2, filtered=True)
        assert not any("max_scan_tuples" in s for s in statements)

        pg_backend.set_hnsw_search_params(model_name=MODEL_NAME, max_scan_tuples=5000)
        with captured_statements(pg_backend.emb_engine) as statements:
            self._hnsw_query(pg_backend, k=2, filtered=True)
        assert "set local hnsw.max_scan_tuples = 5000" in statements
        with pg_backend.emb_engine.connect() as conn:
            assert conn.execute(sa.text("SHOW hnsw.max_scan_tuples")).scalar_one() == "20000"

    def test_set_hnsw_search_params_keeps_the_other_settings(
        self, pg_backend: PGVectorEmbeddingBackend
    ):
        """Updating one query-time setting leaves the other and the graph settings as they were."""
        upsert_all(pg_backend)
        pg_backend.rebuild_index(model_name=MODEL_NAME, index_config=self.HNSW_CONFIG)

        record = pg_backend.set_hnsw_search_params(model_name=MODEL_NAME, max_scan_tuples=50_000)
        assert record.index_config == replace(self.HNSW_CONFIG, max_scan_tuples=50_000)

        record = pg_backend.set_hnsw_search_params(model_name=MODEL_NAME, ef_search=64)
        assert record.index_config == replace(
            self.HNSW_CONFIG, ef_search=64, max_scan_tuples=50_000
        )
        assert registered_model(pg_backend, MODEL_NAME).index_config == record.index_config

    def test_set_hnsw_search_params_rejects_a_flat_model(
        self, pg_backend: PGVectorEmbeddingBackend
    ):
        register_flat(pg_backend)
        with pytest.raises(UnsupportedIndexConfigError, match="no HNSW index"):
            pg_backend.set_hnsw_search_params(model_name=MODEL_NAME, ef_search=64)

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
            backend = _open_writer(BackendType.PGVECTOR, database=scoped.resolved)
            try:
                yield backend, scoped.schemas[Role.PRIMARY]
            finally:
                drop_all_models(backend)
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


@pytest.mark.pgvector
class TestPGVectorPrecision:
    """Storage and index precision on pgvector."""

    HNSW_L2 = HNSWIndexConfig(metric_type=MetricType.L2, num_neighbors=4, ef_construction=16)

    @staticmethod
    def _column_type(backend: PGVectorEmbeddingBackend, model_name: str) -> str:
        record = registered_model(backend, model_name)
        with backend.emb_engine.connect() as conn:
            return conn.execute(
                sa.text(
                    "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
                    "JOIN pg_class c ON c.oid = a.attrelid "
                    "WHERE c.relname = :t AND a.attname = 'embedding'"
                ),
                {"t": record.storage_identifier},
            ).scalar_one()

    def test_half_storage_creates_a_halfvec_column_that_answers_queries(
        self, pg_backend: PGVectorEmbeddingBackend
    ):
        record = pg_backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            dimensions=EMBEDDING_DIM,
            storage_precision=VectorPrecision.FLOAT16,
        )
        pg_backend.upsert_embeddings(
            model_name=MODEL_NAME, records=list(CONCEPT_RECORDS), embeddings=CONCEPT_EMBEDDINGS
        )

        assert record.storage_precision == VectorPrecision.FLOAT16
        assert self._column_type(pg_backend, MODEL_NAME) == f"halfvec({EMBEDDING_DIM})"
        results = pg_backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.L2,
            query_embeddings=CONCEPT_EMBEDDINGS[[HYPERTENSION_ID - 1]],
            k=1,
        )
        assert results[0][0].concept_id == HYPERTENSION_ID

    def test_default_storage_is_float32_and_recorded(self, pg_backend: PGVectorEmbeddingBackend):
        record = register_flat(pg_backend)
        assert record.storage_precision == VectorPrecision.FLOAT32
        assert self._column_type(pg_backend, MODEL_NAME) == f"vector({EMBEDDING_DIM})"

    def test_wide_model_defaults_to_half_storage(self, pg_backend: PGVectorEmbeddingBackend):
        record = register_flat(pg_backend, dimensions=2500)
        assert record.storage_precision == VectorPrecision.FLOAT16
        assert self._column_type(pg_backend, MODEL_NAME) == "halfvec(2500)"

    def test_wide_model_can_keep_float32_and_index_over_a_half_cast(
        self, pg_backend: PGVectorEmbeddingBackend
    ):
        pg_backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            dimensions=2500,
            storage_precision=VectorPrecision.FLOAT32,
        )
        assert self._column_type(pg_backend, MODEL_NAME) == "vector(2500)"

        with pytest.raises(
            UnsupportedIndexConfigError, match=f"index precision '{VectorPrecision.FLOAT16.value}'"
        ):
            pg_backend.rebuild_index(model_name=MODEL_NAME, index_config=self.HNSW_L2)

        record = pg_backend.rebuild_index(
            model_name=MODEL_NAME,
            index_config=replace(self.HNSW_L2, index_precision=VectorPrecision.FLOAT16),
        )
        assert isinstance(record.index_config, HNSWIndexConfig)
        assert record.index_config.index_precision == VectorPrecision.FLOAT16
        (index_name,) = pg_backend.physical_indexes(MODEL_NAME)
        with pg_backend.emb_engine.connect() as conn:
            definition = conn.execute(
                sa.text("SELECT indexdef FROM pg_indexes WHERE indexname = :n"),
                {"n": index_name},
            ).scalar_one()
        assert "halfvec(2500)" in definition
        assert "halfvec_l2_ops" in definition

    def test_registering_again_without_a_precision_keeps_the_stored_one(
        self, pg_backend: PGVectorEmbeddingBackend
    ):
        pg_backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            dimensions=EMBEDDING_DIM,
            storage_precision=VectorPrecision.FLOAT16,
        )
        record = register_flat(pg_backend)
        assert record.storage_precision == VectorPrecision.FLOAT16

    def test_half_index_serves_the_query_through_the_cast_expression(
        self, pg_backend: PGVectorEmbeddingBackend
    ):
        upsert_all(pg_backend)
        pg_backend.rebuild_index(
            model_name=MODEL_NAME,
            index_config=replace(self.HNSW_L2, index_precision=VectorPrecision.FLOAT16),
        )

        captured: list[tuple[str, Any]] = []

        def capture(_connection, _cursor, statement, parameters, _context, _many):
            if "order by" in statement.lower():
                captured.append((statement, parameters))

        sa_event.listen(pg_backend.emb_engine, "before_cursor_execute", capture)
        results = pg_backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.L2,
            query_embeddings=CONCEPT_EMBEDDINGS[[HYPERTENSION_ID - 1]],
            k=1,
        )
        assert results[0][0].concept_id == HYPERTENSION_ID

        (statement, parameters), *_ = captured
        assert VectorColumnType.HALFVEC.value in statement.lower()
        with pg_backend.emb_engine.connect() as conn:
            conn.exec_driver_sql("SET enable_seqscan = off")
            plan = "\n".join(
                row[0] for row in conn.exec_driver_sql("EXPLAIN " + statement, parameters)
            )
        assert "Index Scan using idx_" in plan

    def test_float32_index_leaves_the_query_uncast(self, pg_backend: PGVectorEmbeddingBackend):
        upsert_all(pg_backend)
        pg_backend.rebuild_index(model_name=MODEL_NAME, index_config=self.HNSW_L2)

        captured: list[str] = []

        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            if "order by" in statement.lower():
                captured.append(statement.lower())

        sa_event.listen(pg_backend.emb_engine, "before_cursor_execute", capture)
        pg_backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.L2,
            query_embeddings=CONCEPT_EMBEDDINGS[[HYPERTENSION_ID - 1]],
            k=1,
        )
        assert captured and VectorColumnType.HALFVEC.value not in captured[0]

    @pytest.fixture
    def wide_registration_warnings(self, caplog):
        caplog.set_level(logging.WARNING)
        return lambda: [r for r in caplog.records if "by default" in r.getMessage()]

    def test_defaulting_a_wide_model_to_half_storage_warns_once(
        self, pg_backend: PGVectorEmbeddingBackend, wide_registration_warnings
    ):
        register_flat(pg_backend, dimensions=2500)
        assert len(wide_registration_warnings()) == 1
        register_flat(pg_backend, dimensions=2500)
        assert len(wide_registration_warnings()) == 1

    def test_choosing_float32_for_a_wide_model_never_warns_about_a_default(
        self, pg_backend: PGVectorEmbeddingBackend, wide_registration_warnings
    ):
        pg_backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            dimensions=2500,
            storage_precision=VectorPrecision.FLOAT32,
        )
        register_flat(pg_backend, dimensions=2500)
        assert wide_registration_warnings() == []

    def test_narrow_model_default_does_not_warn(
        self, pg_backend: PGVectorEmbeddingBackend, wide_registration_warnings
    ):
        register_flat(pg_backend)
        assert wide_registration_warnings() == []

    def test_older_record_without_a_precision_derives_it_from_the_dimensions(
        self, pg_backend: PGVectorEmbeddingBackend
    ):
        register_flat(pg_backend, dimensions=2500)
        with pg_backend.emb_engine.begin() as conn:
            table = qualified(
                conn,
                ModelRegistry.__tablename__,
                physical_schema=resolve_registry_physical_schema(conn),
            )
            conn.execute(sa.text(f"UPDATE {table} SET storage_precision = NULL"))

        record = registered_model(pg_backend, MODEL_NAME)
        assert record.storage_precision is None
        assert pg_backend.storage_precision_of(record) == VectorPrecision.FLOAT16
        pg_backend.upsert_embeddings(
            model_name=MODEL_NAME,
            records=[CONCEPT_RECORDS[0]],
            embeddings=np.ones((1, 2500), dtype=np.float32),
        )
        assert pg_backend.get_embedding_count(model_name=MODEL_NAME) == 1

    def test_bundle_round_trip_keeps_half_storage(
        self, pg_backend: PGVectorEmbeddingBackend, tmp_path
    ):
        from omop_emb.storage import embedding_bundle

        pg_backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            dimensions=EMBEDDING_DIM,
            storage_precision=VectorPrecision.FLOAT16,
        )
        pg_backend.upsert_embeddings(
            model_name=MODEL_NAME, records=list(CONCEPT_RECORDS), embeddings=CONCEPT_EMBEDDINGS
        )
        meta, bundle_path = embedding_bundle.export_bundle(
            backend=pg_backend, model_name=MODEL_NAME, output_dir=tmp_path
        )
        assert meta.storage_precision == VectorPrecision.FLOAT16

        pg_backend.delete_model(model_name=MODEL_NAME)
        embedding_bundle.import_bundle(backend=pg_backend, h5_path=bundle_path)

        record = registered_model(pg_backend, MODEL_NAME)
        assert record.storage_precision == VectorPrecision.FLOAT16
        assert self._column_type(pg_backend, MODEL_NAME) == f"halfvec({EMBEDDING_DIM})"


@pytest.mark.pgvector
class TestPGVectorPartialIndexes:
    """Per-domain partial HNSW indexes built next to the global index."""

    DOMAIN_ROWS: ClassVar[dict[str, int]] = {
        "Condition": 4,
        "Drug": 3,
        "Measurement": 1,
        "O'Neil:x%": 3,
    }
    ALL_INVALID: ClassVar[str] = "Measurement"

    @classmethod
    def _config(cls, **options: Any) -> HNSWIndexConfig:
        return HNSWIndexConfig(
            metric_type=MetricType.L2, num_neighbors=4, ef_construction=16, **options
        )

    @classmethod
    def _fill(cls, backend: PGVectorEmbeddingBackend) -> None:
        records = [
            ConceptEmbeddingRecord(
                concept_id=index,
                domain_id=domain,
                vocabulary_id="V",
                is_standard=True,
                is_valid=domain != cls.ALL_INVALID,
            )
            for index, domain in enumerate(
                (d for d, rows in cls.DOMAIN_ROWS.items() for _ in range(rows)), start=1
            )
        ]
        register_flat(backend)
        backend.upsert_embeddings(
            model_name=MODEL_NAME,
            records=records,
            embeddings=np.random.default_rng(0).random((len(records), EMBEDDING_DIM), np.float32),
        )

    @staticmethod
    def _partials(backend: PGVectorEmbeddingBackend) -> dict[str, str]:
        """Definition of every index with a WHERE clause, by name."""
        record = registered_model(backend, MODEL_NAME)
        with backend.emb_engine.connect() as conn:
            rows = conn.execute(
                sa.text(
                    "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = :t "
                    "AND indexdef LIKE '% WHERE %'"
                ),
                {"t": record.storage_identifier},
            ).all()
        return {name: definition for name, definition in rows}

    def test_no_partials_are_built_by_default(self, pg_backend: PGVectorEmbeddingBackend):
        self._fill(pg_backend)
        pg_backend.rebuild_index(model_name=MODEL_NAME, index_config=self._config())

        assert len(pg_backend.physical_indexes(MODEL_NAME)) == 1
        assert self._partials(pg_backend) == {}

    def test_every_domain_gets_a_partial_index(self, pg_backend: PGVectorEmbeddingBackend):
        self._fill(pg_backend)
        pg_backend.rebuild_index(
            model_name=MODEL_NAME, index_config=self._config(partial_domain_indexes=True)
        )

        partials = self._partials(pg_backend)
        assert sorted(name.split("_", 2)[2] for name in partials) == [
            "condition",
            "drug",
            "measurement",
            "o_neil_x",
        ]
        assert any("'O''Neil:x%'" in definition for definition in partials.values())
        assert len(pg_backend.physical_indexes(MODEL_NAME)) == 5

    def test_valid_only_leaves_out_a_domain_without_valid_rows(
        self, pg_backend: PGVectorEmbeddingBackend
    ):
        self._fill(pg_backend)
        pg_backend.rebuild_index(
            model_name=MODEL_NAME,
            index_config=self._config(partial_domain_indexes=True, partial_valid_only=True),
        )

        partials = self._partials(pg_backend)
        assert len(partials) == 3
        assert all("is_valid" in definition for definition in partials.values())
        assert not any(name.endswith("_measurement") for name in partials)

    def test_rebuilding_without_partials_drops_the_existing_ones(
        self, pg_backend: PGVectorEmbeddingBackend
    ):
        self._fill(pg_backend)
        pg_backend.rebuild_index(
            model_name=MODEL_NAME, index_config=self._config(partial_domain_indexes=True)
        )
        pg_backend.rebuild_index(model_name=MODEL_NAME, index_config=self._config())

        assert len(pg_backend.physical_indexes(MODEL_NAME)) == 1
        assert self._partials(pg_backend) == {}

    @pytest.mark.parametrize("domain", ["Condition", "O'Neil:x%"])
    def test_a_single_domain_query_returns_only_that_domain(
        self, pg_backend: PGVectorEmbeddingBackend, domain
    ):
        self._fill(pg_backend)
        pg_backend.rebuild_index(
            model_name=MODEL_NAME, index_config=self._config(partial_domain_indexes=True)
        )

        (matches,) = pg_backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.L2,
            query_embeddings=np.full((1, EMBEDDING_DIM), 0.5, dtype=np.float32),
            concept_filter=EmbeddingConceptFilter(domains=(domain,)),
            k=10,
        )

        assert len(matches) == self.DOMAIN_ROWS[domain]
        assert {match.domain_id for match in matches} == {domain}

    def test_a_rebuild_keeps_an_index_someone_added_by_hand(
        self, pg_backend: PGVectorEmbeddingBackend
    ):
        self._fill(pg_backend)
        record = registered_model(pg_backend, MODEL_NAME)
        table = qualified(
            pg_backend.emb_engine,
            record.storage_identifier,
            physical_schema=physical_schema_of(pg_backend.emb_engine),
        )
        with pg_backend.emb_engine.begin() as conn:
            conn.execute(sa.text(f"CREATE INDEX idx_domain ON {table} (domain_id)"))

        pg_backend.rebuild_index(
            model_name=MODEL_NAME, index_config=self._config(partial_domain_indexes=True)
        )

        with pg_backend.emb_engine.connect() as conn:
            names = conn.execute(
                sa.text("SELECT indexname FROM pg_indexes WHERE tablename = :t"),
                {"t": record.storage_identifier},
            ).scalars()
            assert "idx_domain" in set(names)
        assert "idx_domain" not in pg_backend.physical_indexes(MODEL_NAME)

    def test_rebuilding_twice_with_partials_succeeds(self, pg_backend: PGVectorEmbeddingBackend):
        self._fill(pg_backend)
        config = self._config(partial_domain_indexes=True)
        pg_backend.rebuild_index(model_name=MODEL_NAME, index_config=config)
        pg_backend.rebuild_index(model_name=MODEL_NAME, index_config=config)

        assert len(pg_backend.physical_indexes(MODEL_NAME)) == 5

    def test_the_settings_are_stored_in_the_registry(self, pg_backend: PGVectorEmbeddingBackend):
        self._fill(pg_backend)
        config = self._config(partial_domain_indexes=True, partial_valid_only=True)
        pg_backend.rebuild_index(model_name=MODEL_NAME, index_config=config)

        assert registered_model(pg_backend, MODEL_NAME).index_config == config

    def test_a_domain_that_cannot_be_named_fails_before_building_anything(
        self, pg_backend: PGVectorEmbeddingBackend
    ):
        register_flat(pg_backend)
        pg_backend.upsert_embeddings(
            model_name=MODEL_NAME,
            records=[
                ConceptEmbeddingRecord(
                    concept_id=1,
                    domain_id="***",
                    vocabulary_id="V",
                    is_standard=True,
                    is_valid=True,
                )
            ],
            embeddings=np.ones((1, EMBEDDING_DIM), dtype=np.float32),
        )

        with pytest.raises(ValueError, match="partial index"):
            pg_backend.rebuild_index(
                model_name=MODEL_NAME, index_config=self._config(partial_domain_indexes=True)
            )
        assert pg_backend.physical_indexes(MODEL_NAME) == ()

    def test_deleting_the_model_drops_its_partials(self, pg_backend: PGVectorEmbeddingBackend):
        self._fill(pg_backend)
        pg_backend.rebuild_index(
            model_name=MODEL_NAME, index_config=self._config(partial_domain_indexes=True)
        )
        pg_backend.delete_model(model_name=MODEL_NAME)

        register_flat(pg_backend)
        assert pg_backend.physical_indexes(MODEL_NAME) == ()

    def test_partials_follow_a_half_index_over_a_float32_column(
        self, pg_backend: PGVectorEmbeddingBackend
    ):
        self._fill(pg_backend)
        pg_backend.rebuild_index(
            model_name=MODEL_NAME,
            index_config=self._config(
                partial_domain_indexes=True, index_precision=VectorPrecision.FLOAT16
            ),
        )

        record = registered_model(pg_backend, MODEL_NAME)
        with pg_backend.emb_engine.connect() as conn:
            definitions = conn.execute(
                sa.text("SELECT indexdef FROM pg_indexes WHERE tablename = :t"),
                {"t": record.storage_identifier},
            ).scalars()
            hnsw = [d for d in definitions if "USING hnsw" in d]
        assert len(hnsw) == 5
        assert all("halfvec_l2_ops" in d for d in hnsw)


@pytest.mark.pgvector
class TestPGVectorPartialIndexRouting:
    """The index the planner chooses for each kind of domain filter.

    The statement is prepared with a generic plan, as the driver does for a
    statement it runs repeatedly. Such a plan cannot use a partial index
    through a bound parameter.
    """

    DIMENSIONS = 16
    DOMAIN_ROWS: ClassVar[dict[str, int]] = {"Condition": 12000, "Drug": 6000, "Device": 2000}

    @staticmethod
    def _literal(value: Any) -> str:
        """SQL literal for a bound parameter: a number, a string or a list of strings."""
        if isinstance(value, list):
            return "'{" + ",".join(f'"{item}"' for item in value) + "}'"
        if isinstance(value, str):
            return "'" + value.replace("'", "''") + "'"
        return str(value)

    @pytest.fixture
    def routing(self, pg_backend: PGVectorEmbeddingBackend):
        """Build the indexes, with valid-only partials or not, and return the table and ``choose``.

        ``choose(concept_filter)`` is the name of the index the planner uses for
        a query with that filter.
        """

        def build(valid_only: bool = False):
            return self._build(pg_backend, valid_only)

        return build

    def _build(self, pg_backend: PGVectorEmbeddingBackend, valid_only: bool):
        register_flat(pg_backend, dimensions=self.DIMENSIONS)
        rng = np.random.default_rng(0)
        records = [
            ConceptEmbeddingRecord(
                concept_id=index,
                domain_id=domain,
                vocabulary_id="V",
                is_standard=True,
                is_valid=index % 10 != 0,
            )
            for index, domain in enumerate(
                (d for d, rows in self.DOMAIN_ROWS.items() for _ in range(rows)), start=1
            )
        ]
        pg_backend.upsert_embeddings(
            model_name=MODEL_NAME,
            records=records,
            embeddings=rng.random((len(records), self.DIMENSIONS), dtype=np.float32),
        )
        pg_backend.rebuild_index(
            model_name=MODEL_NAME,
            index_config=HNSWIndexConfig(
                metric_type=MetricType.COSINE,
                num_neighbors=8,
                ef_construction=32,
                partial_domain_indexes=True,
                partial_valid_only=valid_only,
            ),
        )
        record = registered_model(pg_backend, MODEL_NAME)

        def choose(concept_filter: EmbeddingConceptFilter | None) -> str:
            captured: list[tuple[str, Any]] = []

            def capture(_connection, _cursor, statement, parameters, _context, _many):
                if "order by" in statement.lower():
                    captured.append((statement, parameters))

            sa_event.listen(pg_backend.emb_engine, "before_cursor_execute", capture)
            try:
                pg_backend.get_nearest_concepts(
                    model_name=MODEL_NAME,
                    metric_type=MetricType.COSINE,
                    query_embeddings=np.full((1, self.DIMENSIONS), 0.5, dtype=np.float32),
                    concept_filter=concept_filter,
                    k=10,
                )
            finally:
                sa_event.remove(pg_backend.emb_engine, "before_cursor_execute", capture)
            statement, parameters = captured[0]
            names = list(dict.fromkeys(re.findall(r"%\((\w+)\)s", statement)))
            prepared = re.sub(
                r"%\((\w+)\)s", lambda m: f"${names.index(m.group(1)) + 1}", statement
            )
            call = ", ".join(self._literal(parameters[name]) for name in names)
            with pg_backend.emb_engine.connect() as conn:
                conn.exec_driver_sql("SET plan_cache_mode = force_generic_plan")
                conn.exec_driver_sql(f"PREPARE generic_query AS {prepared}")
                plan = "\n".join(
                    row[0] for row in conn.exec_driver_sql(f"EXPLAIN EXECUTE generic_query({call})")
                )
            match = re.search(r"using (idx_\w+)", plan)
            assert match, plan
            return match.group(1)

        return choose, record.storage_identifier

    @pytest.mark.parametrize("domain", ["Condition", "Drug", "Device"])
    def test_a_single_domain_uses_that_domains_partial_index(self, routing, domain):
        choose, table = routing()
        assert choose(EmbeddingConceptFilter(domains=(domain,))) == IndexNames.partial(
            table, domain
        )

    def test_several_domains_use_the_global_index(self, routing):
        choose, table = routing()
        assert choose(EmbeddingConceptFilter(domains=("Drug", "Device"))) == IndexNames.hnsw(
            table, MetricType.COSINE
        )

    def test_no_domain_uses_the_global_index(self, routing):
        choose, table = routing()
        assert choose(None) == IndexNames.hnsw(table, MetricType.COSINE)

    def test_valid_only_partials_serve_a_single_domain_that_requires_active(self, routing):
        choose, table = routing(valid_only=True)
        concept_filter = EmbeddingConceptFilter(domains=("Drug",), require_active=True)

        assert choose(concept_filter) == IndexNames.partial(table, "Drug")

    def test_valid_only_partials_are_skipped_when_invalid_concepts_are_allowed(self, routing):
        choose, table = routing(valid_only=True)

        assert choose(EmbeddingConceptFilter(domains=("Drug",))) == IndexNames.hnsw(
            table, MetricType.COSINE
        )

    def test_domain_only_partials_serve_a_query_that_requires_active(self, routing):
        choose, table = routing()
        concept_filter = EmbeddingConceptFilter(domains=("Drug",), require_active=True)

        assert choose(concept_filter) == IndexNames.partial(table, "Drug")
