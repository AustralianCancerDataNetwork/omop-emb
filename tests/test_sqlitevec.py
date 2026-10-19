"""Tests for the SQLiteVec embedding backend.

In-memory SQLite: no external service required.
"""

from __future__ import annotations

import numpy as np
import pytest
import sqlalchemy as sa

from omop_emb.backends.base_backend import ConceptEmbeddingRecord
from omop_emb.backends.index_config import FlatIndexConfig
from omop_emb.backends.sqlitevec import SQLiteVecEmbeddingBackend
from omop_emb.backends.sqlitevec.sqlitevec_backend import _load_sqlite_vec
from omop_emb.backends.sqlitevec.sqlitevec_sql import (
    MAX_KNN_K,
    ddl_drop_vec0,
    declared_metric,
    recreate_vec0,
)
from omop_emb.config import MetricType
from omop_emb.model_registry import RegistryManager, registry_writer_engine
from omop_emb.model_registry.model_registry_orm import ModelRegistry
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter
from omop_emb.utils.errors import LegacyVectorTableError

from .conftest import (
    ASPIRIN_ID,
    DIABETES_ID,
    L2_SIMILARITY,
    CONCEPT_EMBEDDINGS,
    CONCEPT_RECORDS,
    EMBEDDING_DIM,
    MODEL_NAME,
    PROVIDER_TYPE,
    QUERY_EMBEDDING,
    register_flat,
    sqlite_resolved_database,
)
from .shared_backend_tests import SharedBackendTests


@pytest.mark.unit
class TestSQLiteVecEmbeddingBackend(SharedBackendTests):
    """Runs the full shared suite against an in-memory SQLiteVecEmbeddingBackend."""

    @pytest.fixture
    def backend(self, svec_backend: SQLiteVecEmbeddingBackend):
        return svec_backend


@pytest.mark.unit
class TestSQLiteVecSpecific:
    """SQLiteVec-specific behaviour not covered by the shared suite."""

    def test_existing_old_style_storage_identifier_keeps_working(
        self, svec_backend: SQLiteVecEmbeddingBackend, monkeypatch
    ):
        model_name = "old-style-model"
        storage_identifier = "emb_old_style_model"
        vector = CONCEPT_EMBEDDINGS[:1]
        with svec_backend._registry.emb_session_factory.begin() as session:
            session.add(
                ModelRegistry(
                    database_config_name=svec_backend._registry._database_config_name,
                    model_name=model_name,
                    provider_type=PROVIDER_TYPE,
                    storage_identifier=storage_identifier,
                    dimensions=EMBEDDING_DIM,
                    index_config=FlatIndexConfig(metric_type=MetricType.COSINE),
                    details={},
                )
            )

        record = svec_backend.get_registered_model(model_name=model_name)
        assert record is not None
        assert record.storage_identifier == storage_identifier
        svec_backend._ensure_storage_table(record)
        monkeypatch.setattr(
            RegistryManager,
            "storage_name",
            staticmethod(lambda *_args: pytest.fail("existing row identifier was recomputed")),
        )
        same_record = svec_backend.register_model(
            model_name=model_name,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(),
            dimensions=EMBEDDING_DIM,
        )
        assert same_record.storage_identifier == storage_identifier
        svec_backend.upsert_embeddings(
            model_name=model_name, records=CONCEPT_RECORDS[:1], embeddings=vector
        )
        found = svec_backend.get_embeddings_by_concept_ids(
            model_name=model_name, concept_ids=[CONCEPT_RECORDS[0].concept_id]
        )
        assert np.allclose(found[CONCEPT_RECORDS[0].concept_id], vector[0])

        svec_backend.delete_model(model_name=model_name)
        assert svec_backend.get_registered_model(model_name=model_name) is None
        assert not sa.inspect(svec_backend.emb_engine).has_table(storage_identifier)

    def test_file_backed_engine_constructor(self, tmp_path):
        """A real file-backed (not just in-memory) engine works end to end."""
        db_file = str(tmp_path / "test.db")
        engine = registry_writer_engine(
            sqlite_resolved_database(db_file), extensions=[_load_sqlite_vec]
        )
        backend = SQLiteVecEmbeddingBackend(emb_engine=engine)
        assert register_flat(backend).model_name == MODEL_NAME
        backend.emb_engine.dispose()


def _make_pre_cosine_table(backend: SQLiteVecEmbeddingBackend, record, vectors: np.ndarray) -> None:
    """Replace the storage table with one created without a declared metric."""
    table = record.storage_identifier
    with backend.emb_engine.begin() as conn:
        conn.execute(sa.text(ddl_drop_vec0(table)))
        conn.execute(
            sa.text(
                f'CREATE VIRTUAL TABLE "{table}" USING vec0(concept_id INTEGER PRIMARY KEY, '
                "domain_id TEXT METADATA, vocabulary_id TEXT METADATA, "
                "is_standard BOOLEAN METADATA, is_valid BOOLEAN METADATA DEFAULT 1, "
                f"embedding FLOAT[{vectors.shape[1]}])"
            )
        )
        for i, vector in enumerate(vectors, start=1):
            conn.execute(
                sa.text(
                    f'INSERT INTO "{table}"(concept_id, domain_id, vocabulary_id, is_standard, is_valid, embedding) '
                    "VALUES (:i, 'Condition', 'SNOMED', 1, 1, :v)"
                ),
                {"i": i, "v": vector.astype(np.float32).tobytes()},
            )
    backend._table_cache.clear()


@pytest.mark.unit
class TestSQLiteVecMetricTable:
    """The vec0 table declares the metric the model is registered with, which MATCH queries rely on."""

    def test_pre_cosine_table_raises_with_rebuild_instruction(
        self, svec_backend: SQLiteVecEmbeddingBackend
    ):
        record = register_flat(svec_backend)
        _make_pre_cosine_table(svec_backend, record, CONCEPT_EMBEDDINGS)
        with pytest.raises(LegacyVectorTableError, match="rebuild-index"):
            svec_backend.get_nearest_concepts(
                model_name=MODEL_NAME,
                metric_type=MetricType.COSINE,
                query_embeddings=QUERY_EMBEDDING,
                k=1,
            )

    def test_rebuild_index_recreates_pre_cosine_table_keeping_rows(
        self, svec_backend: SQLiteVecEmbeddingBackend
    ):
        record = register_flat(svec_backend)
        _make_pre_cosine_table(svec_backend, record, CONCEPT_EMBEDDINGS)
        concept_ids = [r.concept_id for r in CONCEPT_RECORDS]

        svec_backend.rebuild_index(model_name=MODEL_NAME, index_config=FlatIndexConfig())

        assert svec_backend.get_embedding_count(model_name=MODEL_NAME) == len(CONCEPT_RECORDS)
        stored = svec_backend.get_embeddings_by_concept_ids(model_name=MODEL_NAME, concept_ids=concept_ids)
        assert np.allclose([stored[i] for i in concept_ids], CONCEPT_EMBEDDINGS)
        results = svec_backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.COSINE,
            query_embeddings=CONCEPT_EMBEDDINGS[[ASPIRIN_ID - 1]],
            k=1,
        )
        assert results[0][0].concept_id == ASPIRIN_ID
        assert results[0][0].similarity == pytest.approx(1.0, abs=1e-6)

    def test_rebuild_index_leaves_cosine_table_untouched(
        self, svec_backend: SQLiteVecEmbeddingBackend
    ):
        """An already-cosine table keeps its rows and is neither recreated nor vacuumed."""
        register_flat(svec_backend)
        svec_backend.upsert_embeddings(
            model_name=MODEL_NAME, records=CONCEPT_RECORDS, embeddings=CONCEPT_EMBEDDINGS
        )
        statements: list[str] = []

        def capture(_conn, _cursor, statement, *_args):
            statements.append(statement.strip().upper())

        sa.event.listen(svec_backend.emb_engine, "before_cursor_execute", capture)
        try:
            svec_backend.rebuild_index(model_name=MODEL_NAME, index_config=FlatIndexConfig())
        finally:
            sa.event.remove(svec_backend.emb_engine, "before_cursor_execute", capture)

        assert svec_backend.get_embedding_count(model_name=MODEL_NAME) == len(CONCEPT_RECORDS)
        assert not any(s.startswith(("VACUUM", "DROP", "CREATE")) for s in statements)

    @pytest.mark.parametrize("fail_at", range(1, 9))
    def test_failed_recreate_leaves_original_table_intact(
        self, svec_backend: SQLiteVecEmbeddingBackend, fail_at: int
    ):
        """A failure at any statement rolls everything back: original rows, no staging table."""
        record = register_flat(svec_backend)
        _make_pre_cosine_table(svec_backend, record, CONCEPT_EMBEDDINGS)
        table = record.storage_identifier
        seen = []

        def fail_on_nth_statement(conn, cursor, statement, parameters, context, executemany):
            if statement.strip().upper().startswith("BEGIN"):
                return
            seen.append(statement)
            if len(seen) == fail_at:
                raise RuntimeError("injected failure")

        sa.event.listen(svec_backend.emb_engine, "before_cursor_execute", fail_on_nth_statement)
        try:
            with pytest.raises(RuntimeError, match="injected failure"):
                recreate_vec0(svec_backend.emb_engine, table, EMBEDDING_DIM, MetricType.COSINE)
        finally:
            sa.event.remove(svec_backend.emb_engine, "before_cursor_execute", fail_on_nth_statement)

        assert declared_metric(svec_backend.emb_engine, table) is None
        assert not sa.inspect(svec_backend.emb_engine).has_table(f"{table}__rebuild")
        with svec_backend.emb_engine.connect() as conn:
            assert conn.execute(sa.text(f'SELECT count(*) FROM "{table}"')).scalar_one() == len(CONCEPT_RECORDS)
        assert recreate_vec0(svec_backend.emb_engine, table, EMBEDDING_DIM, MetricType.COSINE) == len(CONCEPT_RECORDS)
        assert declared_metric(svec_backend.emb_engine, table) == MetricType.COSINE

    def test_registration_locks_cosine_by_default(self, svec_backend: SQLiteVecEmbeddingBackend):
        record = register_flat(svec_backend)
        assert record.metric_type == MetricType.COSINE
        assert declared_metric(svec_backend.emb_engine, record.storage_identifier) == MetricType.COSINE

    @pytest.mark.parametrize("metric", [MetricType.L2, MetricType.L1])
    def test_registration_with_another_metric_locks_the_model_to_it(
        self, svec_backend: SQLiteVecEmbeddingBackend, metric: MetricType
    ):
        record = svec_backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(metric_type=metric),
            dimensions=EMBEDDING_DIM,
        )
        assert record.metric_type == metric
        assert declared_metric(svec_backend.emb_engine, record.storage_identifier) == metric
        svec_backend.upsert_embeddings(
            model_name=MODEL_NAME, records=CONCEPT_RECORDS, embeddings=CONCEPT_EMBEDDINGS
        )
        (matches,) = svec_backend.get_nearest_concepts(
            model_name=MODEL_NAME, metric_type=metric, query_embeddings=QUERY_EMBEDDING, k=4
        )
        assert matches[0].concept_id == DIABETES_ID
        similarities = [m.similarity for m in matches]
        assert similarities == sorted(similarities, reverse=True)
        other = MetricType.COSINE
        with pytest.raises(ValueError, match=f"indexed with metric '{metric.value}'"):
            svec_backend.get_nearest_concepts(
                model_name=MODEL_NAME, metric_type=other, query_embeddings=QUERY_EMBEDDING, k=1
            )

    def test_l2_similarity_values(self, svec_backend: SQLiteVecEmbeddingBackend):
        svec_backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(metric_type=MetricType.L2),
            dimensions=EMBEDDING_DIM,
        )
        svec_backend.upsert_embeddings(
            model_name=MODEL_NAME, records=CONCEPT_RECORDS, embeddings=CONCEPT_EMBEDDINGS
        )
        (matches,) = svec_backend.get_nearest_concepts(
            model_name=MODEL_NAME, metric_type=MetricType.L2, query_embeddings=QUERY_EMBEDDING, k=4
        )
        for match in matches:
            assert match.similarity == pytest.approx(L2_SIMILARITY[match.concept_id], abs=1e-5)

    def test_registering_an_unsupported_metric_raises(self, svec_backend: SQLiteVecEmbeddingBackend):
        with pytest.raises(ValueError, match="not supported"):
            svec_backend.register_model(
                model_name=MODEL_NAME,
                provider_type=PROVIDER_TYPE,
                index_config=FlatIndexConfig(metric_type=MetricType.HAMMING),
                dimensions=EMBEDDING_DIM,
            )

    def test_rebuild_index_switches_the_metric_keeping_rows(
        self, svec_backend: SQLiteVecEmbeddingBackend
    ):
        register_flat(svec_backend)
        svec_backend.upsert_embeddings(
            model_name=MODEL_NAME, records=CONCEPT_RECORDS, embeddings=CONCEPT_EMBEDDINGS
        )

        record = svec_backend.rebuild_index(
            model_name=MODEL_NAME, index_config=FlatIndexConfig(metric_type=MetricType.L2)
        )

        assert record.metric_type == MetricType.L2
        assert declared_metric(svec_backend.emb_engine, record.storage_identifier) == MetricType.L2
        assert svec_backend.get_embedding_count(model_name=MODEL_NAME) == len(CONCEPT_RECORDS)
        (matches,) = svec_backend.get_nearest_concepts(
            model_name=MODEL_NAME, metric_type=MetricType.L2, query_embeddings=QUERY_EMBEDDING, k=1
        )
        assert matches[0].concept_id == DIABETES_ID
        with pytest.raises(ValueError, match="indexed with metric 'l2'"):
            svec_backend.get_nearest_concepts(
                model_name=MODEL_NAME,
                metric_type=MetricType.COSINE,
                query_embeddings=QUERY_EMBEDDING,
                k=1,
            )

    def test_rebuild_index_without_a_metric_keeps_the_current_one(
        self, svec_backend: SQLiteVecEmbeddingBackend
    ):
        svec_backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(metric_type=MetricType.L1),
            dimensions=EMBEDDING_DIM,
        )
        record = svec_backend.rebuild_index(model_name=MODEL_NAME, index_config=FlatIndexConfig())
        assert record.metric_type == MetricType.L1
        assert declared_metric(svec_backend.emb_engine, record.storage_identifier) == MetricType.L1

    def test_registry_row_without_a_metric_raises_then_rebuilds_to_cosine(
        self, svec_backend: SQLiteVecEmbeddingBackend
    ):
        """A row registered before models carried a metric points at a table that declares cosine."""
        register_flat(svec_backend)
        svec_backend.upsert_embeddings(
            model_name=MODEL_NAME, records=CONCEPT_RECORDS, embeddings=CONCEPT_EMBEDDINGS
        )
        svec_backend._registry.update_index_config(
            model_name=MODEL_NAME, index_config=FlatIndexConfig()
        )
        svec_backend._table_cache.clear()

        with pytest.raises(LegacyVectorTableError, match="rebuild-index"):
            svec_backend.get_embedding_count(model_name=MODEL_NAME)

        record = svec_backend.rebuild_index(model_name=MODEL_NAME, index_config=FlatIndexConfig())
        assert record.metric_type == MetricType.COSINE
        assert svec_backend.get_embedding_count(model_name=MODEL_NAME) == len(CONCEPT_RECORDS)

    def test_k_above_match_limit_raises(self, svec_backend: SQLiteVecEmbeddingBackend):
        register_flat(svec_backend)
        with pytest.raises(ValueError, match=str(MAX_KNN_K)):
            svec_backend.get_nearest_concepts(
                model_name=MODEL_NAME,
                metric_type=MetricType.COSINE,
                query_embeddings=QUERY_EMBEDDING,
                k=MAX_KNN_K + 1,
            )

    def test_selective_filter_still_returns_k_rows(self, svec_backend: SQLiteVecEmbeddingBackend):
        """The filter applies during the scan, so k matching rows come back even
        when matching rows are a small share of the table."""
        register_flat(svec_backend)
        rng = np.random.default_rng(0)
        n = 400
        records = [
            ConceptEmbeddingRecord(
                concept_id=i,
                domain_id="Condition" if i % 20 == 0 else "Drug",
                vocabulary_id="SNOMED",
                is_standard=True,
                is_valid=True,
            )
            for i in range(1, n + 1)
        ]
        svec_backend.upsert_embeddings(
            model_name=MODEL_NAME,
            records=records,
            embeddings=rng.normal(size=(n, EMBEDDING_DIM)).astype(np.float32),
        )
        results = svec_backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.COSINE,
            query_embeddings=QUERY_EMBEDDING,
            k=10,
            concept_filter=EmbeddingConceptFilter(domains=("Condition",)),
        )
        assert len(results[0]) == 10
        assert {m.domain_id for m in results[0]} == {"Condition"}
        similarities = [m.similarity for m in results[0]]
        assert similarities == sorted(similarities, reverse=True)


@pytest.mark.unit
class TestSQLiteVecVacuum:
    """rebuild_index returns the space freed by a table recreation unless told not to."""

    @staticmethod
    def _file_backend(tmp_path) -> SQLiteVecEmbeddingBackend:
        engine = registry_writer_engine(
            sqlite_resolved_database(str(tmp_path / "vacuum.db")), extensions=[_load_sqlite_vec]
        )
        return SQLiteVecEmbeddingBackend(emb_engine=engine)

    @staticmethod
    def _freelist(backend: SQLiteVecEmbeddingBackend) -> int:
        with backend.emb_engine.connect() as conn:
            return int(conn.exec_driver_sql("PRAGMA freelist_count").scalar_one())

    def _rebuild_pre_cosine(self, tmp_path, **kwargs) -> SQLiteVecEmbeddingBackend:
        backend = self._file_backend(tmp_path)
        record = register_flat(backend, dimensions=64)
        _make_pre_cosine_table(backend, record, np.random.default_rng(0).normal(size=(600, 64)))
        backend.rebuild_index(model_name=MODEL_NAME, index_config=FlatIndexConfig(), **kwargs)
        return backend

    def test_vacuum_is_the_default_and_returns_freed_pages(self, tmp_path):
        backend = self._rebuild_pre_cosine(tmp_path)
        try:
            assert self._freelist(backend) == 0
            assert backend.get_embedding_count(model_name=MODEL_NAME) == 600
        finally:
            backend.emb_engine.dispose()

    def test_vacuum_false_keeps_the_freed_pages(self, tmp_path):
        backend = self._rebuild_pre_cosine(tmp_path, vacuum=False)
        try:
            assert self._freelist(backend) > 0
            assert backend.get_embedding_count(model_name=MODEL_NAME) == 600
        finally:
            backend.emb_engine.dispose()
