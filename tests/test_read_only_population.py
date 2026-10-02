from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import event, insert, inspect, text

from omop_alchemy.cdm.model.vocabulary import Concept
from omop_emb.backends.base_backend import StoredEmbedding
from omop_emb.backends.index_config import FlatIndexConfig
from omop_emb.backends.sqlitevec import SQLiteVecEmbeddingBackend
from omop_emb.backends.sqlitevec.sqlitevec_backend import _load_sqlite_vec
from omop_emb.config import MetricType
from omop_emb.model_registry import (
    ModelRegistry,
    bootstrap_registry_engine,
    peek_registry_engine,
)
from omop_emb.population import PopulationScope, plan_population
from omop_emb.utils.errors import MissingStorageTableError

from .conftest import sqlite_resolved_database


def _concept(concept_id: int, **overrides):
    values = {
        "concept_id": concept_id,
        "concept_name": f"Concept {concept_id}",
        "domain_id": "Condition",
        "vocabulary_id": "SNOMED",
        "concept_class_id": "Clinical Finding",
        "standard_concept": "S",
        "concept_code": str(concept_id),
        "valid_start_date": date(2020, 1, 1),
        "valid_end_date": date(2099, 12, 31),
        "invalid_reason": None,
    }
    values.update(overrides)
    return values


def test_peek_before_any_write_does_not_create_the_registry(tmp_path) -> None:
    """peek_registry_engine() maps the registry schema tag but never runs
    ensure_registry_table(), so a store that's never been bootstrapped is
    reported as such instead of being silently set up just by being looked at."""
    resolved = sqlite_resolved_database(str(tmp_path / "test.db"))
    engine = peek_registry_engine(resolved, extensions=[_load_sqlite_vec])
    backend = SQLiteVecEmbeddingBackend(emb_engine=engine)

    assert backend.initialized is False
    assert backend.registered_models() == ()
    assert inspect(engine).has_table(ModelRegistry.__tablename__) is False
    backend.close()


def test_peek_after_bootstrap_sees_the_registry(tmp_path) -> None:
    """A peek-constructed backend against the same physical file a writable
    backend already bootstrapped sees the real, now-existing registry."""
    db_path = str(tmp_path / "test.db")
    bootstrap_registry_engine(
        sqlite_resolved_database(db_path), extensions=[_load_sqlite_vec]
    ).dispose()

    engine = peek_registry_engine(
        sqlite_resolved_database(db_path), extensions=[_load_sqlite_vec]
    )
    backend = SQLiteVecEmbeddingBackend(emb_engine=engine)

    assert backend.initialized is True
    assert backend.registered_models() == ()
    backend.close()


def test_peek_sees_registered_models_and_stored_embeddings_without_mutating(tmp_path) -> None:
    """A model registered and populated through the writable path is fully
    visible through a peek-only backend against the same file, and reading it
    issues no mutating SQL."""
    db_path = str(tmp_path / "test.db")
    write_engine = bootstrap_registry_engine(
        sqlite_resolved_database(db_path), extensions=[_load_sqlite_vec]
    )
    write_backend = SQLiteVecEmbeddingBackend(emb_engine=write_engine)
    write_backend.register_model(
        model_name="test-model",
        provider_type="ollama",
        index_config=FlatIndexConfig(),
        dimensions=1,
    )
    write_backend.close()

    engine = peek_registry_engine(
        sqlite_resolved_database(db_path), extensions=[_load_sqlite_vec]
    )

    statements: list[str] = []

    def capture(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.strip().lower())

    event.listen(engine, "before_cursor_execute", capture)
    try:
        with SQLiteVecEmbeddingBackend(emb_engine=engine) as backend:
            assert backend.initialized is True
            assert [m.model_name for m in backend.registered_models()] == ["test-model"]
            assert backend.stored_embeddings("test-model") == ()
    finally:
        event.remove(engine, "before_cursor_execute", capture)

    assert statements
    assert not any(
        statement.startswith(
            ("create ", "alter ", "drop ", "insert ", "update ", "delete ")
        )
        for statement in statements
    )


def test_dropped_storage_table_raises_instead_of_being_recreated(tmp_path) -> None:
    """A model registered but whose physical table was dropped out from under
    it raises MissingStorageTableError for a fresh backend instance (an empty
    in-process table-descriptor cache, like a new peek), instead of the old
    behavior of silently recreating an empty table."""
    db_path = str(tmp_path / "test.db")
    write_engine = bootstrap_registry_engine(
        sqlite_resolved_database(db_path), extensions=[_load_sqlite_vec]
    )
    write_backend = SQLiteVecEmbeddingBackend(emb_engine=write_engine)
    record = write_backend.register_model(
        model_name="test-model",
        provider_type="ollama",
        index_config=FlatIndexConfig(),
        dimensions=1,
    )
    with write_engine.begin() as connection:
        connection.execute(text(f"DROP TABLE {record.storage_identifier}"))
    write_backend.close()

    engine = peek_registry_engine(
        sqlite_resolved_database(db_path), extensions=[_load_sqlite_vec]
    )
    backend = SQLiteVecEmbeddingBackend(emb_engine=engine)
    with pytest.raises(MissingStorageTableError):
        backend.has_any_embeddings(model_name="test-model", metric_type=MetricType.L2)
    backend.close()


def test_population_plan_distinguishes_missing_and_stale_ids() -> None:
    engine = sqlite_resolved_database().create_engine()
    Concept.__table__.create(engine)
    with engine.begin() as connection:
        connection.execute(
            insert(Concept),
            [
                _concept(1),
                _concept(2),
            ],
        )

    class FakeStore:
        initialized = True

        def stored_embeddings(self, _model_name: str):
            return (
                StoredEmbedding(1, "Condition", "SNOMED", True, True),
                StoredEmbedding(3, "Condition", "SNOMED", True, True),
            )

    plan = plan_population(
        engine,
        FakeStore(),
        model_name="test-model",
        scope=PopulationScope(standard_only=True),
    )
    row = plan.rows[0]

    assert row.eligible_ids == frozenset({1, 2})
    assert row.compatible_ids == frozenset({1})
    assert row.missing_ids == frozenset({2})
    assert row.stale_ids == frozenset({3})
    assert plan.pending_ids == frozenset({2})


def test_population_scope_uses_omop_alchemy_standard_and_valid_flags() -> None:
    """Only concept 1 is both standard (``'S'``) and valid (no invalid_reason,
    confirmed directly against omop_alchemy's own ``ConceptFilter``): concept 2
    is a classification concept (``'C'``, not standard) despite its blank
    invalid_reason normalizing to valid; concept 3 has no standard flag at
    all; concept 4 is standard but marked deleted."""
    engine = sqlite_resolved_database().create_engine()
    Concept.__table__.create(engine)
    with engine.begin() as connection:
        connection.execute(
            insert(Concept),
            [
                _concept(1, standard_concept="S", invalid_reason=None),
                _concept(2, standard_concept="C", invalid_reason=" "),
                _concept(3, standard_concept=None, invalid_reason=None),
                _concept(4, standard_concept="S", invalid_reason="D"),
            ],
        )

    class EmptyStore:
        initialized = True

        def iter_stored_embeddings(self, _model_name: str, *, batch_size: int):
            assert batch_size == 2
            return iter(())

    plan = plan_population(
        engine,
        EmptyStore(),
        model_name="test-model",
        scope=PopulationScope(standard_only=True, valid_only=True),
        batch_size=2,
    )

    assert plan.eligible_ids == frozenset({1})
    assert plan.missing_ids == frozenset({1})


def test_filtered_population_does_not_mark_out_of_scope_rows_stale() -> None:
    engine = sqlite_resolved_database().create_engine()
    Concept.__table__.create(engine)
    with engine.begin() as connection:
        connection.execute(
            insert(Concept),
            [
                _concept(1),
                _concept(2, domain_id="Drug", vocabulary_id="RxNorm"),
            ],
        )

    class Store:
        initialized = True

        def stored_embeddings(self, _model_name: str):
            return (
                StoredEmbedding(1, "Condition", "SNOMED", True, True),
                StoredEmbedding(2, "Drug", "RxNorm", True, True),
            )

    plan = plan_population(
        engine,
        Store(),
        model_name="test-model",
        scope=PopulationScope(vocabularies=("SNOMED",)),
    )

    assert tuple(row.vocabulary for row in plan.rows) == ("SNOMED",)
    assert plan.compatible_ids == frozenset({1})
    assert plan.stale_ids == frozenset()


def test_metadata_change_is_pending() -> None:
    engine = sqlite_resolved_database().create_engine()
    Concept.__table__.create(engine)
    with engine.begin() as connection:
        connection.execute(insert(Concept), [_concept(1)])

    class Store:
        initialized = True

        def stored_embeddings(self, _model_name: str):
            return (StoredEmbedding(1, "Measurement", "SNOMED", True, True),)

    plan = plan_population(engine, Store(), model_name="test-model")

    assert plan.metadata_changed_ids == frozenset({1})
    assert plan.pending_ids == frozenset({1})
