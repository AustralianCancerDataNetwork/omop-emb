from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import event, insert, inspect, text

from omop_alchemy.cdm.model.vocabulary import Concept
from omop_emb.backends import open_vector_store_reader, open_vector_store_writer
from omop_emb.backends.embedding_table import ConceptEmbeddingRecord
from omop_emb.backends.index_config import FlatIndexConfig
from omop_emb.model_registry import ModelRegistry
from omop_emb.population import PopulationScope, plan_population
from omop_emb.utils.errors import MissingStorageTableError

from .conftest import sqlite_resolved_database, sqlite_resolved_vector_store


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


def test_reader_before_any_write_does_not_create_the_registry(tmp_path) -> None:
    """A store that's never been set up reports an empty registry instead of
    being set up by being looked at."""
    with open_vector_store_reader(sqlite_resolved_vector_store(str(tmp_path / "test.db"))) as reader:
        assert reader.initialized is False
        assert reader.get_registered_models() == ()
        assert inspect(reader.emb_engine).has_table(ModelRegistry.__tablename__) is False  # ty: ignore[unresolved-attribute]


def test_reader_after_writer_sees_the_registry(tmp_path) -> None:
    store = sqlite_resolved_vector_store(str(tmp_path / "test.db"))
    open_vector_store_writer(store).close()

    with open_vector_store_reader(store) as reader:
        assert reader.initialized is True
        assert reader.get_registered_models() == ()


def test_reader_sees_registered_models_and_stored_embeddings_without_mutating(tmp_path) -> None:
    """A model registered through the writer is visible through a reader on
    the same file, and reading it issues no mutating SQL."""
    store = sqlite_resolved_vector_store(str(tmp_path / "test.db"))
    with open_vector_store_writer(store) as writer:
        writer.register_model(
            model_name="test-model",
            provider_type="ollama",
            index_config=FlatIndexConfig(),
            dimensions=1,
        )

    statements: list[str] = []

    def capture(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.strip().lower())

    with open_vector_store_reader(store) as reader:
        engine = reader.emb_engine  # ty: ignore[unresolved-attribute]
        event.listen(engine, "before_cursor_execute", capture)
        try:
            assert reader.initialized is True
            assert [m.model_name for m in reader.get_registered_models()] == ["test-model"]
            assert tuple(reader.iter_stored_embeddings("test-model")) == ()
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
    in-process table-descriptor cache), instead of silently recreating an
    empty table."""
    store = sqlite_resolved_vector_store(str(tmp_path / "test.db"))
    with open_vector_store_writer(store) as writer:
        record = writer.register_model(
            model_name="test-model",
            provider_type="ollama",
            index_config=FlatIndexConfig(),
            dimensions=1,
        )
        with writer.emb_engine.begin() as connection:
            connection.execute(text(f"DROP TABLE {record.storage_identifier}"))

    with open_vector_store_reader(store) as reader:
        with pytest.raises(MissingStorageTableError):
            reader.has_any_embeddings(model_name="test-model")


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

        def iter_stored_embeddings(self, _model_name: str, *, batch_size: int = 10_000):
            return (
                ConceptEmbeddingRecord(1, "Condition", "SNOMED", True, True),
                ConceptEmbeddingRecord(3, "Condition", "SNOMED", True, True),
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

        def iter_stored_embeddings(self, _model_name: str, *, batch_size: int = 10_000):
            return (
                ConceptEmbeddingRecord(1, "Condition", "SNOMED", True, True),
                ConceptEmbeddingRecord(2, "Drug", "RxNorm", True, True),
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

        def iter_stored_embeddings(self, _model_name: str, *, batch_size: int = 10_000):
            return (ConceptEmbeddingRecord(1, "Measurement", "SNOMED", True, True),)

    plan = plan_population(engine, Store(), model_name="test-model")

    assert plan.metadata_changed_ids == frozenset({1})
    assert plan.pending_ids == frozenset({1})
