from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine, event, insert, inspect

from omop_alchemy.cdm.model.vocabulary import Concept
from omop_emb.backends import ReadOnlyEmbeddingStore, StoredEmbedding
from omop_emb.backends.embedding_table import concept_metadata_table_descriptor
from omop_emb.backends.index_config import FlatIndexConfig
from omop_emb.model_registry import RegistryManager, ensure_registry_table
from omop_emb.population import PopulationScope, plan_population


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


def test_read_only_registry_does_not_create_schema() -> None:
    engine = create_engine("sqlite:///:memory:")
    store = ReadOnlyEmbeddingStore(
        engine,
        backend_type="sqlitevec",
        schema="main",
    )

    assert store.initialized is False
    assert store.registered_models() == ()
    assert inspect(engine).has_table("model_registry") is False
    store.close()


def test_explicit_registry_initialization_is_visible_to_read_only_store() -> None:
    engine = create_engine("sqlite:///:memory:")
    ensure_registry_table(engine)
    store = ReadOnlyEmbeddingStore(
        engine,
        backend_type="sqlitevec",
        schema="main",
    )

    assert store.initialized is True
    assert store.registered_models() == ()
    store.close()


def test_stored_embeddings_use_read_only_core_query() -> None:
    engine = create_engine("sqlite:///:memory:")
    registry = RegistryManager(engine)
    record = registry.register_model(
        model_name="test-model",
        provider_type="ollama",
        index_config=FlatIndexConfig(),
        dimensions=3,
    )
    table = concept_metadata_table_descriptor(record.storage_identifier)
    table.create(engine)
    with engine.begin() as connection:
        connection.execute(
            insert(table),
            [
                {
                    "concept_id": 7,
                    "domain_id": "Condition",
                    "vocabulary_id": "SNOMED",
                    "is_standard": True,
                    "is_valid": True,
                }
            ],
        )

    statements: list[str] = []

    def capture(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.strip().lower())

    event.listen(engine, "before_cursor_execute", capture)
    try:
        with ReadOnlyEmbeddingStore(
            engine,
            backend_type="sqlitevec",
            schema="main",
        ) as store:
            assert store.stored_embeddings("test-model") == (
                StoredEmbedding(7, "Condition", "SNOMED", True, True),
            )
    finally:
        event.remove(engine, "before_cursor_execute", capture)

    assert statements
    assert not any(
        statement.startswith(
            ("create ", "alter ", "drop ", "insert ", "update ", "delete ")
        )
        for statement in statements
    )


def test_read_only_registry_rejects_mutation() -> None:
    engine = create_engine("sqlite:///:memory:")
    ensure_registry_table(engine)
    registry = RegistryManager.read_only(engine)

    with pytest.raises(RuntimeError, match="opened read-only"):
        registry.register_model(
            model_name="test-model",
            provider_type="ollama",
            index_config=FlatIndexConfig(),
            dimensions=3,
        )


def test_population_plan_distinguishes_missing_and_stale_ids() -> None:
    engine = create_engine("sqlite:///:memory:")
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
    engine = create_engine("sqlite:///:memory:")
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

    assert plan.eligible_ids == frozenset({1, 2})
    assert plan.missing_ids == frozenset({1, 2})


def test_filtered_population_does_not_mark_out_of_scope_rows_stale() -> None:
    engine = create_engine("sqlite:///:memory:")
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
    engine = create_engine("sqlite:///:memory:")
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
