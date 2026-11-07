"""``maintenance prune-embeddings`` and the ``--include-invalid`` ingestion flag."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from omop_alchemy.cdm.model.vocabulary import Concept
from omop_alchemy.cross_database import cdm_sessionmaker
from sqlalchemy import insert, text
from typer.testing import CliRunner

from omop_emb.backends import open_vector_store_reader, open_vector_store_writer
from omop_emb.backends.db_utils import table_for
from omop_emb.backends.embedding_table import ConceptEmbeddingRecord
from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.cli import cli_embeddings, cli_maintenance
from omop_emb.config import IndexType, MetricType

from .conftest import (
    EMBEDDING_DIM,
    duckdbvss_resolved_vector_store,
    registered_model,
    sqlite_cdm_database,
    sqlite_resolved_vector_store,
)

MODEL = "test-model:v1"
runner = CliRunner()


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


@pytest.fixture
def pruning_setup(tmp_path, monkeypatch):
    """Store with concept 1 (valid), 2 (invalid in the CDM) and 3 (absent from the CDM)."""
    store = sqlite_resolved_vector_store(str(tmp_path / "test.db"))
    with open_vector_store_writer(store) as writer:
        writer.register_model(
            model_name=MODEL,
            provider_type="ollama",
            index_config=FlatIndexConfig(),
            dimensions=EMBEDDING_DIM,
        )
        writer.upsert_embeddings(
            model_name=MODEL,
            records=[
                ConceptEmbeddingRecord(1, "Condition", "SNOMED", True, True),
                ConceptEmbeddingRecord(2, "Condition", "SNOMED", False, False),
                ConceptEmbeddingRecord(3, "Condition", "SNOMED", True, True),
            ],
            embeddings=np.eye(3, EMBEDDING_DIM, dtype=np.float32),
        )

    cdm = sqlite_cdm_database()
    engine, _ = cdm.create_engines()
    cdm_sessions = cdm_sessionmaker(cdm, primary=engine, vocab=engine)
    table_for(Concept).create(engine)
    with engine.begin() as connection:
        connection.execute(
            insert(Concept),
            [_concept(1), _concept(2, invalid_reason="D", standard_concept=None)],
        )

    @contextmanager
    def cdm_context():
        yield cdm_sessions

    monkeypatch.setattr(cli_maintenance, "open_cdm_sessions", cdm_context)
    monkeypatch.setattr(
        cli_maintenance.OmopEmbConfig,
        "get_config",
        classmethod(lambda _cls: SimpleNamespace(vector_store_name="default")),
    )
    monkeypatch.setattr(
        cli_maintenance.Resolver,
        "from_active_config",
        classmethod(lambda _cls: SimpleNamespace(resolve_vector_store=lambda _name: store)),
    )
    return store


def _stored_ids(store) -> set[int]:
    with open_vector_store_reader(store) as reader:
        return reader.get_stored_concept_ids(model_name=MODEL)


def _updated_at(store):
    with open_vector_store_reader(store) as reader:
        return registered_model(reader, MODEL).updated_at


def test_prune_embeddings_dry_run_reports_and_deletes_nothing(pruning_setup) -> None:
    updated_at = _updated_at(pruning_setup)

    result = runner.invoke(cli_maintenance.app, ["prune-embeddings", "--model", MODEL])

    assert result.exit_code == 0, result.output
    assert "Embeddings kept: 1" in result.output
    assert "Outside the scope: 1" in result.output
    assert "Concept no longer in the CDM: 1" in result.output
    assert "Dry run: 2 embeddings would be deleted" in result.output
    assert _stored_ids(pruning_setup) == {1, 2, 3}
    assert _updated_at(pruning_setup) == updated_at


def test_prune_embeddings_apply_deletes_and_marks_the_model_updated(pruning_setup) -> None:
    updated_at = _updated_at(pruning_setup)

    result = runner.invoke(cli_maintenance.app, ["prune-embeddings", "--model", MODEL, "--apply"])

    assert result.exit_code == 0, result.output
    assert "Deleted 2 embeddings" in result.output
    assert _stored_ids(pruning_setup) == {1}
    assert _updated_at(pruning_setup) > updated_at


def test_prune_embeddings_apply_hints_at_compact_without_compacting(pruning_setup) -> None:
    result = runner.invoke(cli_maintenance.app, ["prune-embeddings", "--model", MODEL, "--apply"])

    assert "Pass --compact" in result.output
    assert "Compacted storage" not in result.output


def test_prune_embeddings_compact_deletes_then_compacts(pruning_setup) -> None:
    result = runner.invoke(
        cli_maintenance.app, ["prune-embeddings", "--model", MODEL, "--apply", "--compact"]
    )

    assert result.exit_code == 0, result.output
    assert "Deleted 2 embeddings" in result.output
    assert "Compacted storage:" in result.output
    assert "Pass --compact" not in result.output
    assert _stored_ids(pruning_setup) == {1}


def test_prune_embeddings_compact_runs_when_nothing_is_left_to_prune(pruning_setup) -> None:
    runner.invoke(cli_maintenance.app, ["prune-embeddings", "--model", MODEL, "--apply"])

    result = runner.invoke(
        cli_maintenance.app, ["prune-embeddings", "--model", MODEL, "--apply", "--compact"]
    )

    assert result.exit_code == 0, result.output
    assert "Nothing to prune." in result.output
    assert "Compacted storage:" in result.output


def test_prune_embeddings_compact_needs_apply(pruning_setup) -> None:
    result = runner.invoke(cli_maintenance.app, ["prune-embeddings", "--model", MODEL, "--compact"])

    assert result.exit_code == 1
    assert "--compact needs --apply" in result.output
    assert _stored_ids(pruning_setup) == {1, 2, 3}


def test_prune_embeddings_include_invalid_keeps_invalid_concepts(pruning_setup) -> None:
    result = runner.invoke(
        cli_maintenance.app,
        ["prune-embeddings", "--model", MODEL, "--include-invalid", "--apply"],
    )

    assert result.exit_code == 0, result.output
    assert _stored_ids(pruning_setup) == {1, 2}


def test_prune_embeddings_standard_only_prunes_non_standard_concepts(pruning_setup) -> None:
    result = runner.invoke(
        cli_maintenance.app,
        ["prune-embeddings", "--model", MODEL, "--include-invalid", "--standard-only", "--apply"],
    )

    assert result.exit_code == 0, result.output
    assert _stored_ids(pruning_setup) == {1}


def test_prune_embeddings_for_unregistered_model_fails(pruning_setup) -> None:
    result = runner.invoke(cli_maintenance.app, ["prune-embeddings", "--model", "other:v1"])

    assert result.exit_code == 1
    assert "No registered model found for 'other:v1'" in result.output


@pytest.mark.parametrize("include_invalid, require_active", [(False, True), (True, False)])
def test_add_embeddings_skips_invalid_concepts_unless_asked(
    monkeypatch, include_invalid, require_active
) -> None:
    seen = []
    registered = []

    class FakeWriter:
        def __init__(self, **_kwargs):
            pass

        def register_model(self, index_config=None):
            registered.append(index_config)

        def count_concepts_requiring_embedding(self, *, concept_filter):
            seen.append(concept_filter)
            return 0

        def get_concepts_requiring_embedding_batched(self, **_kwargs):
            return iter(())

    monkeypatch.setattr(cli_embeddings, "EmbeddingWriterInterface", FakeWriter)
    monkeypatch.setattr(cli_embeddings, "check_concept_cdm", lambda _sessions: None)

    unused: Any = None  # the fake writer ignores the CDM and model arguments
    cli_embeddings._add_embeddings(
        backend=None,
        cdm_sessions=unused,
        resolved_model=unused,
        batch_size=10,
        standard_only=False,
        include_invalid=include_invalid,
        domains=None,
        vocabularies=None,
        num_embeddings=None,
    )

    assert [f.require_active for f in seen] == [require_active]
    assert registered == [FlatIndexConfig(metric_type=MetricType.COSINE)]


def test_add_embeddings_registers_the_requested_metric(monkeypatch) -> None:
    registered = []

    class FakeWriter:
        def __init__(self, **_kwargs):
            pass

        def register_model(self, index_config=None):
            registered.append(index_config)

        def count_concepts_requiring_embedding(self, *, concept_filter):
            return 0

        def get_concepts_requiring_embedding_batched(self, **_kwargs):
            return iter(())

    monkeypatch.setattr(cli_embeddings, "EmbeddingWriterInterface", FakeWriter)
    monkeypatch.setattr(cli_embeddings, "check_concept_cdm", lambda _sessions: None)

    unused: Any = None  # the fake writer ignores the CDM and model arguments
    cli_embeddings._add_embeddings(
        backend=None,
        cdm_sessions=unused,
        resolved_model=unused,
        batch_size=10,
        standard_only=False,
        include_invalid=False,
        domains=None,
        vocabularies=None,
        num_embeddings=None,
        metric_type=MetricType.L2,
    )

    assert registered == [FlatIndexConfig(metric_type=MetricType.L2)]


@pytest.fixture
def indexed_population(tmp_path, monkeypatch, request):
    from omop_emb import interface

    scenario = request.param
    store = duckdbvss_resolved_vector_store(str(tmp_path / "population.duckdb"))
    vectors = np.random.default_rng(5).normal(size=(8, EMBEDDING_DIM)).astype(np.float32)
    initial = 8 if scenario == "nothing_pending" else 2
    hnsw = HNSWIndexConfig(
        metric_type=MetricType.COSINE, num_neighbors=4, ef_search=32, ef_construction=32
    )
    with open_vector_store_writer(store) as backend:
        backend.register_model(
            model_name=MODEL,
            provider_type="ollama",
            dimensions=EMBEDDING_DIM,
            index_config=FlatIndexConfig(),
        )
        backend.upsert_embeddings(
            model_name=MODEL,
            records=[
                ConceptEmbeddingRecord(i, "Condition", "SNOMED", True, True)
                for i in range(1, initial + 1)
            ],
            embeddings=vectors[:initial],
        )
        backend.rebuild_index(model_name=MODEL, index_config=hnsw)

    cdm = sqlite_cdm_database()
    engine, _ = cdm.create_engines()
    sessions = cdm_sessionmaker(cdm, primary=engine, vocab=engine)

    @contextmanager
    def cdm_context():
        yield sessions

    monkeypatch.setattr(
        cli_embeddings, "_get_config", lambda: SimpleNamespace(vector_store_name="test")
    )
    monkeypatch.setattr(cli_embeddings, "_resolve_model", lambda *_: None)
    monkeypatch.setattr(cli_embeddings, "open_cdm_sessions", cdm_context)
    monkeypatch.setattr(
        cli_embeddings.Resolver,
        "from_active_config",
        classmethod(lambda _: SimpleNamespace(resolve_vector_store=lambda _: store)),
    )
    calls = []

    def embed_texts(texts, *, role, batch_size):
        assert role == interface.EmbeddingRole.DOCUMENT
        assert batch_size == 2
        calls.append(texts)
        if scenario == "provider_failure" and len(calls) == 3:
            raise RuntimeError("provider failed on the third batch")
        return [vectors[int(text.split()[-1]) - 1].tolist() for text in texts]

    model = SimpleNamespace(
        model=MODEL, provider="ollama", dimensions=lambda: EMBEDDING_DIM, embed_texts=embed_texts
    )
    monkeypatch.setattr(interface, "build_model_backend_from_resolved", lambda _: model)
    try:
        table_for(Concept).create(engine)
        with engine.begin() as connection:
            connection.execute(insert(Concept), [_concept(i) for i in range(1, 9)])

        with open_vector_store_writer(store) as backend:
            rebuild = Mock(wraps=backend.rebuild_index)
            refresh = Mock(wraps=backend.refresh_model_updated_at_timestamp)
            monkeypatch.setattr(backend, "rebuild_index", rebuild)
            monkeypatch.setattr(backend, "refresh_model_updated_at_timestamp", refresh)
            monkeypatch.setattr(cli_embeddings, "open_vector_store_writer", lambda _: backend)
            write = backend._upsert_embeddings_impl
            writes = []

            def upsert(**kwargs):
                writes.append(kwargs)
                if scenario == "store_failure" and len(writes) == 3:
                    raise RuntimeError("store failed on the third batch")
                return write(**kwargs)

            monkeypatch.setattr(backend, "_upsert_embeddings_impl", upsert)
            yield SimpleNamespace(
                backend=backend,
                sessions=sessions,
                vectors=vectors,
                hnsw=hnsw,
                rebuild=rebuild,
                refresh=refresh,
                scenario=scenario,
            )
    finally:
        engine.dispose()


@pytest.mark.duckdbvss
@pytest.mark.db_dialect
@pytest.mark.parametrize(
    "indexed_population",
    ["success", "provider_failure", "store_failure", "nothing_pending"],
    indirect=True,
)
def test_population_of_reopened_hnsw_syncs_once_even_after_a_partial_failure(
    indexed_population, capsys
):
    state = indexed_population
    kwargs: dict[str, Any] = {
        "backend": state.backend,
        "cdm_sessions": state.sessions,
        "resolved_model": None,
        "batch_size": 2,
        "standard_only": False,
        "include_invalid": False,
        "domains": None,
        "vocabularies": None,
        "num_embeddings": None,
    }
    failed = state.scenario in ("provider_failure", "store_failure")
    if failed:
        with pytest.raises(cli_embeddings.typer.Exit) as error:
            cli_embeddings._add_embeddings(**kwargs)
        assert error.value.exit_code == 1
        assert isinstance(error.value.__context__, RuntimeError)
        assert "failed on the third batch" in str(error.value.__context__)
    else:
        cli_embeddings._add_embeddings(**kwargs)
    expected = 6 if failed else 8
    assert state.backend.get_embedding_count(model_name=MODEL) == expected
    assert state.rebuild.call_count == (0 if state.scenario == "nothing_pending" else 1)
    assert state.refresh.call_count == state.rebuild.call_count
    assert state.backend.get_registered_model(model_name=MODEL).index_config == state.hnsw
    assert state.backend.physical_indexes(MODEL)
    (matches,) = state.backend.get_nearest_concepts(
        model_name=MODEL,
        metric_type=MetricType.COSINE,
        query_embeddings=state.vectors[expected - 1 : expected],
        k=1,
    )
    assert matches[0].concept_id == expected
    assert matches[0].domain_id == "Condition" and matches[0].vocabulary_id == "SNOMED"
    assert matches[0].is_standard and matches[0].is_active
    if state.scenario == "success":
        assert "Processed 6 concepts." in capsys.readouterr().out


@pytest.mark.duckdbvss
@pytest.mark.db_dialect
@pytest.mark.parametrize(
    "indexed_population, target",
    [
        ("success", "same"),
        ("success", "changed"),
        ("success", "flat"),
        ("nothing_pending", "missing"),
    ],
    indirect=["indexed_population"],
)
def test_combined_population_reuses_only_a_matching_existing_index(indexed_population, target):
    state = indexed_population
    if target == "missing":
        with state.backend.emb_engine.begin() as connection:
            for statement in state.backend.drop_index_sql(MODEL):
                connection.execute(text(statement))
        assert not state.backend.physical_indexes(MODEL)
    neighbors = 8 if target == "changed" else 4
    cli_embeddings.add_embeddings_with_index(
        model_name=MODEL,
        batch_size=2,
        standard_only=False,
        include_invalid=False,
        vocabularies=None,
        domains=None,
        num_embeddings=None,
        metric_type=MetricType.COSINE,
        index_type=IndexType.FLAT if target == "flat" else IndexType.HNSW,
        index_hnsw_num_neighbors=neighbors,
        index_hnsw_ef_search=32,
        index_hnsw_ef_construction=32,
    )
    assert state.rebuild.call_count == (2 if target in ("changed", "flat") else 1)
    config = state.backend.get_registered_model(model_name=MODEL).index_config
    if target == "flat":
        assert config == FlatIndexConfig()
        assert not state.backend.physical_indexes(MODEL)
    else:
        assert config == HNSWIndexConfig(
            metric_type=MetricType.COSINE, num_neighbors=neighbors, ef_search=32, ef_construction=32
        )
        assert state.backend.physical_indexes(MODEL)


@pytest.mark.duckdbvss
@pytest.mark.db_dialect
@pytest.mark.parametrize("indexed_population", ["nothing_pending"], indirect=True)
def test_create_index_still_forces_rebuilding_a_matching_index(indexed_population):
    cli_embeddings.create_index(
        model_name=MODEL,
        metric_type=MetricType.COSINE,
        index_type=IndexType.HNSW,
        index_hnsw_num_neighbors=4,
        index_hnsw_ef_search=32,
        index_hnsw_ef_construction=32,
    )
    assert indexed_population.rebuild.call_count == 1


@pytest.mark.duckdbvss
@pytest.mark.db_dialect
@pytest.mark.parametrize("indexed_population", ["provider_failure"], indirect=True)
def test_combined_command_keeps_population_failure_sync(indexed_population, monkeypatch):
    create = Mock(wraps=cli_embeddings._create_index)
    monkeypatch.setattr(cli_embeddings, "_create_index", create)
    with pytest.raises(cli_embeddings.typer.Exit):
        cli_embeddings.add_embeddings_with_index(
            model_name=MODEL,
            batch_size=2,
            standard_only=False,
            include_invalid=False,
            vocabularies=None,
            domains=None,
            num_embeddings=None,
            metric_type=MetricType.COSINE,
            index_type=IndexType.HNSW,
            index_hnsw_num_neighbors=8,
            index_hnsw_ef_search=32,
            index_hnsw_ef_construction=32,
        )
    create.assert_not_called()
    assert indexed_population.rebuild.call_count == 1
    assert indexed_population.refresh.call_count == 1
    assert indexed_population.backend.physical_indexes(MODEL)
