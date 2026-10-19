"""``maintenance prune-embeddings`` and the ``--include-invalid`` ingestion flag."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
from omop_alchemy.cdm.model.vocabulary import Concept
from omop_alchemy.cross_database import cdm_sessionmaker
from sqlalchemy import insert
from typer.testing import CliRunner

from omop_emb.backends import open_vector_store_reader, open_vector_store_writer
from omop_emb.backends.embedding_table import ConceptEmbeddingRecord
from omop_emb.backends.index_config import FlatIndexConfig
from omop_emb.cli import cli_embeddings, cli_maintenance
from omop_emb.config import MetricType

from .conftest import EMBEDDING_DIM, sqlite_cdm_database, sqlite_resolved_vector_store

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
    Concept.__table__.create(engine)
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
        return reader.get_registered_model(model_name=MODEL).updated_at


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

    cli_embeddings._add_embeddings(
        backend=None,
        cdm_sessions=None,
        resolved_model=None,
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

    cli_embeddings._add_embeddings(
        backend=None,
        cdm_sessions=None,
        resolved_model=None,
        batch_size=10,
        standard_only=False,
        include_invalid=False,
        domains=None,
        vocabularies=None,
        num_embeddings=None,
        metric_type=MetricType.L2,
    )

    assert registered == [FlatIndexConfig(metric_type=MetricType.L2)]
