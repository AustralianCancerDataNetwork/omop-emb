"""``--partial-domain-indexes`` and ``--partial-valid-only`` on the index commands."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from omop_emb.backends import open_vector_store_writer
from omop_emb.cli import cli_embeddings, cli_maintenance
from omop_emb.config import BackendType, IndexType

from .conftest import (
    EMBEDDING_DIM,
    MODEL_NAME,
    PROVIDER_TYPE,
    duckdbvss_resolved_vector_store,
    sqlite_resolved_vector_store,
)

runner = CliRunner()


def _use_store(monkeypatch, resolved) -> None:
    monkeypatch.setattr(
        cli_maintenance.OmopEmbConfig,
        "get_config",
        classmethod(lambda _cls: SimpleNamespace(vector_store_name="default")),
    )
    monkeypatch.setattr(
        cli_maintenance.Resolver,
        "from_active_config",
        classmethod(lambda _cls: SimpleNamespace(resolve_vector_store=lambda _name: resolved)),
    )


def _rebuild(index_type: IndexType, *options: str):
    return runner.invoke(
        cli_maintenance.app,
        ["rebuild-index", "--model", MODEL_NAME, "--index-type", index_type.value, *options],
    )


@pytest.fixture
def sqlite_store(tmp_path, monkeypatch):
    resolved = sqlite_resolved_vector_store(str(tmp_path / "test.db"))
    with open_vector_store_writer(resolved) as writer:
        writer.register_model(
            model_name=MODEL_NAME, provider_type=PROVIDER_TYPE, dimensions=EMBEDDING_DIM
        )
    _use_store(monkeypatch, resolved)


@pytest.mark.parametrize(
    ("app", "command"),
    [
        (cli_maintenance.app, "rebuild-index"),
        (cli_embeddings.app, "create-index"),
        (cli_embeddings.app, "add-embeddings-with-index"),
    ],
)
def test_the_commands_that_build_an_index_list_the_options(app, command):
    result = runner.invoke(app, [command, "--help"], env={"COLUMNS": "200"})

    assert "--partial-domain-indexes" in result.output
    assert "--no-partial-domain-indexes" in result.output
    assert "--partial-valid-only" in result.output
    assert "--no-partial-valid-only" in result.output


def test_valid_only_needs_partial_indexes(sqlite_store) -> None:
    result = _rebuild(IndexType.HNSW, "--partial-valid-only")

    assert result.exit_code == 2
    assert "partial_domain_indexes" in result.output


def test_partial_options_are_rejected_for_a_flat_index(sqlite_store) -> None:
    result = _rebuild(IndexType.FLAT, "--partial-domain-indexes")

    assert result.exit_code == 2
    assert "HNSW" in result.output


@pytest.mark.duckdbvss
@pytest.mark.db_dialect
def test_partial_indexes_are_rejected_on_a_backend_without_them(tmp_path, monkeypatch) -> None:
    resolved = duckdbvss_resolved_vector_store(str(tmp_path / "test.duckdb"))
    with open_vector_store_writer(resolved) as writer:
        writer.register_model(
            model_name=MODEL_NAME, provider_type=PROVIDER_TYPE, dimensions=EMBEDDING_DIM
        )
    _use_store(monkeypatch, resolved)

    result = _rebuild(IndexType.HNSW, "--partial-domain-indexes")

    assert result.exit_code == 1
    assert "partial_domain_indexes" in result.output
    assert f"'{BackendType.DUCKDBVSS.value}'" in result.output
