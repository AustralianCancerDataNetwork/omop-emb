"""``--partial-domain-indexes`` and ``--partial-valid-only`` on the index commands."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import typer.main
from typer.testing import CliRunner

from omop_emb.backends import open_vector_store_writer
from omop_emb.cli import cli_embeddings, cli_maintenance
from omop_emb.config import BackendType, IndexType

from .conftest import (
    EMBEDDING_DIM,
    MODEL_NAME,
    PROVIDER_TYPE,
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


def _register_model(resolved) -> None:
    with open_vector_store_writer(resolved) as writer:
        writer.register_model(
            model_name=MODEL_NAME, provider_type=PROVIDER_TYPE, dimensions=EMBEDDING_DIM
        )


@pytest.fixture
def store(vector_store, monkeypatch):
    """Each backend's store with a model registered, and the CLI pointed at it."""
    _register_model(vector_store)
    _use_store(monkeypatch, vector_store)


@pytest.mark.parametrize(
    ("app", "command"),
    [
        (cli_maintenance.app, "rebuild-index"),
        (cli_embeddings.app, "create-index"),
        (cli_embeddings.app, "add-embeddings-with-index"),
    ],
)
def test_the_commands_that_build_an_index_list_the_options(app, command):
    params = typer.main.get_group(app).commands[command].params
    options = {option for param in params for option in (*param.opts, *param.secondary_opts)}

    assert {
        "--partial-domain-indexes",
        "--no-partial-domain-indexes",
        "--partial-valid-only",
        "--no-partial-valid-only",
    } <= options


def test_valid_only_needs_partial_indexes(store) -> None:
    result = _rebuild(IndexType.HNSW, "--partial-valid-only")

    assert result.exit_code == 2
    assert "partial_domain_indexes" in result.output


def test_partial_options_are_rejected_for_a_flat_index(store) -> None:
    result = _rebuild(IndexType.FLAT, "--partial-domain-indexes")

    assert result.exit_code == 2
    assert "HNSW" in result.output


@pytest.mark.duckdbvss
def test_partial_indexes_are_rejected_on_a_backend_without_them(
    duckdbvss_store, monkeypatch
) -> None:
    _register_model(duckdbvss_store)
    _use_store(monkeypatch, duckdbvss_store)

    result = _rebuild(IndexType.HNSW, "--partial-domain-indexes")

    assert result.exit_code == 1
    assert "partial_domain_indexes" in result.output
    assert f"'{BackendType.DUCKDBVSS.value}'" in result.output
