"""``maintenance rebuild-index`` and its ``--metric-type`` handling on sqlite-vec."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from omop_emb.backends import open_vector_store_reader, open_vector_store_writer
from omop_emb.cli import cli_maintenance
from omop_emb.config import MetricType

from .conftest import MODEL_NAME, PROVIDER_TYPE, EMBEDDING_DIM, sqlite_resolved_vector_store

runner = CliRunner()


@pytest.fixture
def store(tmp_path, monkeypatch):
    resolved = sqlite_resolved_vector_store(str(tmp_path / "test.db"))
    with open_vector_store_writer(resolved) as writer:
        writer.register_model(
            model_name=MODEL_NAME, provider_type=PROVIDER_TYPE, dimensions=EMBEDDING_DIM
        )
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
    return resolved


def _metric(store) -> MetricType | None:
    with open_vector_store_reader(store) as reader:
        return reader.get_registered_model(model_name=MODEL_NAME).metric_type


def test_rebuild_index_metric_type_changes_the_sqlite_vec_metric(store) -> None:
    assert _metric(store) == MetricType.COSINE

    result = runner.invoke(
        cli_maintenance.app, ["rebuild-index", "--model", MODEL_NAME, "--metric-type", "l2"]
    )

    assert result.exit_code == 0, result.output
    assert "(metric=l2)" in result.output
    assert _metric(store) == MetricType.L2


def test_rebuild_index_without_metric_type_keeps_the_sqlite_vec_metric(store) -> None:
    runner.invoke(cli_maintenance.app, ["rebuild-index", "--model", MODEL_NAME, "--metric-type", "l1"])

    result = runner.invoke(cli_maintenance.app, ["rebuild-index", "--model", MODEL_NAME])

    assert result.exit_code == 0, result.output
    assert _metric(store) == MetricType.L1
