"""``maintenance set-ef-search`` and its rejections."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from omop_emb.backends import open_vector_store_reader, open_vector_store_writer
from omop_emb.backends.index_config import HNSWIndexConfig
from omop_emb.cli import cli_maintenance
from omop_emb.config import MetricType

from .conftest import (
    EMBEDDING_DIM,
    MODEL_NAME,
    PROVIDER_TYPE,
    duckdbvss_resolved_vector_store,
    registered_model,
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


def _set(*args: str):
    return runner.invoke(cli_maintenance.app, ["set-ef-search", "--model", MODEL_NAME, *args])


@pytest.fixture
def sqlite_store(tmp_path, monkeypatch):
    resolved = sqlite_resolved_vector_store(str(tmp_path / "test.db"))
    with open_vector_store_writer(resolved) as writer:
        writer.register_model(
            model_name=MODEL_NAME, provider_type=PROVIDER_TYPE, dimensions=EMBEDDING_DIM
        )
    _use_store(monkeypatch, resolved)
    return resolved


def test_set_ef_search_needs_a_setting(sqlite_store) -> None:
    result = _set()

    assert result.exit_code == 1
    assert "--value" in result.output


def test_set_ef_search_rejects_a_model_without_an_hnsw_index(sqlite_store) -> None:
    result = _set("--value", "64")

    assert result.exit_code == 1
    assert "no HNSW index" in result.output


def test_set_ef_search_reports_an_unknown_model(sqlite_store) -> None:
    result = runner.invoke(
        cli_maintenance.app, ["set-ef-search", "--model", "missing", "--value", "64"]
    )

    assert result.exit_code == 1
    assert "No registered model" in result.output


@pytest.mark.duckdbvss
@pytest.mark.db_dialect
class TestOnDuckDB:
    @pytest.fixture
    def duckdb_store(self, tmp_path, monkeypatch):
        pytest.importorskip("duckdb_sqlalchemy")
        resolved = duckdbvss_resolved_vector_store(str(tmp_path / "test.duckdb"))
        with open_vector_store_writer(resolved) as writer:
            writer.register_model(
                model_name=MODEL_NAME, provider_type=PROVIDER_TYPE, dimensions=EMBEDDING_DIM
            )
            writer.rebuild_index(
                model_name=MODEL_NAME,
                index_config=HNSWIndexConfig(metric_type=MetricType.COSINE),
            )
        _use_store(monkeypatch, resolved)
        return resolved

    def test_set_ef_search_updates_the_registry(self, duckdb_store) -> None:
        result = _set("--value", "64")

        assert result.exit_code == 0, result.output
        assert "ef_search=64" in result.output
        with open_vector_store_reader(duckdb_store) as reader:
            config = registered_model(reader, MODEL_NAME).index_config
        assert isinstance(config, HNSWIndexConfig)
        assert config.ef_search == 64

    def test_set_ef_search_rejects_max_scan_tuples(self, duckdb_store) -> None:
        result = _set("--max-scan-tuples", "100000")

        assert result.exit_code == 1
        assert "max_scan_tuples" in result.output
