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
    registered_model,
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


def _register_model(resolved) -> None:
    with open_vector_store_writer(resolved) as writer:
        writer.register_model(
            model_name=MODEL_NAME, provider_type=PROVIDER_TYPE, dimensions=EMBEDDING_DIM
        )


def _build_hnsw(resolved) -> None:
    with open_vector_store_writer(resolved) as writer:
        writer.rebuild_index(
            model_name=MODEL_NAME, index_config=HNSWIndexConfig(metric_type=MetricType.COSINE)
        )


@pytest.fixture
def store(vector_store, monkeypatch):
    """Each backend's store with a model registered, and the CLI pointed at it."""
    _register_model(vector_store)
    _use_store(monkeypatch, vector_store)
    return vector_store


def test_set_ef_search_needs_a_setting(store) -> None:
    result = _set()

    assert result.exit_code == 1
    assert "--value" in result.output


def test_set_ef_search_rejects_a_model_without_an_hnsw_index(store) -> None:
    result = _set("--value", "64")

    assert result.exit_code == 1
    assert "no HNSW index" in result.output


def test_set_ef_search_reports_an_unknown_model(store) -> None:
    result = runner.invoke(
        cli_maintenance.app, ["set-ef-search", "--model", "missing", "--value", "64"]
    )

    assert result.exit_code == 1
    assert "No registered model" in result.output


@pytest.fixture
def hnsw_model_store(hnsw_store, monkeypatch):
    """Each HNSW backend's store with a model that has an HNSW index, and the CLI pointed at it."""
    _register_model(hnsw_store)
    _build_hnsw(hnsw_store)
    _use_store(monkeypatch, hnsw_store)
    return hnsw_store


def test_set_ef_search_updates_the_registry(hnsw_model_store) -> None:
    result = _set("--value", "64")

    assert result.exit_code == 0, result.output
    assert "ef_search=64" in result.output
    with open_vector_store_reader(hnsw_model_store) as reader:
        config = registered_model(reader, MODEL_NAME).index_config
    assert isinstance(config, HNSWIndexConfig)
    assert config.ef_search == 64


@pytest.mark.duckdbvss
def test_set_ef_search_rejects_max_scan_tuples_on_a_backend_without_it(
    duckdbvss_store, monkeypatch
) -> None:
    _register_model(duckdbvss_store)
    _build_hnsw(duckdbvss_store)
    _use_store(monkeypatch, duckdbvss_store)

    result = _set("--max-scan-tuples", "100000")

    assert result.exit_code == 1
    assert "max_scan_tuples" in result.output
