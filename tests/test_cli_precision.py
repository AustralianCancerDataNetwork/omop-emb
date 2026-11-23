"""``maintenance rebuild-index --index-precision`` and its rejections."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from omop_emb.backends import open_vector_store_writer
from omop_emb.cli import cli_maintenance
from omop_emb.config import BackendType, IndexType, VectorPrecision

from .conftest import (
    EMBEDDING_DIM,
    MODEL_NAME,
    PROVIDER_TYPE,
)

runner = CliRunner()


@pytest.mark.duckdbvss
def test_rebuild_index_rejects_half_precision_on_a_backend_without_it(
    duckdbvss_store, monkeypatch
) -> None:
    resolved = duckdbvss_store
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

    result = runner.invoke(
        cli_maintenance.app,
        [
            "rebuild-index",
            "--model",
            MODEL_NAME,
            "--index-type",
            IndexType.HNSW.value,
            "--index-precision",
            VectorPrecision.FLOAT16.value,
        ],
    )

    assert result.exit_code == 1
    assert f"'{VectorPrecision.FLOAT16.value}'" in result.output
    assert f"'{BackendType.DUCKDBVSS.value}'" in result.output
