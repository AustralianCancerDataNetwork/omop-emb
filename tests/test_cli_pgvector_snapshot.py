"""Backend registry integration snapshot tests."""

from __future__ import annotations

import pytest

from omop_emb.backends.index_config import FlatIndexConfig

from .conftest import EMBEDDING_DIM, MODEL_NAME, PROVIDER_TYPE


@pytest.mark.pgvector
@pytest.mark.integration
def test_get_registered_models_filter_without_match(pg_backend) -> None:
    results = pg_backend.get_registered_models(
        model_name="no-such-model",
    )
    assert results == ()


@pytest.mark.pgvector
@pytest.mark.integration
def test_get_registered_models_filters_after_registration(pg_backend) -> None:
    pg_backend.register_model(
        model_name=MODEL_NAME,
        provider_type=PROVIDER_TYPE,
        index_config=FlatIndexConfig(),
        dimensions=EMBEDDING_DIM,
    )
    results = pg_backend.get_registered_models(
        model_name=MODEL_NAME,
        provider_type=PROVIDER_TYPE,
    )
    assert len(results) == 1
    assert results[0].model_name == MODEL_NAME
