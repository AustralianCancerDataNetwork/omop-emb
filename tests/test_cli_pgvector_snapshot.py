"""CLI and registry integration snapshot tests."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from omop_emb.config import IndexType, ProviderType
from omop_emb.interface import list_registered_models
from omop_emb.storage import PGVectorEmbeddingBackend
from omop_emb.storage.index_config import FlatIndexConfig

from .conftest import EMBEDDING_DIM, MODEL_NAME, PROVIDER_TYPE


@pytest.mark.pgvector
@pytest.mark.integration
def test_list_registered_models_empty(emb_engine: Engine) -> None:
    """list_registered_models returns empty tuple when no models are registered."""
    results = list_registered_models(
        emb_engine=emb_engine,
        model_name="no-such-model",
    )
    assert results == ()


@pytest.mark.pgvector
@pytest.mark.integration
def test_list_registered_models_after_registration(
    session,
    emb_engine: Engine,
    cdm_engine: Engine,
) -> None:
    """list_registered_models reflects the newly registered model."""
    backend = PGVectorEmbeddingBackend(
        emb_engine=emb_engine,
        omop_cdm_engine=cdm_engine,
    )
    backend.register_model(
        model_name=MODEL_NAME,
        provider_type=PROVIDER_TYPE,
        index_config=FlatIndexConfig(),
        dimensions=EMBEDDING_DIM,
    )

    results = list_registered_models(
        emb_engine=emb_engine,
        model_name=MODEL_NAME,
        provider_type=PROVIDER_TYPE,
        index_type=IndexType.FLAT,
    )
    assert len(results) == 1
    assert results[0].model_name == MODEL_NAME
    assert results[0].dimensions == EMBEDDING_DIM


@pytest.mark.pgvector
@pytest.mark.integration
def test_registry_survives_backend_recreate(
    session,
    emb_engine: Engine,
    cdm_engine: Engine,
) -> None:
    """Registry records persisted in Postgres survive backend object recreation."""
    backend1 = PGVectorEmbeddingBackend(
        emb_engine=emb_engine,
        omop_cdm_engine=cdm_engine,
    )
    backend1.register_model(
        model_name=MODEL_NAME,
        provider_type=PROVIDER_TYPE,
        index_config=FlatIndexConfig(),
        dimensions=EMBEDDING_DIM,
    )

    # Simulate a new backend instance (e.g. after process restart)
    backend2 = PGVectorEmbeddingBackend(
        emb_engine=emb_engine,
        omop_cdm_engine=cdm_engine,
    )
    record = backend2.get_registered_model(
        model_name=MODEL_NAME,
        provider_type=PROVIDER_TYPE,
        index_type=IndexType.FLAT,
    )
    assert record is not None
    assert record.model_name == MODEL_NAME
