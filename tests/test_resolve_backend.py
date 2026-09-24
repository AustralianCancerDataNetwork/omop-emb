"""resolve_backend() is the real production entry point every CLI/config-
driven caller goes through to build an EmbeddingBackend. This suite
tests regressions in resolve_backend() itself (e.g. wrong dialect detection,
a dropped registry_schema_translate_map key, wiring a different backend class)
"""

from __future__ import annotations

import pytest

from omop_emb.backends.base_backend import resolve_backend
from omop_emb.backends.pgvector import PGVectorEmbeddingBackend
from omop_emb.config import BackendType

from .conftest import EMBEDDING_DIM, MODEL_NAME, PROVIDER_TYPE

pytestmark = [pytest.mark.postgresql, pytest.mark.db_dialect]


def test_resolve_backend_builds_a_working_pgvector_backend(pg_db):
    backend = resolve_backend(BackendType.PGVECTOR, database=pg_db.resolved)
    try:
        assert isinstance(backend, PGVectorEmbeddingBackend)

        record = backend.register_model(
            model_name=MODEL_NAME, provider_type=PROVIDER_TYPE, dimensions=EMBEDDING_DIM
        )
        assert record.model_name == MODEL_NAME

        registered = {r.model_name for r in backend.get_registered_models()}
        assert MODEL_NAME in registered
    finally:
        for record in backend.get_registered_models():
            try:
                backend.delete_model(model_name=record.model_name)
            except Exception:
                pass


def test_resolve_backend_accepts_the_backend_type_as_a_plain_string(pg_db):
    """Callers resolving from a `[vector_stores.*]` config entry pass a plain
    string, not the BackendType enum member."""
    backend = resolve_backend("pgvector", database=pg_db.resolved)
    try:
        assert isinstance(backend, PGVectorEmbeddingBackend)
    finally:
        for record in backend.get_registered_models():
            try:
                backend.delete_model(model_name=record.model_name)
            except Exception:
                pass


def test_resolve_backend_rejects_an_unknown_backend_type(pg_db):
    with pytest.raises(RuntimeError, match="Unknown backend"):
        resolve_backend("not-a-real-backend", database=pg_db.resolved)
