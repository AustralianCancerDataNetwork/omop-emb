"""_open_writer() builds every writable EmbeddingBackend, behind
open_vector_store_writer(). This suite tests regressions in it (e.g. wrong dialect detection,
a dropped registry_schema_translate_map key, wiring a different backend class)
"""

from __future__ import annotations

import pytest

from omop_emb.backends.base_backend import _backend_class_for, _open_writer
from omop_emb.backends.pgvector import PGVectorEmbeddingBackend
from omop_emb.config import BackendType
from omop_emb.utils.errors import (
    EmbeddingBackendConfigurationError,
    EmbeddingBackendDependencyError,
    UnknownEmbeddingBackendError,
)

from .conftest import EMBEDDING_DIM, MODEL_NAME, PROVIDER_TYPE, sqlite_resolved_database

pytestmark = [pytest.mark.postgresql, pytest.mark.db_dialect]


def test_open_writer_builds_a_working_pgvector_backend(pg_db):
    backend = _open_writer(BackendType.PGVECTOR, database=pg_db.resolved)
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


def test_open_writer_accepts_the_backend_type_as_a_plain_string(pg_db):
    """Callers resolving from a `[vector_stores.*]` config entry pass a plain
    string, not the BackendType enum member."""
    backend = _open_writer("pgvector", database=pg_db.resolved)
    try:
        assert isinstance(backend, PGVectorEmbeddingBackend)
    finally:
        for record in backend.get_registered_models():
            try:
                backend.delete_model(model_name=record.model_name)
            except Exception:
                pass


def test_open_writer_rejects_an_unknown_backend_type(pg_db):
    with pytest.raises(UnknownEmbeddingBackendError, match="Invalid backend type"):
        _open_writer("not-a-real-backend", database=pg_db.resolved)


def test_open_writer_rejects_a_dialect_mismatch():
    with pytest.raises(EmbeddingBackendConfigurationError, match="PostgreSQL"):
        _open_writer(BackendType.PGVECTOR, database=sqlite_resolved_database())


def test_missing_pgvector_dependency_raises_dependency_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "omop_emb.backends.pgvector", None)
    with pytest.raises(EmbeddingBackendDependencyError, match="omop-emb\\[pgvector\\]"):
        _backend_class_for(BackendType.PGVECTOR, dialect="postgresql")
