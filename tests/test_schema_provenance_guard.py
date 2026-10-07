"""schema-provenance guard wired into PGVectorEmbeddingBackend's
registry-schema and storage-table creation (ensure_registry_table(),
create_pg_embedding_table()).

Only the "fires on a genuinely reconfigured schema" case is covered here.
The guard's own agree/no-op/test_only semantics are already exhaustively
covered at the primitive level in oa-configurator's own test suite; what's
worth proving per consuming repo is that this call site is actually wired
to it, and a wiring mistake would show up here too.

pg_engine is a real, committing engine, so every provenance row this test
writes is a genuine commit. The Role-tag rows are reset around each test,
and the registry row around the test that rewrites it.
"""

from __future__ import annotations

import pytest
from oa_configurator import GenericDatabaseConfig, Resolver, SchemaDriftError
from oa_configurator.cli import app
from oa_configurator.testing import guarded_resolver, isolated_test_schema, reset_schema_registry_rows
from typer.testing import CliRunner

from omop_emb.backends.base_backend import _open_writer
from omop_emb.model_registry import REGISTRY_SCHEMA_KEY

from .conftest import EMBEDDING_DIM, MODEL_NAME, PROVIDER_TYPE

pytestmark = [pytest.mark.postgresql, pytest.mark.db_dialect]


@pytest.fixture(autouse=True)
def _fresh_role_rows(pg_engine, cleanup_after_test):
    reset_schema_registry_rows(cleanup_after_test, pg_engine)


def _resolver(pg_db, *, database_config_name: str, schema: str) -> Resolver:
    """Resolver with a generic database entry on pg_db's own connection name,
    test_only flipped to False via guarded_resolver() so the guard doesn't
    no-op against pg_db's own test-only marking.

    Reuses pg_db's own connection name deliberately as StackConfig rejects
    a test_only connection that duplicates a non-test_only one's physical identity.
    """
    resolver = guarded_resolver(pg_db.resolved)
    return resolver.with_overrides(
        databases={
            database_config_name: GenericDatabaseConfig(
                connection=pg_db.resolved.connection.name, schema_name=schema
            )
        },
    )


def _resolved(pg_db, *, database_config_name: str, schema: str):
    return _resolver(pg_db, database_config_name=database_config_name, schema=schema).resolve_database(
        database_config_name
    )


def test_backend_construction_is_unaffected_by_the_primary_schema_changing(pg_db, pg_engine):
    with (
        isolated_test_schema(pg_engine, prefix="emb_guard_a") as schema_a,
        isolated_test_schema(pg_engine, prefix="emb_guard_b") as schema_b,
    ):
        resolved_a = _resolved(pg_db, database_config_name="emb_guard_a", schema=schema_a)
        backend_a = _open_writer("pgvector", database=resolved_a)
        assert backend_a is not None

        # A different config entry: the registry row belongs to the physical
        # database, so it is shared across entries pointed at it.
        resolved_b = _resolved(pg_db, database_config_name="emb_guard_b", schema=schema_b)
        backend_b = _open_writer("pgvector", database=resolved_b)
        assert backend_b is not None


def test_registry_schema_guard_fires_on_genuine_registry_drift(pg_db, pg_engine, cleanup_after_test, monkeypatch):
    reset_schema_registry_rows(cleanup_after_test, pg_engine, [REGISTRY_SCHEMA_KEY])
    resolver = _resolver(pg_db, database_config_name="emb_guard", schema="unrelated_primary_schema")
    monkeypatch.setattr("oa_configurator.cli.load_stack_config", lambda: resolver.config)

    result = CliRunner().invoke(
        app,
        [
            "acknowledge-schema-migration",
            "--database", "emb_guard",
            "--schema-tag", REGISTRY_SCHEMA_KEY,
            "--new-schema", "a_previous_registry_schema_that_is_not_current",
            "--reason", "test: force a stale baseline",
        ],
    )
    assert result.exit_code == 0, result.output

    with pytest.raises(SchemaDriftError):
        _open_writer("pgvector", database=resolver.resolve_database("emb_guard"))


def test_primary_schema_guard_fires_on_genuine_primary_drift(pg_db, pg_engine):
    """Drift is now caught at create_engine() construction time (see
    oa-configurator's architectural note on guard_schema_provenance_for()):
    reusing one database_config_name ("emb_guard") with a genuinely
    different primary schema raises while building backend_b itself,
    before register_model() is even reachable."""
    with (
        isolated_test_schema(pg_engine, prefix="emb_guard_primary_a") as schema_a,
        isolated_test_schema(pg_engine, prefix="emb_guard_primary_b") as schema_b,
    ):
        resolved_a = _resolved(pg_db, database_config_name="emb_guard", schema=schema_a)
        backend_a = _open_writer("pgvector", database=resolved_a)
        backend_a.register_model(
            model_name=MODEL_NAME, provider_type=PROVIDER_TYPE, dimensions=EMBEDDING_DIM
        )

        resolved_b = _resolved(pg_db, database_config_name="emb_guard", schema=schema_b)
        with pytest.raises(SchemaDriftError):
            _open_writer("pgvector", database=resolved_b)
