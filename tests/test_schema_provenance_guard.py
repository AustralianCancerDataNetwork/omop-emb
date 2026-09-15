"""schema-provenance guard wired into PGVectorEmbeddingBackend's
registry-schema and storage-table creation (ensure_registry_table(),
create_pg_embedding_table()).

Only the "fires on a genuinely reconfigured schema" case is covered here.
The guard's own agree/no-op/test_only semantics are already exhaustively
covered at the primitive level in oa-configurator's own test suite; what's
worth proving per consuming repo is that this call site is actually wired
to it, and a wiring mistake would show up here too.

pg_engine is a real, committing engine, so every provenance row this test
writes is a genuine commit. cleanup_after_test deletes this test's own
schema_provenance rows at teardown (see Phase 10.12 in the plan).
"""

from __future__ import annotations

import dataclasses
import uuid

import pytest
from oa_configurator import SchemaDriftError, record_schema_provenance
from oa_configurator.domains.resources.sql import SCHEMA_PROVENANCE_SCHEMA, _schema_provenance_table
from oa_configurator.testing import delete_rows_on_cleanup, isolated_test_schema

from omop_emb.backends.pgvector.pg_backend import PGVectorEmbeddingBackend
from omop_emb.config import MODEL_REGISTRY_SCHEMA

pytestmark = [pytest.mark.postgresql, pytest.mark.db_dialect]


def _resolved(pg_db, *, database_name: str, schema: str):
    """pg_db.resolved with a unique name (the guard's own key includes it),
    schema_name overridden, and connection.test_only forced False so the
    guard doesn't no-op against pg_db's own test-only marking.
    """
    return dataclasses.replace(
        pg_db.resolved,
        name=database_name,
        schema_name=schema,
        connection=dataclasses.replace(pg_db.resolved.connection, test_only=False),
    )


def _establish_registry_baseline(pg_db, pg_engine, cleanup_after_test) -> None:
    """The registry's own provenance row is keyed by shared_as=MODEL_REGISTRY_SCHEMA
    The phyical registry tables outlives any single test run, so a delete-only reset
    leaves "table already has rows, but no provenance record". 
    Overwrite it with a known-correct baseline instead, via record_schema_provenance
    (which always overwrites, no "already populated" check), then register cleanup.
    """
    table = _schema_provenance_table(SCHEMA_PROVENANCE_SCHEMA)
    with pg_engine.begin() as connection:
        record_schema_provenance(
            connection,
            pg_db.resolved,
            role=MODEL_REGISTRY_SCHEMA,
            new_schema=MODEL_REGISTRY_SCHEMA,
            reason="test setup: establish a known-correct baseline",
            shared_as=MODEL_REGISTRY_SCHEMA,
        )
    delete_rows_on_cleanup(
        cleanup_after_test, pg_engine, table, table.c.database_name == MODEL_REGISTRY_SCHEMA
    )


def test_backend_construction_is_unaffected_by_the_primary_schema_changing(
    pg_db, pg_engine, cleanup_after_test
):
    _establish_registry_baseline(pg_db, pg_engine, cleanup_after_test)
    with (
        isolated_test_schema(pg_engine, prefix="emb_guard_a") as schema_a,
        isolated_test_schema(pg_engine, prefix="emb_guard_b") as schema_b,
    ):
        database_name = f"emb_guard_db_{uuid.uuid4().hex[:8]}"
        resolved_a = _resolved(pg_db, database_name=database_name, schema=schema_a)
        engine_a = pg_engine.execution_options(schema_translate_map={"primary": schema_a})
        backend_a = PGVectorEmbeddingBackend(emb_engine=engine_a, resolved=resolved_a)
        assert backend_a is not None

        # A different database_name too: the registry provenance row must
        # be shared correctly across distinct database entries pointed at
        # the same connection, not just across two calls with one name.
        resolved_b = _resolved(
            pg_db, database_name=f"emb_guard_db_{uuid.uuid4().hex[:8]}", schema=schema_b
        )
        engine_b = pg_engine.execution_options(schema_translate_map={"primary": schema_b})
        backend_b = PGVectorEmbeddingBackend(emb_engine=engine_b, resolved=resolved_b)
        assert backend_b is not None


def test_registry_schema_guard_fires_on_genuine_registry_drift(pg_db, pg_engine, cleanup_after_test):
    table = _schema_provenance_table(SCHEMA_PROVENANCE_SCHEMA)
    delete_rows_on_cleanup(
        cleanup_after_test, pg_engine, table, table.c.database_name == MODEL_REGISTRY_SCHEMA
    )
    resolved = _resolved(
        pg_db,
        database_name=f"emb_guard_db_{uuid.uuid4().hex[:8]}",
        schema="unrelated_primary_schema",
    )

    with pg_engine.begin() as connection:
        record_schema_provenance(
            connection,
            resolved,
            role=MODEL_REGISTRY_SCHEMA,
            new_schema="a_previous_registry_schema_that_is_not_current",
            reason="test: force a stale baseline",
            shared_as=MODEL_REGISTRY_SCHEMA,
        )

    engine = pg_engine.execution_options(schema_translate_map={"primary": "unrelated_primary_schema"})
    with pytest.raises(SchemaDriftError):
        PGVectorEmbeddingBackend(emb_engine=engine, resolved=resolved)
