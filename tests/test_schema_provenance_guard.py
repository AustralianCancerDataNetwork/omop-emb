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
from oa_configurator.testing import cleanup_schema_registry_rows, isolated_test_schema

from omop_emb.backends.pgvector.pg_backend import PGVectorEmbeddingBackend
from omop_emb.config import MODEL_REGISTRY_SCHEMA
from omop_emb.model_registry import REGISTRY_SCHEMA_KEY

from .conftest import EMBEDDING_DIM, MODEL_NAME, PROVIDER_TYPE

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


def _restore_registry_baseline(pg_engine) -> None:
    """(Re-)establish the registry's schema_tag=REGISTRY_SCHEMA_KEY row at
    its known-correct value.

    Never deleted, only ever overwritten: the physical registry schema/table
    is genuinely persistent, never dropped between tests or between runs
    (like the self-referential reservation row ensure_registry_table() also
    registers, whose value never changes at all and is never touched here).
    A delete-only reset would leave that persistent content "populated with
    no baseline" for whichever test or run touches it next, since nothing
    ever drops the real schema/table alongside the deleted bookkeeping row.
    record_schema_provenance always overwrites, no "already populated" check.
    """
    with pg_engine.begin() as connection:
        record_schema_provenance(
            connection,
            database_name=MODEL_REGISTRY_SCHEMA,
            schema_tag=REGISTRY_SCHEMA_KEY,
            new_physical_schema=MODEL_REGISTRY_SCHEMA,
            reason="test setup/teardown: restore the known-correct baseline",
        )


def _establish_registry_baseline(pg_db, pg_engine, cleanup_after_test) -> None:
    """Establish the registry's baseline for this test, and restore it again
    at teardown in case this test (or a helper it calls) changes it."""
    _restore_registry_baseline(pg_engine)
    cleanup_after_test(lambda: _restore_registry_baseline(pg_engine))


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
        engine_a = resolved_a.create_engine()
        backend_a = PGVectorEmbeddingBackend(emb_engine=engine_a, resolved=resolved_a)
        assert backend_a is not None

        # A different database_name too: the registry provenance row must
        # be shared correctly across distinct database entries pointed at
        # the same connection, not just across two calls with one name.
        resolved_b = _resolved(
            pg_db, database_name=f"emb_guard_db_{uuid.uuid4().hex[:8]}", schema=schema_b
        )
        engine_b = resolved_b.create_engine()
        backend_b = PGVectorEmbeddingBackend(emb_engine=engine_b, resolved=resolved_b)
        assert backend_b is not None


def test_registry_schema_guard_fires_on_genuine_registry_drift(pg_db, pg_engine, cleanup_after_test):
    # Restored, not deleted, at teardown -- see _restore_registry_baseline.
    cleanup_after_test(lambda: _restore_registry_baseline(pg_engine))
    resolved = _resolved(
        pg_db,
        database_name=f"emb_guard_db_{uuid.uuid4().hex[:8]}",
        schema="unrelated_primary_schema",
    )

    with pg_engine.begin() as connection:
        record_schema_provenance(
            connection,
            database_name=MODEL_REGISTRY_SCHEMA,
            schema_tag=REGISTRY_SCHEMA_KEY,
            new_physical_schema="a_previous_registry_schema_that_is_not_current",
            reason="test: force a stale baseline",
        )

    engine = resolved.create_engine()
    with pytest.raises(SchemaDriftError):
        PGVectorEmbeddingBackend(emb_engine=engine, resolved=resolved)


def test_primary_schema_guard_fires_on_genuine_primary_drift(pg_db, pg_engine, cleanup_after_test):
    """Tests embedding storage table's own guard. Only triggered
    once a model is actually registered, not at backend construction time.
    Restores the PRIMARY schema-tag drift coverage a prior rewrite replaced instead
    of adding alongside"""
    _establish_registry_baseline(pg_db, pg_engine, cleanup_after_test)
    database_name = f"emb_guard_db_{uuid.uuid4().hex[:8]}"
    cleanup_schema_registry_rows(cleanup_after_test, pg_engine, database_name)

    with (
        isolated_test_schema(pg_engine, prefix="emb_guard_primary_a") as schema_a,
        isolated_test_schema(pg_engine, prefix="emb_guard_primary_b") as schema_b,
    ):
        resolved_a = _resolved(pg_db, database_name=database_name, schema=schema_a)
        engine_a = resolved_a.create_engine()
        backend_a = PGVectorEmbeddingBackend(emb_engine=engine_a, resolved=resolved_a)
        backend_a.register_model(
            model_name=MODEL_NAME, provider_type=PROVIDER_TYPE, dimensions=EMBEDDING_DIM
        )

        resolved_b = _resolved(pg_db, database_name=database_name, schema=schema_b)
        engine_b = resolved_b.create_engine()
        # Constructing backend_b already reloads the model registered above
        # (shared registry schema, same database_name) and tries to load its
        # storage table under the new schema, which is where the guard fires
        with pytest.raises(SchemaDriftError):
            PGVectorEmbeddingBackend(emb_engine=engine_b, resolved=resolved_b)
