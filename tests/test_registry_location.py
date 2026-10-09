"""ensure_registry_table() refuses to create an empty registry while one exists in another schema."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from oa_configurator import Role
from oa_configurator.testing import isolated_test_schema, scoped_test_schema

from omop_emb.model_registry.model_registry_orm import (
    _reject_incompatible_registry,
    registry_reader_engine,
)
from omop_emb.utils.errors import MisplacedRegistryError

pytestmark = [pytest.mark.postgresql, pytest.mark.db_dialect]


def _create_registry_table(engine: sa.Engine, schema: str) -> None:
    with engine.begin() as connection:
        connection.execute(sa.text(f'CREATE TABLE "{schema}".model_registry (model_name text)'))


def test_registry_left_in_another_schema_is_refused(pg_db) -> None:
    with (
        scoped_test_schema(pg_db.resolved, prefix="emb_misplaced") as scoped,
        isolated_test_schema(pg_db.committing_engine, prefix="emb_registry") as registry_schema,
    ):
        primary_schema = scoped.schemas[Role.PRIMARY]
        _create_registry_table(scoped.engine, primary_schema)
        with scoped.engine.connect() as connection:
            with pytest.raises(MisplacedRegistryError) as exc_info:
                _reject_incompatible_registry(connection, registry_schema=registry_schema)
        assert primary_schema in str(exc_info.value)
        assert "SET SCHEMA" in str(exc_info.value)


def test_registry_in_its_own_schema_passes(pg_db) -> None:
    with (
        scoped_test_schema(pg_db.resolved, prefix="emb_placed") as scoped,
        isolated_test_schema(pg_db.committing_engine, prefix="emb_registry") as registry_schema,
    ):
        _create_registry_table(scoped.engine, scoped.schemas[Role.PRIMARY])
        _create_registry_table(scoped.engine, registry_schema)
        with scoped.engine.connect() as connection:
            _reject_incompatible_registry(connection, registry_schema=registry_schema)


def test_reader_refuses_a_registry_left_in_another_schema(pg_db, monkeypatch) -> None:
    """open_vector_store_reader/registry_reader_engine must fail closed on
    a misplaced registry. 
    
    Notes
    -----
    MODEL_REGISTRY_SCHEMA is monkeypatched to an isolated,
    guaranteed-empty schema so the proper location is genuinely absent for
    this test, regardless of what the shared test database's real
    omop_emb_registry schema already holds from other tests."""
    import omop_emb.model_registry.model_registry_orm as model_registry_orm

    with (
        isolated_test_schema(pg_db.committing_engine, prefix="emb_registry_proper") as proper_schema,
        isolated_test_schema(pg_db.committing_engine, prefix="emb_registry_legacy") as legacy_schema,
    ):
        _create_registry_table(pg_db.committing_engine, legacy_schema)
        monkeypatch.setattr(model_registry_orm, "MODEL_REGISTRY_SCHEMA", proper_schema)

        with pytest.raises(MisplacedRegistryError) as exc_info:
            registry_reader_engine(pg_db.resolved)
        assert legacy_schema in str(exc_info.value)


def test_reader_passes_when_the_registry_is_correctly_placed(pg_db, monkeypatch) -> None:
    import omop_emb.model_registry.model_registry_orm as model_registry_orm

    with isolated_test_schema(pg_db.committing_engine, prefix="emb_registry_proper") as proper_schema:
        _create_registry_table(pg_db.committing_engine, proper_schema)
        monkeypatch.setattr(model_registry_orm, "MODEL_REGISTRY_SCHEMA", proper_schema)

        registry_reader_engine(pg_db.resolved).dispose()  # must not raise
