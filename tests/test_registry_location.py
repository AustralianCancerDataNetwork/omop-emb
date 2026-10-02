"""ensure_registry_table() refuses to create an empty registry while one exists in another schema."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from oa_configurator import Role
from oa_configurator.testing import isolated_test_schema, scoped_test_schema

from omop_emb.model_registry.model_registry_orm import _reject_misplaced_registry
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
                _reject_misplaced_registry(connection, registry_schema=registry_schema)
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
            _reject_misplaced_registry(connection, registry_schema=registry_schema)
