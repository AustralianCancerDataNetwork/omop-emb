"""in_values() sends a value list as one bind parameter on both dialects."""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from omop_emb.backends.db_utils import in_values

from .conftest import sqlite_resolved_database

# Above PostgreSQL's 65,535 bind-parameter cap and SQLite's default 32,766.
_MANY = 70_000

_metadata = sa.MetaData()
_table = sa.Table(
    "in_values_probe",
    _metadata,
    sa.Column("concept_id", sa.Integer, primary_key=True),
    sa.Column("domain_id", sa.String),
)


def _count(engine: sa.Engine, condition) -> int:
    with engine.connect() as connection:
        return connection.execute(sa.select(sa.func.count()).select_from(_table).where(condition)).scalar_one()


def _check(engine: sa.Engine, dialect: str) -> None:
    statements: list[tuple[str, object]] = []

    def capture(_connection, _cursor, statement, parameters, _context, _many):
        statements.append((statement, parameters))

    sa.event.listen(engine, "before_cursor_execute", capture)
    try:
        assert _count(engine, in_values(_table.c.concept_id, list(range(_MANY)), dialect=dialect)) == 1000
    finally:
        sa.event.remove(engine, "before_cursor_execute", capture)
    _, parameters = statements[-1]
    assert len(parameters) == 1

    assert _count(engine, in_values(_table.c.concept_id, [1, 1, 2], dialect=dialect)) == 2
    assert _count(engine, in_values(_table.c.domain_id, ["Drug"], dialect=dialect)) == 500
    assert _count(engine, in_values(_table.c.concept_id, [], dialect=dialect)) == 0


def _populate(engine: sa.Engine) -> None:
    _metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            sa.insert(_table),
            [{"concept_id": i, "domain_id": "Drug" if i % 2 else "Condition"} for i in range(1000)],
        )


def test_sqlite_sends_one_parameter_past_the_bind_limit():
    engine = sqlite_resolved_database().create_engine()
    _populate(engine)
    _check(engine, "sqlite")


@pytest.mark.postgresql
@pytest.mark.db_dialect
def test_postgres_sends_one_parameter_past_the_bind_limit(pg_db):
    engine = pg_db.committing_engine
    _populate(engine)
    try:
        _check(engine, "postgresql")
    finally:
        _metadata.drop_all(engine)
