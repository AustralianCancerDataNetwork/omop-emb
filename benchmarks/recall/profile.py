"""What the store holds and how it is configured: counts, index sizes and server settings."""

from __future__ import annotations

from typing import Any

from sqlalchemy import and_, distinct, func, select, text
from sqlalchemy.orm import Session

from benchmarks.recall.plans import label_index
from benchmarks.recall.store import Store
from omop_emb.backends.embedding_table import EMBEDDING_COLUMN_NAME

_SETTINGS = (
    "shared_buffers",
    "maintenance_work_mem",
    "work_mem",
    "effective_cache_size",
    "max_parallel_workers_per_gather",
    "max_parallel_maintenance_workers",
)


def server_info(store: Store) -> dict[str, Any]:
    """PostgreSQL and pgvector versions and the settings that decide index and query speed."""
    with Session(store.engine) as session:
        session.execute(text("SET LOCAL transaction_read_only = on"))
        version = session.execute(text("SELECT version()")).scalar_one()
        pgvector = session.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one_or_none()
        settings = {
            name: session.execute(text("SELECT current_setting(:n)"), {"n": name}).scalar_one()
            for name in _SETTINGS
        }
    return {"postgresql": version, "pgvector": pgvector, "settings": settings}


def index_sizes(store: Store, domains: list[str]) -> list[dict[str, Any]]:
    """Every index on the model's table with its size in bytes and what it serves."""
    table = store.record.storage_identifier
    with Session(store.engine) as session:
        session.execute(text("SET LOCAL transaction_read_only = on"))
        rows = session.execute(
            text(
                "SELECT c.relname, pg_relation_size(c.oid) FROM pg_index i "
                "JOIN pg_class c ON c.oid = i.indexrelid JOIN pg_class t ON t.oid = i.indrelid "
                "WHERE t.relname = :table ORDER BY c.relname"
            ),
            {"table": table},
        ).all()
    return [
        {
            "name": name,
            "serves": label_index(name, table, store.metric, domains),
            "bytes": int(size),
        }
        for name, size in rows
    ]


def table_bytes(store: Store) -> int:
    """Size of the embedding table with its TOAST data, without indexes."""
    with Session(store.engine) as session:
        session.execute(text("SET LOCAL transaction_read_only = on"))
        return int(
            session.execute(
                text("SELECT pg_table_size(CAST(:t AS regclass))"),
                {"t": _qualified(store)},
            ).scalar_one()
        )


def _qualified(store: Store) -> str:
    from oa_configurator import physical_schema_of, qualified

    return qualified(
        store.engine,
        store.record.storage_identifier,
        physical_schema=physical_schema_of(store.engine),
    )


def domain_counts(store: Store) -> list[dict[str, Any]]:
    """Rows per domain, with how many are valid, standard, or both."""
    table = store.table
    valid, standard = table.is_valid.is_(True), table.is_standard.is_(True)
    statement = (
        select(
            table.domain_id,
            func.count().label("rows"),
            func.count().filter(valid).label("valid"),
            func.count().filter(standard).label("standard"),
            func.count().filter(and_(valid, standard)).label("valid_standard"),
        )
        .group_by(table.domain_id)
        .order_by(func.count().desc())
    )
    with Session(store.engine) as session:
        session.execute(text("SET LOCAL transaction_read_only = on"))
        return [dict(row._mapping) for row in session.execute(statement)]


def duplicate_vectors(store: Store) -> int:
    """Rows whose vector equals another row's. Scans and sorts every vector, so it is slow."""
    column = getattr(store.table, EMBEDDING_COLUMN_NAME)
    with Session(store.engine) as session:
        session.execute(text("SET LOCAL transaction_read_only = on"))
        return int(
            session.execute(select(func.count() - func.count(distinct(column)))).scalar_one()
        )
