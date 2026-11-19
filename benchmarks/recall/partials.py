"""Build and drop the per-domain partial indexes of a model, leaving the global index alone."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from oa_configurator import Role, guard_schema_provenance_for, physical_schema_of, qualified
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateIndex

from benchmarks.recall.scoring import estimate_index_bytes
from benchmarks.recall.store import Store
from omop_emb.backends.index_naming import IndexNames
from omop_emb.backends.pgvector.pg_sql import partial_hnsw_index
from omop_emb.config import VectorPrecision


def _require_hnsw(store: Store):
    config = store.index_config
    if config is None:
        raise RuntimeError("The model has no HNSW index. Build it first with 'rebuild-index'.")
    return config


def partial_index_names(store: Store) -> list[str]:
    """Names of the partial indexes that exist for the model."""
    table = store.record.storage_identifier
    global_name = IndexNames.hnsw(table, store.metric)
    return [
        name
        for name in store.backend.physical_indexes(store.record.model_name)
        if name != global_name
    ]


def plan_build(
    store: Store, *, valid_only: bool, domains: list[str] | None
) -> list[dict[str, Any]]:
    """Rows and estimated index size per domain a build would cover, and the memory available.

    Each entry has ``domain``, ``rows``, ``estimated_bytes`` and ``fits`` against
    ``maintenance_work_mem``.
    """
    config = _require_hnsw(store)
    table = store.table
    statement = select(table.domain_id, func.count()).group_by(table.domain_id)
    if valid_only:
        statement = statement.where(table.is_valid.is_(True))
    with Session(store.engine) as session:
        counts = session.execute(statement).all()
        work_mem = int(
            session.execute(
                text("SELECT pg_size_bytes(current_setting('maintenance_work_mem'))")
            ).scalar_one()
        )
    value_bytes = (
        2 if config.effective_precision(store.storage_precision) == VectorPrecision.FLOAT16 else 4
    )
    return [
        {
            "domain": str(domain),
            "rows": int(rows),
            "estimated_bytes": (
                estimate := estimate_index_bytes(
                    int(rows), store.record.dimensions, value_bytes, config.num_neighbors
                )
            ),
            "fits": estimate <= work_mem,
            "maintenance_work_mem": work_mem,
        }
        for domain, rows in sorted(counts, key=lambda item: -item[1])
        if domains is None or domain in domains
    ]


def _size(store: Store, name: str) -> int:
    reference = qualified(store.engine, name, physical_schema=physical_schema_of(store.engine))
    with Session(store.engine) as session:
        return int(
            session.execute(
                text("SELECT pg_relation_size(CAST(:n AS regclass))"), {"n": reference}
            ).scalar_one()
        )


def drop_partials(store: Store) -> list[str]:
    """Drop the partial indexes and clear their settings in the registry."""
    config = _require_hnsw(store)
    names = partial_index_names(store)
    with store.engine.begin() as conn, guard_schema_provenance_for(conn, schema_tag=Role.PRIMARY):
        for name in names:
            reference = qualified(
                store.engine, name, physical_schema=physical_schema_of(store.engine)
            )
            conn.execute(text(f"DROP INDEX IF EXISTS {reference}"))
    store.backend._registry.update_index_config(
        model_name=store.record.model_name,
        index_config=replace(config, partial_domain_indexes=False, partial_valid_only=False),
    )
    return names


def build_partials(
    store: Store, *, valid_only: bool, domains: list[str] | None, analyze: bool = True
) -> dict[str, Any]:
    """Replace the partial indexes with one per domain, timing and sizing each.

    The registry then records the partial index settings, so queries name a
    single domain in a way the planner can match to its index.
    """
    config = _require_hnsw(store)
    planned = plan_build(store, valid_only=valid_only, domains=domains)
    new_config = replace(config, partial_domain_indexes=True, partial_valid_only=valid_only)
    indexes = {
        entry["domain"]: partial_hnsw_index(
            store.record, new_config, store.storage_precision, entry["domain"]
        )
        for entry in planned
    }
    names = {
        entry["domain"]: IndexNames.partial(store.record.storage_identifier, entry["domain"])
        for entry in planned
    }
    if len(set(names.values())) < len(names):
        raise ValueError("Two domains share a partial index name.")
    drop_partials(store)
    built = []
    for entry in planned:
        started = time.perf_counter()
        with (
            store.engine.begin() as conn,
            guard_schema_provenance_for(conn, schema_tag=Role.PRIMARY),
        ):
            conn.execute(CreateIndex(indexes[entry["domain"]]))
        built.append(
            {
                "domain": entry["domain"],
                "rows": entry["rows"],
                "name": names[entry["domain"]],
                "seconds": time.perf_counter() - started,
                "bytes": _size(store, names[entry["domain"]]),
            }
        )
    store.backend._registry.update_index_config(
        model_name=store.record.model_name, index_config=new_config
    )
    if analyze:
        store.backend._vacuum_analyze(store.record)
    return {"valid_only": valid_only, "indexes": built}
