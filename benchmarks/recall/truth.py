"""Exact nearest concepts per query and scenario, computed once and cached."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np

from benchmarks.recall.files import Workspace, write_json
from benchmarks.recall.scenarios import Query, Scenario, scenario_queries
from benchmarks.recall.scoring import K_VALUES
from benchmarks.recall.store import Store

TRUTH_K = max(K_VALUES)


def truth_key(scenario: Scenario, query: Query) -> str:
    return f"{scenario.name}|{query.id}"


def store_fingerprint(store: Store) -> dict[str, Any]:
    """What the exact results depend on, so a cache is not reused for a changed store."""
    return {
        "model": store.record.model_name,
        "metric": store.metric.value,
        "rows": store.row_count(),
        "dimensions": store.record.dimensions,
    }


def load_truth(workspace: Workspace) -> dict[str, Any]:
    if not workspace.truth_path.exists():
        return {"fingerprint": None, "items": {}}
    return json.loads(workspace.truth_path.read_text(encoding="utf-8"))


def compute_truth(
    store: Store,
    workspace: Workspace,
    *,
    save_every: int = 20,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Compute the exact top results that are missing from the cache.

    Returns
    -------
    int
        Number of results computed in this call.

    Raises
    ------
    RuntimeError
        If the cache was computed for a different store.
    """
    plan = workspace.load_plan()
    queries = workspace.queries()
    vectors = workspace.vectors()
    fingerprint = store_fingerprint(store)
    cache = load_truth(workspace)
    if cache["fingerprint"] not in (None, fingerprint):
        raise RuntimeError(
            f"{workspace.truth_path} was computed for another store {cache['fingerprint']}, "
            f"this one is {fingerprint}. Delete it to start again."
        )
    cache["fingerprint"] = fingerprint
    items: dict[str, list[list[float]]] = cache["items"]

    todo = [
        (scenario, query)
        for scenario in workspace.scenarios()
        for query in scenario_queries(scenario, queries, plan["none_per_domain"])
        if truth_key(scenario, query) not in items
    ]
    for done, (scenario, query) in enumerate(todo, start=1):
        results = store.exact(np.asarray(vectors[query.id]), scenario.concept_filter(), TRUTH_K)
        items[truth_key(scenario, query)] = [[cid, distance] for cid, distance in results]
        if done % save_every == 0:
            write_json(workspace.truth_path, cache)
        if progress:
            progress(done, len(todo))
    write_json(workspace.truth_path, cache)
    return len(todo)
