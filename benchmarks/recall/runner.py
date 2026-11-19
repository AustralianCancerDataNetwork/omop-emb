"""Run the scenarios against the current index state and score them."""

from __future__ import annotations

import random
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy import event

from benchmarks.recall import profile
from benchmarks.recall.files import Workspace
from benchmarks.recall.plans import plan_labels
from benchmarks.recall.scenarios import Query, Scenario, scenario_queries
from benchmarks.recall.scoring import K_VALUES, latency_summary, merge_top_k, recall_at_k, share
from benchmarks.recall.store import Setting, Store
from benchmarks.recall.truth import TRUTH_K, load_truth, truth_key

COMBINED = "combined"
SPLIT_SEQUENTIAL = "split-sequential"
SPLIT_PARALLEL = "split-parallel"
SHORT_RECALL = 0.8


@dataclass(frozen=True)
class Item:
    scenario: Scenario
    query: Query
    arm: str


@dataclass
class Cell:
    """What one scenario, arm and setting produced."""

    latencies: list[float] = field(default_factory=list)
    results: dict[int, list[tuple[int, float]]] = field(default_factory=dict)


def run_item(
    store: Store,
    vector: np.ndarray,
    item: Item,
    setting: Setting,
    pool: ThreadPoolExecutor | None,
) -> tuple[list[tuple[int, float]], float]:
    """Run one query by the arm of item and return its results and milliseconds."""
    scenario = item.scenario
    if item.arm == COMBINED:
        return store.search(vector, scenario.concept_filter(), TRUTH_K, setting)
    legs = [scenario.concept_filter((domain,)) for domain in scenario.domains]
    if item.arm == SPLIT_SEQUENTIAL:
        outcomes = [store.search(vector, leg, TRUTH_K, setting) for leg in legs]
        return merge_top_k([o[0] for o in outcomes], TRUTH_K), sum(o[1] for o in outcomes)
    assert pool is not None
    started = time.perf_counter()
    outcomes = list(pool.map(lambda leg: store.search(vector, leg, TRUTH_K, setting), legs))
    elapsed = (time.perf_counter() - started) * 1000
    return merge_top_k([o[0] for o in outcomes], TRUTH_K), elapsed


def build_items(
    scenarios: Sequence[Scenario],
    queries: Sequence[Query],
    none_per_domain: int,
    split_arms: bool,
) -> list[Item]:
    """Every query of every scenario, with the split arms added to two-domain scenarios."""
    items = []
    for scenario in scenarios:
        arms = [COMBINED]
        if split_arms and len(scenario.domains) == 2:
            arms += [SPLIT_SEQUENTIAL, SPLIT_PARALLEL]
        for query in scenario_queries(scenario, queries, none_per_domain):
            items += [Item(scenario, query, arm) for arm in arms]
    return items


def capture_plans(
    store: Store,
    vectors: np.ndarray,
    items: Sequence[Item],
    setting: Setting,
    domains: list[str],
    per_cell: int,
) -> dict[str, dict[str, dict[str, int]]]:
    """Index labels the planner uses for the first queries of every scenario, by arm."""
    captured: list[tuple[str, Any]] = []

    def capture(_conn, _cursor, statement, parameters, _context, _many) -> None:
        if "order by" in statement.lower():
            captured.append((statement, parameters))

    taken: Counter[tuple[str, str]] = Counter()
    plans: dict[str, dict[str, Counter[str]]] = {}
    for item in items:
        if item.arm == SPLIT_PARALLEL:
            continue
        cell = (item.scenario.name, item.arm)
        if taken[cell] >= per_cell:
            continue
        taken[cell] += 1
        captured.clear()
        event.listen(store.engine, "before_cursor_execute", capture)
        try:
            run_item(store, vectors[item.query.id], item, setting, None)
        finally:
            event.remove(store.engine, "before_cursor_execute", capture)
        counts = plan_labels(
            store.engine,
            list(captured),
            table=store.record.storage_identifier,
            metric=store.metric,
            domains=domains,
        )
        merged = plans.setdefault(
            f"{cell[0]}|{cell[1]}", {"custom": Counter(), "generic": Counter()}
        )
        for kind, counter in counts.items():
            merged[kind].update(counter)
    return {
        key: {kind: dict(counter) for kind, counter in kinds.items()}
        for key, kinds in plans.items()
    }


def score_cell(
    store: Store,
    vectors: np.ndarray,
    truth: dict[str, list[list[float]]],
    scenario: Scenario,
    cell: Cell,
) -> dict[str, Any]:
    """Recall, shortfall and latency of one cell against the exact results."""
    recalls: dict[int, list[float]] = {k: [] for k in K_VALUES}
    short: list[bool] = []
    zero: list[bool] = []
    for query_id, results in cell.results.items():
        exact_top = [(int(c), float(d)) for c, d in truth[f"{scenario.name}|{query_id}"]]
        distances = store.exact_distances(vectors[query_id], [cid for cid, _ in results])
        rescored = [(cid, distances.get(cid, float("inf"))) for cid, _ in results]
        for k in K_VALUES:
            value = recall_at_k(rescored, exact_top, k)
            if value is not None:
                recalls[k].append(value)
        expected = min(TRUTH_K, len(exact_top))
        short.append(len(results) < expected)
        zero.append(not results and expected > 0)
    scored: dict[str, Any] = {"n": len(cell.results), "latency_ms": latency_summary(cell.latencies)}
    for k in K_VALUES:
        scored[f"recall@{k}"] = {
            "mean": float(np.mean(recalls[k])) if recalls[k] else None,
            f"below_{SHORT_RECALL}": share([r < SHORT_RECALL for r in recalls[k]]),
        }
    scored["short_share"] = share(short)
    scored["zero_share"] = share(zero)
    return scored


def run_benchmark(
    store: Store,
    workspace: Workspace,
    settings: Sequence[Setting],
    *,
    label: str,
    passes: int,
    seed: int,
    plan_queries: int,
    split_arms: bool,
    progress: Callable[[str], None] = lambda _message: None,
) -> dict[str, Any]:
    """Measure the current index state and return the result document.

    Every setting runs one warm pass whose timings are dropped, then ``passes``
    timed passes over all queries in a random order. Recall is scored on the
    results of the first timed pass.

    Raises
    ------
    RuntimeError
        If the ground truth of a scenario is missing.
    """
    plan = workspace.load_plan()
    queries, vectors = workspace.queries(), workspace.vectors()
    truth = load_truth(workspace)["items"]
    scenarios = workspace.scenarios()
    partials = [
        entry
        for entry in profile.index_sizes(store, plan["domains"])
        if entry["serves"].startswith("partial:")
    ]
    items = build_items(scenarios, queries, plan["none_per_domain"], split_arms and bool(partials))
    missing = [
        truth_key(item.scenario, item.query)
        for item in items
        if truth_key(item.scenario, item.query) not in truth
    ]
    if missing:
        raise RuntimeError(f"{len(missing)} exact results are missing. Run 'truth' first.")

    rng = random.Random(seed)
    cells: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for setting in settings:
            accumulated: dict[tuple[str, str], Cell] = {}
            for pass_index in range(passes + 1):
                order = list(items)
                rng.shuffle(order)
                for item in order:
                    results, milliseconds = run_item(
                        store, vectors[item.query.id], item, setting, pool
                    )
                    if pass_index == 0:
                        continue
                    cell = accumulated.setdefault((item.scenario.name, item.arm), Cell())
                    cell.latencies.append(milliseconds)
                    if pass_index == 1:
                        cell.results[item.query.id] = results
                progress(f"{setting.name}: pass {pass_index} of {passes} done")
            for scenario in scenarios:
                for arm in sorted({i.arm for i in items if i.scenario == scenario}):
                    cell = accumulated.get((scenario.name, arm))
                    if cell is None:
                        continue
                    cells.append(
                        {
                            "setting": setting.name,
                            "scenario": scenario.name,
                            "scope": scenario.scope,
                            "profile": scenario.profile,
                            "arm": arm,
                            **score_cell(store, vectors, truth, scenario, cell),
                        }
                    )
    return {
        "label": label,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "meta": {
            **profile.server_info(store),
            "model": store.record.model_name,
            "dimensions": store.record.dimensions,
            "metric": store.metric.value,
            "storage_precision": store.storage_precision.value,
            "index_config": store.record.index_config.to_dict(),
            "rows": store.row_count(),
            "table_bytes": profile.table_bytes(store),
            "indexes": profile.index_sizes(store, plan["domains"]),
            "registry_uses_partials": store.inlines_single_domain,
            "seed": plan["seed"],
            "passes": passes,
            "none_per_domain": plan["none_per_domain"],
            "split_arms": split_arms and bool(partials),
        },
        "settings": [s.name for s in settings],
        "cells": cells,
        "plans": capture_plans(store, vectors, items, settings[0], plan["domains"], plan_queries),
    }
