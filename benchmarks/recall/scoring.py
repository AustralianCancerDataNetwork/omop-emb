"""Pure scoring helpers: recall by distance, latency summaries, merges and estimates."""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence

import numpy as np

K_VALUES = (10, 25)
"""Result sizes recall is reported at."""


def distance_tolerance(distance: float) -> float:
    """Slack allowed on a distance, so float rounding does not count a tie as a miss."""
    return 1e-6 + 1e-6 * abs(distance)


def recall_at_k(
    returned: Sequence[tuple[int, float]], truth: Sequence[tuple[int, float]], k: int
) -> float | None:
    """Share of the exact top k that the returned list reaches, counted by distance.

    A returned concept counts when its exact distance is no larger than the
    exact k-th distance plus a tolerance, so concepts with identical vectors
    are interchangeable.

    Parameters
    ----------
    returned : Sequence[tuple[int, float]]
        Returned ``(concept_id, exact distance)`` pairs, closest first.
    truth : Sequence[tuple[int, float]]
        Exact ``(concept_id, distance)`` pairs, closest first.
    k : int
        Result size.

    Returns
    -------
    float or None
        Recall in ``[0, 1]``, or ``None`` when the exact search found nothing.
    """
    eligible = min(k, len(truth))
    if eligible == 0:
        return None
    cutoff = truth[eligible - 1][1]
    limit = cutoff + distance_tolerance(cutoff)
    found = len({cid for cid, distance in returned[:k] if distance <= limit})
    return min(found, eligible) / eligible


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile of values."""
    return float(np.percentile(values, q))


def latency_summary(milliseconds: Sequence[float]) -> dict[str, float]:
    """p50 and p95 of a list of latencies in milliseconds."""
    return {"p50": percentile(milliseconds, 50), "p95": percentile(milliseconds, 95)}


def merge_top_k(lists: Sequence[Sequence[tuple[int, float]]], k: int) -> list[tuple[int, float]]:
    """Merge lists of ``(concept_id, distance)`` into the k closest distinct concepts."""
    best: dict[int, float] = {}
    for items in lists:
        for concept_id, distance in items:
            if concept_id not in best or distance < best[concept_id]:
                best[concept_id] = distance
    return heapq.nsmallest(k, best.items(), key=lambda item: item[1])


def weighted_mean(values: Mapping[str, float | None], weights: Mapping[str, float]) -> float | None:
    """Mean of the values that exist, weighted by weights and renormalised over them."""
    present = {key: value for key, value in values.items() if value is not None}
    total = sum(weights[key] for key in present)
    if not present or total == 0:
        return None
    return sum(weights[key] * value for key, value in present.items()) / total


def estimate_index_bytes(rows: int, dimensions: int, value_bytes: int, num_neighbors: int) -> int:
    """Rough size of an HNSW index over rows vectors.

    The stored vector plus about 128 bytes of graph per neighbour. This matches
    8.2 KB per row measured at 1,024 float32 dimensions and 32 neighbours.
    """
    return rows * (dimensions * value_bytes + 128 * num_neighbors)


def parse_setting(text: str) -> tuple[int, int | None]:
    """Parse ``EF`` or ``EF:CAP`` into ``ef_search`` and ``max_scan_tuples``."""
    ef_search, _, cap = text.partition(":")
    return int(ef_search), int(cap) if cap else None


def share(flags: Sequence[bool]) -> float:
    """Fraction of True in flags."""
    return sum(flags) / len(flags) if flags else 0.0
