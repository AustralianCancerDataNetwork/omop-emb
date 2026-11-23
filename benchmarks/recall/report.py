"""Compare labelled runs as markdown tables."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from benchmarks.recall.files import Workspace
from benchmarks.recall.runner import COMBINED, SHORT_RECALL, SPLIT_PARALLEL, SPLIT_SEQUENTIAL
from benchmarks.recall.scenarios import NONE_SCOPE
from benchmarks.recall.scoring import weighted_mean

RECALL_TARGET = 0.95
"""Recall at 10 every cell should reach at the reference setting."""

REFERENCE_SETTING = "ef200/capdefault"
FAN_OUT_P95_FACTOR = 2.0
"""Largest p95 growth over the combined filter at which a split arm is acceptable."""

DEFAULT_WEIGHTS = {"single": 0.835, "none": 0.165}


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _cell(value: str) -> str:
    return value.replace("|", "/")


def _gib(value: int) -> str:
    return f"{value / 2**30:.2f} GiB"


def _recall(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _cell_text(cell: Mapping[str, Any]) -> str:
    latency = cell["latency_ms"]
    return (
        f"R@10 {_recall(cell['recall@10']['mean'])} · R@25 {_recall(cell['recall@25']['mean'])} · "
        f"<{SHORT_RECALL} {cell['recall@10'][f'below_{SHORT_RECALL}']:.0%} · "
        f"short {cell['short_share']:.0%} · zero {cell['zero_share']:.0%} · "
        f"{latency['p50']:.1f}/{latency['p95']:.1f} ms"
    )


def load_runs(workspace: Workspace, labels: Sequence[str] | None) -> list[dict[str, Any]]:
    """The result documents of labels, or of every run in the work directory."""
    if labels:
        paths = [workspace.run_path(label) for label in labels]
    else:
        paths = sorted((workspace.root / "runs").glob("*.json"))
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not runs:
        raise RuntimeError(f"No runs found in {workspace.root / 'runs'}.")
    return runs


def weighted_recall(
    cells: Sequence[Mapping[str, Any]], setting: str, profile: str, weights: Mapping[str, float]
) -> float | None:
    """Recall at 10 over the expected workload of single-domain and no-domain searches."""
    chosen = [
        c
        for c in cells
        if c["setting"] == setting and c["profile"] == profile and c["arm"] == COMBINED
    ]
    single = [
        c["recall@10"]["mean"] for c in chosen if c["scope"] != NONE_SCOPE and "+" not in c["scope"]
    ]
    single = [value for value in single if value is not None]
    none = [c["recall@10"]["mean"] for c in chosen if c["scope"] == NONE_SCOPE]
    return weighted_mean(
        {
            "single": sum(single) / len(single) if single else None,
            "none": none[0] if none else None,
        },
        weights,
    )


def _index_states(runs: Sequence[Mapping[str, Any]], builds: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for run in runs:
        meta = run["meta"]
        indexes = meta["indexes"]
        partial = [i for i in indexes if i["serves"].startswith("partial:")]
        built = [b for b in builds if b["label"] == run["label"]]
        seconds = sum(i["seconds"] for b in built for i in b["indexes"])
        rows.append(
            [
                run["label"],
                _gib(sum(i["bytes"] for i in indexes if i["serves"] == "global")),
                f"{len(partial)} ({_gib(sum(i['bytes'] for i in partial))})",
                f"{seconds:.0f} s" if built else "n/a",
                str(meta["index_config"].get("ef_construction")),
                meta["index_config"].get("index_precision") or meta["storage_precision"],
                meta["settings"]["shared_buffers"],
                meta["settings"]["maintenance_work_mem"],
            ]
        )
    return _table(
        [
            "Run",
            "Global index",
            "Partial indexes",
            "Partial build",
            "ef_construction",
            "Index precision",
            "shared_buffers",
            "maintenance_work_mem",
        ],
        rows,
    )


def _setting_tables(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    sections = []
    settings = list(dict.fromkeys(s for run in runs for s in run["settings"]))
    profiles = list(dict.fromkeys(c["profile"] for run in runs for c in run["cells"]))
    for setting in settings:
        for profile in profiles:
            keys = list(
                dict.fromkeys(
                    (c["scope"], c["arm"])
                    for run in runs
                    for c in run["cells"]
                    if c["setting"] == setting and c["profile"] == profile
                )
            )
            rows = []
            for scope, arm in keys:
                row = [scope if arm == COMBINED else f"{scope} ({arm})"]
                for run in runs:
                    match = [
                        c
                        for c in run["cells"]
                        if (c["setting"], c["profile"], c["scope"], c["arm"])
                        == (setting, profile, scope, arm)
                    ]
                    row.append(_cell_text(match[0]) if match else "")
                rows.append(row)
            if rows:
                sections.append(
                    f"### {setting}, {profile}\n\n"
                    + _table(["Scope", *[run["label"] for run in runs]], rows)
                )
    return sections


def _workload(runs: Sequence[Mapping[str, Any]], weights: Mapping[str, float]) -> str:
    settings = list(dict.fromkeys(s for run in runs for s in run["settings"]))
    profiles = list(dict.fromkeys(c["profile"] for run in runs for c in run["cells"]))
    rows = [
        [
            setting,
            profile,
            *[_recall(weighted_recall(r["cells"], setting, profile, weights)) for r in runs],
        ]
        for setting in settings
        for profile in profiles
    ]
    return _table(["Setting", "Profile", *[run["label"] for run in runs]], rows)


def _plans(runs: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for run in runs:
        for key, kinds in run["plans"].items():
            rows.append(
                [
                    run["label"],
                    _cell(key),
                    ", ".join(f"{k} x{n}" for k, n in sorted(kinds["custom"].items())),
                    ", ".join(f"{k} x{n}" for k, n in sorted(kinds["generic"].items())),
                ]
            )
    return _table(["Run", "Scenario|arm", "Index, one-off plan", "Index, generic plan"], rows)


def _acceptance(runs: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for run in runs:
        failing = [
            f"{c['scenario']} ({_recall(c['recall@10']['mean'])})"
            for c in run["cells"]
            if c["setting"] == REFERENCE_SETTING
            and c["arm"] == COMBINED
            and "+" not in c["scope"]
            and (c["recall@10"]["mean"] or 0) < RECALL_TARGET
        ]
        if REFERENCE_SETTING not in run["settings"]:
            lines.append(f"- {run['label']}: no run at {REFERENCE_SETTING}")
        elif failing:
            lines.append(f"- {run['label']}: below {RECALL_TARGET} in " + ", ".join(failing))
        else:
            lines.append(
                f"- {run['label']}: every single-domain and no-domain cell reaches {RECALL_TARGET}"
            )
    return "\n".join(lines)


def _fan_out(runs: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for run in runs:
        for combined in run["cells"]:
            if (
                combined["arm"] != COMBINED
                or "+" not in combined["scope"]
                or combined["setting"] != REFERENCE_SETTING
            ):
                continue
            for arm in (SPLIT_SEQUENTIAL, SPLIT_PARALLEL):
                split = next(
                    (
                        c
                        for c in run["cells"]
                        if c["scenario"] == combined["scenario"]
                        and c["setting"] == REFERENCE_SETTING
                        and c["arm"] == arm
                    ),
                    None,
                )
                if split is None:
                    continue
                before, after = combined["recall@10"]["mean"], split["recall@10"]["mean"]
                factor = split["latency_ms"]["p95"] / max(combined["latency_ms"]["p95"], 1e-9)
                meets = (
                    before is not None
                    and after is not None
                    and before < RECALL_TARGET <= after
                    and factor <= FAN_OUT_P95_FACTOR
                )
                rows.append(
                    [
                        run["label"],
                        _cell(combined["scenario"]),
                        arm,
                        f"{_recall(before)} to {_recall(after)}",
                        f"{factor:.1f}x",
                        "yes" if meets else "no",
                    ]
                )
    if not rows:
        return "No split arms were run."
    return _table(
        ["Run", "Scenario", "Arm", "Recall@10, combined to split", "p95 growth", "Meets the rule"],
        rows,
    )


def compare(
    workspace: Workspace, labels: Sequence[str] | None, weights: Mapping[str, float] | None = None
) -> str:
    """The comparison of runs as markdown."""
    runs = load_runs(workspace, labels)
    weights = weights or DEFAULT_WEIGHTS
    sections = [
        "# Recall benchmark comparison",
        "## Index states\n\n" + _index_states(runs, workspace.read_builds()),
        "## Expected workload, recall@10\n\n"
        f"Weights {dict(weights)}: single-domain searches against searches without a domain.\n\n"
        + _workload(runs, weights),
        f"## Reference setting {REFERENCE_SETTING}, recall@10 at least {RECALL_TARGET}\n\n"
        + _acceptance(runs),
        "## Index chosen by the planner\n\n" + _plans(runs),
        f"## Fan-out\n\nA split arm meets the rule when the combined filter is below {RECALL_TARGET} "
        f"recall@10, the split reaches it, and its p95 is at most {FAN_OUT_P95_FACTOR:g}x the combined filter's.\n\n"
        + _fan_out(runs),
        (
            "## Results\n\nEach cell is recall@10, recall@25, the share of queries below "
            f"{SHORT_RECALL} recall@10, the share returning fewer than 25 results, "
            "the share returning none, and p50/p95 latency."
        ),
        *_setting_tables(runs),
    ]
    return "\n\n".join(sections) + "\n"
