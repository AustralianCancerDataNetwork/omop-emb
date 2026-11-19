"""Command line of the recall benchmark: ``python -m benchmarks.recall``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from benchmarks.recall import partials, profile, report
from benchmarks.recall.files import Workspace, write_json
from benchmarks.recall.queries import embed_queries, read_queries, sample_queries
from benchmarks.recall.runner import run_benchmark
from benchmarks.recall.scenarios import PROFILES, parse_pair
from benchmarks.recall.scoring import parse_setting
from benchmarks.recall.store import Setting, open_store, resolve_model_backend
from benchmarks.recall.truth import compute_truth
from omop_emb.config import open_cdm_sessions

app = typer.Typer(
    help="Recall and latency of pgvector HNSW indexes, global and per domain. "
    "Read the README in benchmarks/recall for the order of the commands.",
    no_args_is_help=True,
)

DEFAULT_DOMAINS = ["Condition", "Drug", "Measurement", "Observation", "Procedure"]
DEFAULT_PAIRS = ["Measurement+Observation", "Measurement+Procedure", "Condition+Drug"]
DEFAULT_PROFILES = ["any", "active", "active+standard"]
DEFAULT_SETTINGS = ["100", "200", "400", "200:100000"]

WorkDir = Annotated[
    Path, typer.Option("--work-dir", help="Directory for the queries, results and reports.")
]
ModelName = Annotated[
    str | None,
    typer.Option(
        "--model-name",
        "-m",
        help="Name of a model entry in the configuration. Defaults to the configured model.",
    ),
]


def _workspace(work_dir: Path) -> Workspace:
    return Workspace(work_dir)


@app.command()
def queries(
    work_dir: WorkDir = Path("recall_benchmark"),
    model_name: ModelName = None,
    per_domain: Annotated[int, typer.Option(help="Queries sampled per domain.")] = 25,
    seed: Annotated[str, typer.Option(help="Seed of the sample.")] = "recall-benchmark-v1",
    domains: Annotated[
        list[str] | None, typer.Option("--domain", help="Domain to sample. Repeat for more.")
    ] = None,
    pairs: Annotated[
        list[str] | None,
        typer.Option("--pair", help="Two domains as A+B searched together. Repeat for more."),
    ] = None,
    profiles: Annotated[
        list[str] | None,
        typer.Option(
            "--profile", help=f"Filter profile, one of {list(PROFILES)}. Repeat for more."
        ),
    ] = None,
    none_per_domain: Annotated[
        int,
        typer.Option(
            help="Queries per domain used for searches without a domain, whose exact search is slow."
        ),
    ] = 5,
    queries_file: Annotated[
        Path | None,
        typer.Option(
            "--queries", help="JSON list of {q or text, domain} to use instead of sampling."
        ),
    ] = None,
    vectors_file: Annotated[
        Path | None,
        typer.Option("--vectors", help=".npy of one vector per query, to skip embedding."),
    ] = None,
    batch_size: Annotated[int, typer.Option(help="Batch size for embedding.")] = 32,
) -> None:
    """Choose the queries and scenarios, and embed the queries. Run this first."""
    domains = domains or DEFAULT_DOMAINS
    pairs = pairs or DEFAULT_PAIRS
    profiles = profiles or DEFAULT_PROFILES
    for pair in pairs:
        if not set(parse_pair(pair)) <= set(domains):
            raise typer.BadParameter(f"Pair {pair} names a domain that is not sampled: {domains}.")
    if queries_file is not None:
        chosen = read_queries(queries_file)
    else:
        with open_cdm_sessions() as cdm_sessions:
            chosen = sample_queries(cdm_sessions, domains, per_domain, seed)
    if vectors_file is not None:
        vectors = np.load(vectors_file).astype(np.float32)
        if len(vectors) != len(chosen):
            raise typer.BadParameter("--vectors needs one vector per query.")
    else:
        vectors = embed_queries(resolve_model_backend(model_name), chosen, batch_size)
    _workspace(work_dir).save_plan(
        queries=chosen,
        vectors=vectors,
        seed=seed,
        per_domain=per_domain,
        domains=domains,
        pairs=pairs,
        profiles=profiles,
        none_per_domain=none_per_domain,
    )
    counts = {d: sum(q.domain == d for q in chosen) for d in domains}
    typer.echo(f"Saved {len(chosen)} queries to {work_dir}: {counts}")


@app.command(name="profile")
def profile_store(
    work_dir: WorkDir = Path("recall_benchmark"),
    model_name: ModelName = None,
    duplicates: Annotated[
        bool, typer.Option(help="Also count rows that share a vector. Scans every vector.")
    ] = False,
) -> None:
    """Count rows per domain, list the indexes and show the server settings. Read-only."""
    store = open_store(model_name)
    domains = [row["domain_id"] for row in profile.domain_counts(store)]
    result = {
        "server": profile.server_info(store),
        "rows": store.row_count(),
        "table_bytes": profile.table_bytes(store),
        "domains": profile.domain_counts(store),
        "indexes": profile.index_sizes(store, domains),
        "index_config": store.record.index_config.to_dict(),
    }
    if duplicates:
        result["duplicate_vectors"] = profile.duplicate_vectors(store)
    write_json(work_dir / "profile.json", result)
    typer.echo(json.dumps(result, indent=2))


@app.command()
def truth(
    work_dir: WorkDir = Path("recall_benchmark"),
    model_name: ModelName = None,
) -> None:
    """Compute the exact results the runs are scored against. Read-only, slow, resumable."""
    store = open_store(model_name)
    computed = compute_truth(
        store,
        _workspace(work_dir),
        progress=lambda done, total: (
            typer.echo(f"exact results {done} of {total}") if done % 10 == 0 else None
        ),
    )
    typer.echo(f"Computed {computed} exact results.")


@app.command()
def run(
    label: Annotated[str, typer.Option(help="Name of this index state, for example 'global-64'.")],
    work_dir: WorkDir = Path("recall_benchmark"),
    model_name: ModelName = None,
    settings: Annotated[
        list[str] | None,
        typer.Option(
            "--setting", help="ef_search or ef_search:max_scan_tuples to measure. Repeat for more."
        ),
    ] = None,
    passes: Annotated[int, typer.Option(help="Timed passes after one warm pass.")] = 3,
    plan_queries: Annotated[
        int, typer.Option(help="Queries per scenario whose plan is captured.")
    ] = 2,
    split_arms: Annotated[
        bool, typer.Option(help="Also run the two-domain split arms when partial indexes exist.")
    ] = True,
    seed: Annotated[int, typer.Option(help="Seed of the query order.")] = 0,
) -> None:
    """Measure recall and latency of the current index state. Read-only."""
    store = open_store(model_name)
    parsed = [Setting(*parse_setting(text)) for text in (settings or DEFAULT_SETTINGS)]
    existing_partials = bool(partials.partial_index_names(store))
    if existing_partials != store.inlines_single_domain:
        typer.echo(
            "Warning: the registry "
            f"{'has' if store.inlines_single_domain else 'does not have'} partial indexes enabled "
            f"but the database {'has' if existing_partials else 'has no'} partial indexes. "
            "Queries will not use them as a deployment would."
        )
    result = run_benchmark(
        store,
        _workspace(work_dir),
        parsed,
        label=label,
        passes=passes,
        seed=seed,
        plan_queries=plan_queries,
        split_arms=split_arms,
        progress=typer.echo,
    )
    path = _workspace(work_dir).run_path(label)
    write_json(path, result)
    typer.echo(f"Wrote {path}")


@app.command()
def build(
    label: Annotated[str, typer.Option(help="Name recorded with the build, for the comparison.")],
    work_dir: WorkDir = Path("recall_benchmark"),
    model_name: ModelName = None,
    valid_only: Annotated[
        bool, typer.Option("--valid-only", help="Index valid concepts only.")
    ] = False,
    domains: Annotated[
        list[str] | None,
        typer.Option("--domain", help="Build only this domain. Repeat for more."),
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Print the estimate and stop.")] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Build without asking.")] = False,
    analyze: Annotated[bool, typer.Option(help="Run VACUUM ANALYZE afterwards.")] = True,
) -> None:
    """Replace the partial indexes with one per domain. Writes to the store, leaves the global index."""
    store = open_store(model_name, writable=True)
    planned = partials.plan_build(store, valid_only=valid_only, domains=domains)
    for entry in planned:
        typer.echo(
            f"{entry['domain']}: {entry['rows']:,} rows, about "
            f"{entry['estimated_bytes'] / 2**30:.2f} GiB "
            f"({'fits' if entry['fits'] else 'exceeds'} maintenance_work_mem of "
            f"{entry['maintenance_work_mem'] / 2**30:.2f} GiB)"
        )
    if any(not entry["fits"] for entry in planned):
        typer.echo(
            "An index larger than maintenance_work_mem builds much more slowly. Raise "
            "maintenance_work_mem, rebuild the global index with --index-precision float16, "
            "or build with max_parallel_maintenance_workers = 0."
        )
    if dry_run:
        return
    if not yes and not typer.confirm("Drop the current partial indexes and build these?"):
        raise typer.Exit(1)
    result = partials.build_partials(store, valid_only=valid_only, domains=domains, analyze=analyze)
    _workspace(work_dir).append_build({"label": label, **result})
    typer.echo(f"Built {len(result['indexes'])} partial indexes.")


@app.command()
def drop(
    model_name: ModelName = None,
    yes: Annotated[bool, typer.Option("--yes", help="Drop without asking.")] = False,
) -> None:
    """Drop the partial indexes and turn them off in the registry. Writes to the store."""
    store = open_store(model_name, writable=True)
    names = partials.partial_index_names(store)
    typer.echo(f"Partial indexes: {names or 'none'}")
    if names and not yes and not typer.confirm("Drop them?"):
        raise typer.Exit(1)
    partials.drop_partials(store)


@app.command()
def compare(
    labels: Annotated[
        list[str] | None, typer.Argument(help="Runs to compare. Default all.")
    ] = None,
    work_dir: WorkDir = Path("recall_benchmark"),
    weight_single: Annotated[
        float, typer.Option(help="Weight of single-domain searches in the workload.")
    ] = 0.835,
    weight_none: Annotated[
        float, typer.Option(help="Weight of searches without a domain in the workload.")
    ] = 0.165,
) -> None:
    """Write the comparison of runs to comparison.md and print it. Read-only."""
    workspace = _workspace(work_dir)
    text = report.compare(workspace, labels, {"single": weight_single, "none": weight_none})
    (work_dir / "comparison.md").write_text(text, encoding="utf-8")
    typer.echo(text)
