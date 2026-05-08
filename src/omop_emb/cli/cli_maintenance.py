"""Maintenance and management commands for omop-emb."""

import logging
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv

from .utils import configure_logging_level, resolve_backend
from omop_emb.backends.index_config import index_config_from_index_type
from omop_emb.config import (
    IndexType,
    MetricType,
    ProviderType,
)
from omop_emb.embeddings.embedding_providers import get_provider_from_provider_type
from omop_emb.interface import list_registered_models

logger = logging.getLogger(__name__)
app = typer.Typer(help="Maintenance and management commands for omop-emb.")


@app.command(name="list-models", help="List all registered embedding models.")
def list_models(
    provider_type: Annotated[Optional[ProviderType], typer.Option(
        "--provider-type",
        help="Filter by embedding provider.",
    )] = None,
    model: Annotated[Optional[str], typer.Option(
        "--model", "-m",
        help="Filter by model name.",
    )] = None,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()

    backend = resolve_backend()
    records = list_registered_models(
        backend=backend,
        provider_type=provider_type,
        model_name=model,
    )

    if not records:
        typer.echo("No registered models found.")
        return

    typer.echo(f"{'Model':<40} {'Provider':<10} {'Metric':<8} {'Index':<6} {'Dims':<6} {'Table'}")
    typer.echo("-" * 100)
    for r in records:
        index_str = r.index_type.value if r.index_type else "none"
        metric_str = r.metric_type.value if r.metric_type else "any"
        provider_str = r.provider_type.value if r.provider_type else "-"
        typer.echo(
            f"{r.model_name:<40} {provider_str:<10} {metric_str:<8} "
            f"{index_str:<6} {r.dimensions:<6} {r.storage_identifier}"
        )


@app.command(name="rebuild-index", help="Build or rebuild the index on an embedding table.")
def rebuild_index(
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Canonical model name to rebuild the index for.",
    )],
    provider_type: Annotated[Optional[ProviderType], typer.Option(
        "--provider-type",
        help="Embedding provider type. Required to determine canonical model name.",
    )] = None,
    index_type: Annotated[IndexType, typer.Option(
        "--index-type",
        help="Index type to build (FLAT = exact scan, HNSW = approximate; pgvector only).",
        rich_help_panel="Index Options",
    )] = IndexType.FLAT,
    metric_type: Annotated[MetricType, typer.Option(
        "--metric-type",
        help="Distance metric. Required and locked in when --index-type is HNSW.",
        rich_help_panel="Index Options",
    )] = MetricType.COSINE,
    index_hnsw_num_neighbors: Annotated[Optional[int], typer.Option(
        "--index-hnsw-num-neighbors",
        help="HNSW: number of neighbors per graph node.",
        rich_help_panel="Index Options",
    )] = None,
    index_hnsw_ef_search: Annotated[Optional[int], typer.Option(
        "--index-hnsw-ef-search",
        help="HNSW: ef parameter controlling recall during search.",
        rich_help_panel="Index Options",
    )] = None,
    index_hnsw_ef_construction: Annotated[Optional[int], typer.Option(
        "--index-ef-construction",
        help="HNSW: ef parameter controlling graph quality during construction.",
        rich_help_panel="Index Options",
    )] = None,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()

    if provider_type is not None:
        embedding_provider = get_provider_from_provider_type(provider_type)
        model = embedding_provider.canonical_model_name(model)

    backend = resolve_backend()

    record = backend.get_registered_model(model_name=model)
    if record is None:
        typer.echo(
            f"No registered model found for '{model}'.\n"
            f"Could be that you didn't canonicalize the model name correctly or that the model was never registered.\n"
            f"Registered models: {[r.model_name for r in backend.get_registered_models()]}\n."
        )
        raise typer.Exit(1)

    index_config = index_config_from_index_type(
        index_type,
        num_neighbors=index_hnsw_num_neighbors,
        ef_search=index_hnsw_ef_search,
        ef_construction=index_hnsw_ef_construction,
        metric_type=metric_type if index_type == IndexType.HNSW else None,
    )
    backend.rebuild_index(
        model_name=model,
        index_config=index_config,
    )
    metric_info = f" (metric={metric_type.value})" if index_type == IndexType.HNSW else ""
    typer.echo(f"Rebuilt {index_type.value} index for '{model}'{metric_info}.")


@app.command(name="delete-model", help="Irreversibly delete a registered model and all its embeddings.")
def delete_model(
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Canonical model name to delete.",
    )],
    provider_type: Annotated[Optional[ProviderType], typer.Option(
        "--provider-type",
        help="Embedding provider type. Used to check canonical model_name if provided.",
    )] = None,
    confirm: Annotated[bool, typer.Option(
        "--yes", "-y",
        help="Skip confirmation prompt.",
    )] = False,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()

    if provider_type is not None:
        embedding_provider = get_provider_from_provider_type(provider_type)
        model = embedding_provider.canonical_model_name(model)

    if not confirm:
        typer.confirm(
            f"Delete model '{model}' and ALL associated embeddings? This cannot be undone.",
            abort=True,
        )

    backend = resolve_backend()
    record = backend.get_registered_model(model_name=model)
    if record is None:
        typer.echo(
            f"No registered model found for '{model}'.\n"
            f"Could be that you didn't canonicalize the model name correctly or that the model was never registered.\n"
            f"Registered models: {[r.model_name for r in backend.get_registered_models()]}\n."
            f"Not deleting {model}."
        )
        raise typer.Exit(1)

    backend.delete_model(model_name=model)
    typer.echo(f"Deleted model '{model}'.")


@app.command(name="export-faiss-cache", help="Export a FAISS sidecar cache from the embedding store.")
def export_faiss_cache(
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Canonical model name to export FAISS cache for.",
    )],
    cache_dir: Annotated[str, typer.Option(
        "--cache-dir",
        help="Root directory where the FAISS index files will be written.",
    )],
    metric_type: Annotated[MetricType, typer.Option(
        "--metric-type",
        help="Distance metric for the FAISS index. Must be L2 or COSINE.",
        rich_help_panel="Index Options",
    )] = MetricType.COSINE,
    index_type: Annotated[IndexType, typer.Option(
        "--index-type",
        help="FAISS index type: FLAT (exact scan) or HNSW (approximate, faster at scale).",
        rich_help_panel="Index Options",
    )] = IndexType.FLAT,
    hnsw_m: Annotated[int, typer.Option(
        "--hnsw-m",
        help="HNSW number of neighbours (M). Only used when --index-type=HNSW.",
        rich_help_panel="Index Options",
    )] = 32,
    provider_type: Annotated[Optional[ProviderType], typer.Option(
        "--provider-type",
        help="Embedding provider type. Used to canonicalize the model name if provided.",
    )] = None,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size when streaming embeddings from the backend.",
    )] = 100_000,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()

    if provider_type is not None:
        embedding_provider = get_provider_from_provider_type(provider_type)
        model = embedding_provider.canonical_model_name(model)

    try:
        from omop_emb.storage.faiss import FAISSCache
    except ImportError as exc:
        typer.echo(
            f"FAISS optional dependency not installed: {exc}. "
            "Install it with: pip install omop-emb[faiss]",
            err=True,
        )
        raise typer.Exit(1)

    index_config = index_config_from_index_type(
        index_type,
        metric_type=metric_type,
        num_neighbors=hnsw_m,
    )
    backend = resolve_backend()
    cache = FAISSCache(model_name=model, cache_dir=cache_dir)
    cache.export(backend=backend, metric_type=metric_type, index_config=index_config, batch_size=batch_size)
    typer.echo(
        f"FAISS cache exported to '{cache.model_dir}' for '{model}' "
        f"(metric={metric_type.value}, index={index_type.value})."
    )


@app.command(name="check-faiss-cache", help="Check whether the FAISS sidecar cache is fresh.")
def check_faiss_cache(
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Canonical model name to check.",
    )],
    cache_dir: Annotated[str, typer.Option(
        "--cache-dir",
        help="Root cache directory passed to FAISSCache.",
    )],
    metric_type: Annotated[MetricType, typer.Option(
        "--metric-type",
        help="Distance metric of the index to check.",
        rich_help_panel="Index Options",
    )] = MetricType.COSINE,
    index_type: Annotated[IndexType, typer.Option(
        "--index-type",
        help="Index type to check (FLAT or HNSW).",
        rich_help_panel="Index Options",
    )] = IndexType.FLAT,
    provider_type: Annotated[Optional[ProviderType], typer.Option(
        "--provider-type",
        help="Embedding provider type. Used to canonicalize the model name if provided.",
    )] = None,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()

    if provider_type is not None:
        embedding_provider = get_provider_from_provider_type(provider_type)
        model = embedding_provider.canonical_model_name(model)

    try:
        from omop_emb.storage.faiss import FAISSCache
    except ImportError as exc:
        typer.echo(
            f"FAISS optional dependency not installed: {exc}. "
            "Install it with: pip install omop-emb[faiss]",
            err=True,
        )
        raise typer.Exit(1)

    backend = resolve_backend()
    record = backend.get_registered_model(model_name=model)
    if record is None:
        typer.echo(
            f"No registered model found for '{model}'. "
            f"Registered models: {[r.model_name for r in backend.get_registered_models()]}",
            err=True,
        )
        raise typer.Exit(1)

    index_config = index_config_from_index_type(index_type, metric_type=metric_type)
    cache = FAISSCache(model_name=model, cache_dir=cache_dir)
    info = cache.staleness_info(record, metric_type, index_config)
    status = "FRESH" if info["is_fresh"] else "STALE"
    typer.echo(
        f"Model: {model} | "
        f"Index: {index_type.value}/{metric_type.value} | "
        f"Exported: {info['exported_at'] or 'never'} | "
        f"Cached rows: {info['cached_row_count'] or 'unknown'} | "
        f"Model updated: {info['model_updated_at'] or 'unknown'} | "
        f"Status: {status}"
    )
    raise typer.Exit(0 if info["is_fresh"] else 1)
