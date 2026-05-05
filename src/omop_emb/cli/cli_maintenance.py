"""Maintenance and management commands for omop-emb."""

import logging
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv

from .utils import configure_logging_level, resolve_backend
from omop_emb.backends.index_config import index_config_from_index_type
from omop_emb.config import (
    BackendType,
    IndexType,
    MetricType,
    ProviderType,
    SUPPORTED_INDICES_AND_METRICS_PER_BACKEND,
)
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
        typer.echo(
            f"{r.model_name:<40} {r.provider_type:<10} {metric_str:<8} "
            f"{index_str:<6} {r.dimensions:<6} {r.storage_identifier}"
        )


@app.command(name="rebuild-index", help="Build or rebuild the index on an embedding table.")
def rebuild_index(
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Canonical model name to rebuild the index for.",
    )],
    provider_type: Annotated[ProviderType, typer.Option(
        "--provider-type",
        help="Provider the model was registered with.",
    )] = ProviderType.OPENAI,
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

    backend = resolve_backend()

    record = backend.get_registered_model(
        model_name=model,
        provider_type=provider_type,
    )
    if record is None:
        raise typer.BadParameter(
            f"No registered model found for '{model}' (provider={provider_type.value})."
        )

    index_config = index_config_from_index_type(
        index_type,
        num_neighbors=index_hnsw_num_neighbors,
        ef_search=index_hnsw_ef_search,
        ef_construction=index_hnsw_ef_construction,
        metric_type=metric_type if index_type == IndexType.HNSW else None,
    )
    backend.rebuild_index(
        model_name=model,
        provider_type=provider_type,
        index_config=index_config,
    )
    metric_info = f" (metric={metric_type.value})" if index_type == IndexType.HNSW else ""
    typer.echo(f"Rebuilt {index_type.value} index for '{model}'{metric_info}.")
    logger.info("Completed index rebuild for model '%s'.", model)


@app.command(name="delete-model", help="Irreversibly delete a registered model and all its embeddings.")
def delete_model(
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Canonical model name to delete.",
    )],
    provider_type: Annotated[ProviderType, typer.Option(
        "--provider-type",
        help="Provider the model was registered with.",
    )] = ProviderType.OPENAI,
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

    if not confirm:
        typer.confirm(
            f"Delete model '{model}' (provider={provider_type.value}) "
            "and ALL associated embeddings? This cannot be undone.",
            abort=True,
        )

    backend = resolve_backend()
    backend.delete_model(
        model_name=model,
        provider_type=provider_type,
    )
    typer.echo(f"Deleted model '{model}' (provider={provider_type.value}).")


@app.command(name="export-faiss-cache", help="Export a FAISS sidecar cache from the embedding store.")
def export_faiss_cache(
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Canonical model name to export FAISS cache for.",
    )],
    cache_dir: Annotated[str, typer.Option(
        "--cache-dir",
        help="Directory where the FAISS index files will be written.",
    )],
    metric_type: Annotated[MetricType, typer.Option(
        "--metric-type",
        help="Distance metric for the FAISS index.",
        rich_help_panel="Index Options",
    )] = MetricType.COSINE,
    provider_type: Annotated[ProviderType, typer.Option(
        "--provider-type",
        help="Provider the model was registered with.",
    )] = ProviderType.OPENAI,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size when streaming embeddings.",
    )] = 100_000,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()

    try:
        from omop_emb.storage.faiss import FAISSCache
    except ImportError as exc:
        raise typer.Exit(1) from typer.BadParameter(
            f"FAISS optional dependency not installed: {exc}. "
            "Install it with: pip install omop-emb[faiss]"
        )

    backend = resolve_backend()
    cache = FAISSCache(
        backend=backend,
        model_name=model,
        provider_type=provider_type,
        metric_type=metric_type,
        cache_dir=cache_dir,
    )
    cache.export(batch_size=batch_size)
    typer.echo(f"FAISS cache exported to '{cache_dir}' for '{model}' (metric={metric_type.value}).")


@app.command(name="check-faiss-cache", help="Check whether the FAISS sidecar cache is stale.")
def check_faiss_cache(
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Canonical model name to check.",
    )],
    metric_type: Annotated[MetricType, typer.Option(
        "--metric-type",
        help="Distance metric of the FAISS index to check.",
    )] = MetricType.COSINE,
    provider_type: Annotated[ProviderType, typer.Option(
        "--provider-type",
        help="Provider the model was registered with.",
    )] = ProviderType.OPENAI,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()

    try:
        from omop_emb.storage.faiss import FAISSCache
    except ImportError as exc:
        raise typer.Exit(1) from typer.BadParameter(
            f"FAISS optional dependency not installed: {exc}. "
            "Install it with: pip install omop-emb[faiss]"
        )

    backend = resolve_backend()
    record = backend.get_registered_model(
        model_name=model,
        provider_type=provider_type,
    )
    if record is None:
        typer.echo(
            f"No registered model found for '{model}' (provider={provider_type.value})."
        )
        raise typer.Exit(1)

    cache_meta = record.metadata.get("faiss_cache") if record.metadata else None
    if cache_meta is None:
        typer.echo(f"No FAISS cache metadata for '{model}'. Run 'export-faiss-cache' first.")
        raise typer.Exit(1)

    cache_dir = cache_meta.get("cache_dir")
    if not cache_dir:
        typer.echo("FAISS cache metadata is missing 'cache_dir'. Re-export the cache.")
        raise typer.Exit(1)

    cache = FAISSCache(
        backend=backend,
        model_name=model,
        provider_type=provider_type,
        metric_type=metric_type,
        cache_dir=cache_dir,
    )

    stale = cache.is_stale()
    exported_at = cache_meta.get("exported_at", "unknown")
    row_count = cache_meta.get("row_count", "unknown")
    typer.echo(
        f"Model: {model} | Metric: {metric_type.value} | "
        f"Exported: {exported_at} | Rows: {row_count} | "
        f"Status: {'STALE' if stale else 'FRESH'}"
    )
    raise typer.Exit(1 if stale else 0)
