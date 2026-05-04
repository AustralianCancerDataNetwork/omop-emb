"""Maintenance and management commands for omop-emb."""

from typing import Annotated, Optional
from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)
import typer
app = typer.Typer(help="Maintenance and management commands for omop-emb.")

from .utils import configure_logging_level, resolve_omop_cdm_engine, resolve_emb_engine
from omop_emb.config import (
    IndexType,
    MetricType,
    ProviderType,
    BackendType,
    SUPPORTED_INDICES_AND_METRICS_PER_BACKEND,
)
from omop_emb.storage import PGVectorEmbeddingBackend
from omop_emb.storage.index_config import index_config_from_index_type
from omop_emb.interface import list_registered_models


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
    index_type: Annotated[Optional[IndexType], typer.Option(
        "--index-type",
        help="Filter by index type.",
    )] = None,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()

    emb_engine = resolve_emb_engine()
    records = list_registered_models(
        emb_engine=emb_engine,
        provider_type=provider_type,
        model_name=model,
        index_type=index_type,
    )

    if not records:
        typer.echo("No registered models found.")
        return

    typer.echo(f"{'Model':<40} {'Provider':<10} {'Index':<6} {'Dims':<6} {'Table'}")
    typer.echo("-" * 90)
    for r in records:
        typer.echo(
            f"{r.model_name:<40} {r.provider_type:<10} {r.index_type:<6} "
            f"{r.dimensions:<6} {r.storage_identifier}"
        )


@app.command(name="rebuild-index", help="Rebuild pgvector indexes for a registered model.")
def rebuild_index(
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Canonical model name to rebuild indexes for.",
    )],
    provider_type: Annotated[ProviderType, typer.Option(
        "--provider-type",
        help="Provider the model was registered with.",
    )] = ProviderType.OPENAI,
    index_type: Annotated[IndexType, typer.Option(
        "--index-type",
        help="Index type the model was registered with.",
        rich_help_panel="Index Options",
    )] = IndexType.HNSW,
    metric_types: Annotated[Optional[list[MetricType]], typer.Option(
        "--metric-type",
        help="Metrics to rebuild. Repeat to rebuild multiple. Defaults to all supported metrics.",
        rich_help_panel="Index Options",
    )] = None,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size when streaming embeddings during rebuild.",
    )] = 100_000,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()

    emb_engine = resolve_emb_engine()
    omop_cdm_engine = resolve_omop_cdm_engine()
    backend = PGVectorEmbeddingBackend(emb_engine=emb_engine, omop_cdm_engine=omop_cdm_engine)

    record = backend.get_registered_model(
        model_name=model,
        provider_type=provider_type,
        index_type=index_type,
    )
    if record is None:
        raise typer.BadParameter(
            f"No registered model found for '{model}' "
            f"(provider={provider_type.value}, index={index_type.value})."
        )

    resolved_metrics: tuple[MetricType, ...]
    if metric_types:
        resolved_metrics = tuple(metric_types)
    else:
        resolved_metrics = SUPPORTED_INDICES_AND_METRICS_PER_BACKEND.get(
            BackendType.PGVECTOR, {}
        ).get(index_type, ())

    backend.rebuild_model_indexes(
        model_name=model,
        provider_type=provider_type,
        index_type=index_type,
        metric_types=resolved_metrics,
        batch_size=batch_size,
        _model_record=record,
    )
    logger.info("Completed index rebuild for model '%s'.", model)
    typer.echo(f"Rebuilt {len(resolved_metrics)} index(es) for '{model}' ({index_type.value}).")


@app.command(name="switch-index-type", help="Register and build a new index type for an existing model.")
def switch_index_type(
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Canonical model name to add the new index for.",
    )],
    new_index_type: Annotated[IndexType, typer.Option(
        "--new-index-type",
        help="New index type to register and build.",
        rich_help_panel="Index Options",
    )] = IndexType.HNSW,
    provider_type: Annotated[ProviderType, typer.Option(
        "--provider-type",
        help="Provider the model was registered with.",
    )] = ProviderType.OPENAI,
    metric_types: Annotated[Optional[list[MetricType]], typer.Option(
        "--metric-type",
        help="Metrics to build. Defaults to all supported metrics for the new index type.",
        rich_help_panel="Index Options",
    )] = None,
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
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size when streaming embeddings during rebuild.",
    )] = 100_000,
    rebuild: Annotated[bool, typer.Option(
        "--rebuild/--no-rebuild",
        help="Build the new index immediately after registration.",
    )] = True,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()

    emb_engine = resolve_emb_engine()
    omop_cdm_engine = resolve_omop_cdm_engine()
    backend = PGVectorEmbeddingBackend(emb_engine=emb_engine, omop_cdm_engine=omop_cdm_engine)

    existing = backend.get_registered_models(model_name=model, provider_type=provider_type)
    if not existing:
        raise typer.BadParameter(
            f"No registration found for model '{model}' with provider '{provider_type.value}'. "
            "Register the model first via 'omop-emb embeddings add-embeddings'."
        )

    new_index_config = index_config_from_index_type(
        new_index_type,
        num_neighbors=index_hnsw_num_neighbors,
        ef_search=index_hnsw_ef_search,
        ef_construction=index_hnsw_ef_construction,
    )
    new_record = backend.register_model(
        model_name=model,
        dimensions=existing[0].dimensions,
        provider_type=provider_type,
        index_config=new_index_config,
    )
    typer.echo(f"Registered '{model}' with index_type={new_index_type.value}.")

    if rebuild:
        resolved_metrics: tuple[MetricType, ...]
        if metric_types:
            resolved_metrics = tuple(metric_types)
        else:
            resolved_metrics = SUPPORTED_INDICES_AND_METRICS_PER_BACKEND.get(
                BackendType.PGVECTOR, {}
            ).get(new_index_type, ())

        backend.rebuild_model_indexes(
            model_name=model,
            provider_type=provider_type,
            index_type=new_index_type,
            metric_types=resolved_metrics,
            batch_size=batch_size,
            _model_record=new_record,
        )
        typer.echo(f"Built {len(resolved_metrics)} index(es) for '{model}' ({new_index_type.value}).")


@app.command(name="export-faiss-cache", help="Export a FAISS sidecar cache from pgvector for a registered model.")
def export_faiss_cache(
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Canonical model name to export FAISS cache for.",
    )],
    api_base: Annotated[str, typer.Option(
        "--api-base",
        help="Base URL of the embedding API (used to resolve provider type).",
        rich_help_panel="Embedding API Options",
    )],
    api_key: Annotated[str, typer.Option(
        "--api-key",
        help="API key for the embedding API.",
        rich_help_panel="Embedding API Options",
    )],
    cache_dir: Annotated[str, typer.Option(
        "--cache-dir",
        help="Directory where the FAISS index files and HDF5 snapshot will be written.",
    )],
    provider_type: Annotated[ProviderType, typer.Option(
        "--provider-type",
        help="Provider the model was registered with.",
    )] = ProviderType.OPENAI,
    index_type: Annotated[IndexType, typer.Option(
        "--index-type",
        help="Registered index type to cache.",
        rich_help_panel="Index Options",
    )] = IndexType.FLAT,
    metric_types: Annotated[Optional[list[MetricType]], typer.Option(
        "--metric-type",
        help="Metrics to build FAISS indices for. Defaults to all supported FAISS metrics.",
        rich_help_panel="Index Options",
    )] = None,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size when streaming embeddings from pgvector.",
    )] = 100_000,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()

    try:
        from omop_emb.storage.faiss_cache import FAISSCache
    except ImportError as exc:
        raise typer.Exit(1) from typer.BadParameter(
            f"FAISS optional dependency not installed: {exc}. "
            "Install it with: pip install omop-emb[faiss]"
        )

    from omop_emb.config import SUPPORTED_FAISS_CACHE_METRICS

    emb_engine = resolve_emb_engine()
    omop_cdm_engine = resolve_omop_cdm_engine()
    backend = PGVectorEmbeddingBackend(emb_engine=emb_engine, omop_cdm_engine=omop_cdm_engine)

    resolved_metrics: tuple[MetricType, ...]
    if metric_types:
        resolved_metrics = tuple(metric_types)
    else:
        resolved_metrics = tuple(SUPPORTED_FAISS_CACHE_METRICS)

    cache = FAISSCache(
        backend=backend,
        model_name=model,
        provider_type=provider_type,
        index_type=index_type,
        cache_dir=cache_dir,
    )
    cache.export(metric_types=resolved_metrics, batch_size=batch_size)
    typer.echo(f"FAISS cache exported to '{cache_dir}' for '{model}' ({index_type.value}).")


@app.command(name="check-faiss-cache", help="Check whether the FAISS sidecar cache is stale for a registered model.")
def check_faiss_cache(
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Canonical model name to check.",
    )],
    provider_type: Annotated[ProviderType, typer.Option(
        "--provider-type",
        help="Provider the model was registered with.",
    )] = ProviderType.OPENAI,
    index_type: Annotated[IndexType, typer.Option(
        "--index-type",
        help="Registered index type to check.",
    )] = IndexType.FLAT,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()

    try:
        from omop_emb.storage.faiss_cache import FAISSCache
    except ImportError as exc:
        raise typer.Exit(1) from typer.BadParameter(
            f"FAISS optional dependency not installed: {exc}. "
            "Install it with: pip install omop-emb[faiss]"
        )

    emb_engine = resolve_emb_engine()
    omop_cdm_engine = resolve_omop_cdm_engine()
    backend = PGVectorEmbeddingBackend(emb_engine=emb_engine, omop_cdm_engine=omop_cdm_engine)

    record = backend.get_registered_model(
        model_name=model,
        provider_type=provider_type,
        index_type=index_type,
    )
    if record is None:
        typer.echo(f"No registered model found for '{model}' ({provider_type.value}, {index_type.value}).")
        raise typer.Exit(1)

    cache_meta = record.metadata.get("faiss_cache") if record.metadata else None
    if cache_meta is None:
        typer.echo(f"No FAISS cache metadata found for '{model}'. Run 'export-faiss-cache' first.")
        raise typer.Exit(1)

    cache_dir = cache_meta.get("cache_dir")
    if not cache_dir:
        typer.echo("FAISS cache metadata is missing 'cache_dir'. Re-export the cache.")
        raise typer.Exit(1)

    cache = FAISSCache(
        backend=backend,
        model_name=model,
        provider_type=provider_type,
        index_type=index_type,
        cache_dir=cache_dir,
    )

    stale = cache.is_stale()
    exported_at = cache_meta.get("exported_at", "unknown")
    row_count = cache_meta.get("row_count", "unknown")
    typer.echo(
        f"Model: {model} | Index: {index_type.value} | "
        f"Exported: {exported_at} | Rows: {row_count} | "
        f"Status: {'STALE' if stale else 'FRESH'}"
    )
    raise typer.Exit(1 if stale else 0)
