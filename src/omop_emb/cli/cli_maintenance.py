"""Maintenance and management commands for omop-emb."""

import logging
from typing import Annotated, Optional

import typer
from oa_configurator import Resolver
from omop_llm.providers import canonical_model_name

from omop_emb.backends import (
    EmbeddingStoreReader,
    open_vector_store_reader,
    open_vector_store_writer,
)
from omop_emb.backends.index_config import index_config_from_index_type
from omop_emb.config import (
    IndexType,
    MetricType,
    OmopEmbConfig,
    open_cdm_sessions,
)
from omop_emb.population import PopulationScope, plan_population
from omop_emb.storage import embedding_bundle
from omop_emb.utils.cdm import CDMSessionFactory

logger = logging.getLogger(__name__)
app = typer.Typer(help="Maintenance and management commands for omop-emb.")


@app.command(name="list-models", help="List all registered embedding models.")
def list_models(
    provider_type: Annotated[
        Optional[str],
        typer.Option(
            "--provider-type",
            help="Filter by embedding provider.",
        ),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Filter by model name.",
        ),
    ] = None,
):

    cfg = OmopEmbConfig.get_config()
    resolved = Resolver.from_active_config().resolve_vector_store(cfg.vector_store_name)
    with open_vector_store_reader(resolved) as store:
        records = store.get_registered_models(model_name=model, provider_type=provider_type)

    if not records:
        typer.echo("No registered models found.")
        return

    typer.echo(
        f"{'Model':<40} {'Provider':<10} {'Metric':<8} {'Index':<6} {'Dims':<6} {'Table'}"
    )
    typer.echo("-" * 100)
    for r in records:
        index_str = r.index_type.value if r.index_type else "none"
        metric_str = r.metric_type.value if r.metric_type else "any"
        provider_str = r.provider_type if r.provider_type else "-"
        typer.echo(
            f"{r.model_name:<40} {provider_str:<10} {metric_str:<8} "
            f"{index_str:<6} {r.dimensions:<6} {r.storage_identifier}"
        )


@app.command(
    name="rebuild-index", help="Build or rebuild the index on an embedding table."
)
def rebuild_index(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Canonical model name to rebuild the index for.",
        ),
    ],
    provider_type: Annotated[
        Optional[str],
        typer.Option(
            "--provider-type",
            help="Embedding provider type. Required to determine canonical model name.",
        ),
    ] = None,
    index_type: Annotated[
        IndexType,
        typer.Option(
            "--index-type",
            help="Index type to build (FLAT = exact scan, HNSW = approximate; pgvector only).",
            rich_help_panel="Index Options",
        ),
    ] = IndexType.FLAT,
    metric_type: Annotated[
        MetricType,
        typer.Option(
            "--metric-type",
            help="Distance metric. Locked in for HNSW and for FLAT on sqlite-vec. "
            "Ignored for FLAT on pgvector, which takes the metric per query.",
            rich_help_panel="Index Options",
        ),
    ] = MetricType.COSINE,
    index_hnsw_num_neighbors: Annotated[
        Optional[int],
        typer.Option(
            "--index-hnsw-num-neighbors",
            help="HNSW: number of neighbors per graph node.",
            rich_help_panel="Index Options",
        ),
    ] = None,
    index_hnsw_ef_search: Annotated[
        Optional[int],
        typer.Option(
            "--index-hnsw-ef-search",
            help="HNSW: ef parameter controlling recall during search.",
            rich_help_panel="Index Options",
        ),
    ] = None,
    index_hnsw_ef_construction: Annotated[
        Optional[int],
        typer.Option(
            "--index-ef-construction",
            help="HNSW: ef parameter controlling graph quality during construction.",
            rich_help_panel="Index Options",
        ),
    ] = None,
    vacuum: Annotated[
        bool,
        typer.Option(
            "--vacuum/--no-vacuum",
            help="Run the backend's maintenance step after the rebuild: VACUUM (ANALYZE) on the table "
            "for pgvector, VACUUM of the database file after a table recreation for sqlite-vec "
            "(needs free disk space for a second copy of the data).",
            rich_help_panel="Index Options",
        ),
    ] = True,
):

    if provider_type is not None:
        model = canonical_model_name(provider_type, model)

    cfg = OmopEmbConfig.get_config()
    resolved = Resolver.from_active_config().resolve_vector_store(cfg.vector_store_name)
    backend = open_vector_store_writer(resolved)

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
        metric_type=metric_type,
    )
    record = backend.rebuild_index(
        model_name=model,
        index_config=index_config,
        vacuum=vacuum,
    )
    metric_info = f" (metric={record.metric_type.value})" if record.metric_type else ""
    typer.echo(f"Rebuilt {index_type.value} index for '{model}'{metric_info}.")


@app.command(
    name="delete-model",
    help="Irreversibly delete a registered model and all its embeddings.",
)
def delete_model(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Canonical model name to delete.",
        ),
    ],
    provider_type: Annotated[
        Optional[str],
        typer.Option(
            "--provider-type",
            help="Embedding provider type. Used to check canonical model_name if provided.",
        ),
    ] = None,
    confirm: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip confirmation prompt.",
        ),
    ] = False,
):

    if provider_type is not None:
        model = canonical_model_name(provider_type, model)

    if not confirm:
        typer.confirm(
            f"Delete model '{model}' and ALL associated embeddings? This cannot be undone.",
            abort=True,
        )

    cfg = OmopEmbConfig.get_config()
    resolved = Resolver.from_active_config().resolve_vector_store(cfg.vector_store_name)
    backend = open_vector_store_writer(resolved)
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


@app.command(
    name="prune-embeddings",
    help="Remove stored embeddings that fall outside the given scope (dry run unless --apply).",
)
def prune_embeddings(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Canonical model name to prune.",
        ),
    ],
    provider_type: Annotated[
        Optional[str],
        typer.Option(
            "--provider-type",
            help="Embedding provider type. Used to canonicalize the model name if provided.",
        ),
    ] = None,
    standard_only: Annotated[
        bool,
        typer.Option(
            "--standard-only",
            help="Keep only OMOP standard concepts (standard_concept in 'S' or 'C').",
            rich_help_panel="Scope",
        ),
    ] = False,
    include_invalid: Annotated[
        bool,
        typer.Option(
            "--include-invalid",
            help="Keep invalid concepts (invalid_reason 'D' or 'U'). Pruned by default.",
            rich_help_panel="Scope",
        ),
    ] = False,
    vocabularies: Annotated[
        Optional[list[str]],
        typer.Option(
            "--vocabulary",
            help="Keep only concepts from specific OMOP vocabularies. Repeat to add multiple.",
            rich_help_panel="Scope",
        ),
    ] = None,
    domains: Annotated[
        Optional[list[str]],
        typer.Option(
            "--domain",
            help="Keep only concepts from specific OMOP domains. Repeat to add multiple.",
            rich_help_panel="Scope",
        ),
    ] = None,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Delete the rows. Without it the command only reports what would be deleted.",
        ),
    ] = False,
):
    """Delete stored embeddings of concepts that are no longer in scope.

    A row is pruned when its concept is not eligible under the scope (for
    example invalid, non-standard or outside the listed vocabularies and
    domains) or no longer exists in the CDM. Run ``rebuild-index`` afterwards
    on HNSW models to reclaim index space.
    """
    if provider_type is not None:
        model = canonical_model_name(provider_type, model)

    cfg = OmopEmbConfig.get_config()
    resolved = Resolver.from_active_config().resolve_vector_store(cfg.vector_store_name)
    scope = PopulationScope(
        standard_only=standard_only,
        valid_only=not include_invalid,
        vocabularies=tuple(vocabularies or ()),
        domains=tuple(domains or ()),
    )

    with open_cdm_sessions() as cdm_sessions:
        if not apply:
            with open_vector_store_reader(resolved) as store:
                prunable = _prune_candidates(store, cdm_sessions, model, scope)
            if prunable:
                typer.echo(
                    f"Dry run: {len(prunable):,} embeddings would be deleted. Pass --apply to delete them."
                )
            return

        with open_vector_store_writer(resolved) as writer:
            prunable = _prune_candidates(writer, cdm_sessions, model, scope)
            if not prunable:
                return
            deleted = writer.delete_embeddings(model_name=model, concept_ids=prunable)
            writer.refresh_model_updated_at_timestamp(model_name=model)
        typer.echo(f"Deleted {deleted:,} embeddings from '{model}'.")


def _prune_candidates(
    store: EmbeddingStoreReader,
    cdm_sessions: CDMSessionFactory,
    model: str,
    scope: PopulationScope,
) -> list[int]:
    """Print the pruning summary for model and return the concept IDs to delete."""
    if store.get_registered_model(model_name=model) is None:
        typer.echo(f"No registered model found for '{model}'.")
        raise typer.Exit(1)
    plan = plan_population(cdm_sessions, store, model_name=model, scope=scope)
    kept = len(plan.compatible_ids | plan.metadata_changed_ids)
    typer.echo(f"Embeddings kept: {kept:,}")
    typer.echo(f"Outside the scope: {len(plan.out_of_scope_ids):,}")
    typer.echo(f"Concept no longer in the CDM: {len(plan.stale_ids):,}")
    if not plan.prunable_ids:
        typer.echo("Nothing to prune.")
    return sorted(plan.prunable_ids)


@app.command(
    name="export",
    help="Export raw embeddings + metadata into a single, self-describing HDF5 bundle.",
)
def export_bundle_cmd(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Canonical model name to export.",
        ),
    ],
    output_dir: Annotated[
        str,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to write the HDF5 bundle into. The filename is derived "
            "from the model's storage identifier.",
        ),
    ],
    provider_type: Annotated[
        Optional[str],
        typer.Option(
            "--provider-type",
            help="Embedding provider type. Used to canonicalize the model name if provided.",
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Number of rows streamed from the backend (and written to the bundle) per batch.",
        ),
    ] = 100_000,
):

    if provider_type is not None:
        model = canonical_model_name(provider_type, model)

    cfg = OmopEmbConfig.get_config()
    resolved = Resolver.from_active_config().resolve_vector_store(cfg.vector_store_name)
    with open_vector_store_reader(resolved) as store:
        meta, h5_path = embedding_bundle.export_bundle(
            backend=store,
            model_name=model,
            output_dir=output_dir,
            batch_size=batch_size,
        )
    typer.echo(f"Exported {meta.row_count} embeddings for '{model}' to '{h5_path}'.")


@app.command(
    name="build-faiss-cache",
    help="Build (or rebuild) a local FAISS search cache directly from the backend.",
)
def build_faiss_cache(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Canonical model name to build the FAISS cache for.",
        ),
    ],
    faiss_cache_dir: Annotated[
        str,
        typer.Option(
            "--faiss-cache-dir",
            help="Root directory where the FAISS index files will be written.",
        ),
    ],
    provider_type: Annotated[
        Optional[str],
        typer.Option(
            "--provider-type",
            help="Embedding provider type. Used to canonicalize the model name if provided.",
        ),
    ] = None,
    metric_type: Annotated[
        MetricType,
        typer.Option(
            "--metric-type",
            help="Distance metric for the FAISS index. Must be L2 or COSINE.",
            rich_help_panel="Index Options",
        ),
    ] = MetricType.COSINE,
    index_type: Annotated[
        IndexType,
        typer.Option(
            "--index-type",
            help="FAISS index type: FLAT (exact scan) or HNSW (approximate, faster at scale).",
            rich_help_panel="Index Options",
        ),
    ] = IndexType.FLAT,
    hnsw_m: Annotated[
        int,
        typer.Option(
            "--hnsw-m",
            help="HNSW number of neighbours (M). Only used when --index-type=HNSW.",
            rich_help_panel="Index Options",
        ),
    ] = 32,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Number of rows streamed from the backend per batch.",
        ),
    ] = 100_000,
):

    try:
        from omop_emb.storage.faiss import FAISSCache
    except ImportError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    if provider_type is not None:
        model = canonical_model_name(provider_type, model)

    cfg = OmopEmbConfig.get_config()
    resolved = Resolver.from_active_config().resolve_vector_store(cfg.vector_store_name)
    index_config = index_config_from_index_type(
        index_type,
        metric_type=metric_type,
        num_neighbors=hnsw_m,
    )
    cache = FAISSCache(model_name=model, cache_dir=faiss_cache_dir)
    with open_vector_store_reader(resolved) as store:
        cache.build_from_backend(
            backend=store,
            metric_type=metric_type,
            index_config=index_config,
            batch_size=batch_size,
        )
    typer.echo(
        f"FAISS cache built at '{cache.model_dir}' for '{model}' "
        f"(metric={metric_type.value}, index={index_type.value})."
    )


@app.command(
    name="check-faiss-cache", help="Check whether the FAISS sidecar cache is fresh."
)
def check_faiss_cache(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Canonical model name to check.",
        ),
    ],
    cache_dir: Annotated[
        str,
        typer.Option(
            "--cache-dir",
            help="Root cache directory passed to FAISSCache.",
        ),
    ],
    metric_type: Annotated[
        MetricType,
        typer.Option(
            "--metric-type",
            help="Distance metric of the index to check.",
            rich_help_panel="Index Options",
        ),
    ] = MetricType.COSINE,
    index_type: Annotated[
        IndexType,
        typer.Option(
            "--index-type",
            help="Index type to check (FLAT or HNSW).",
            rich_help_panel="Index Options",
        ),
    ] = IndexType.FLAT,
    provider_type: Annotated[
        Optional[str],
        typer.Option(
            "--provider-type",
            help="Embedding provider type. Used to canonicalize the model name if provided.",
        ),
    ] = None,
):

    if provider_type is not None:
        model = canonical_model_name(provider_type, model)

    cfg = OmopEmbConfig.get_config()
    resolved = Resolver.from_active_config().resolve_vector_store(cfg.vector_store_name)
    with open_vector_store_reader(resolved) as store:
        record = store.get_registered_model(model_name=model)
        registered = [r.model_name for r in store.get_registered_models()]
    if record is None:
        typer.echo(
            f"No registered model found for '{model}'. Registered models: {registered}",
            err=True,
        )
        raise typer.Exit(1)

    index_config = index_config_from_index_type(index_type, metric_type=metric_type)
    try:
        from omop_emb.storage.faiss import FAISSCache

        cache = FAISSCache(model_name=model, cache_dir=cache_dir)
    except ImportError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

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


@app.command(
    name="import",
    help="Import embeddings from an export bundle into the backend.",
)
def import_bundle_cmd(
    bundle_file: Annotated[
        str,
        typer.Option(
            "--bundle-file",
            help="Path to the HDF5 bundle produced by 'maintenance export'.",
        ),
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite existing embeddings without prompting.",
        ),
    ] = False,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Number of vectors upserted per backend call.",
        ),
    ] = 10_000,
    rebuild_index: Annotated[
        bool,
        typer.Option(
            "--rebuild-index",
            help="Build the index recorded in the bundle's metadata right after import.",
        ),
    ] = False,
):

    cfg = OmopEmbConfig.get_config()
    resolved = Resolver.from_active_config().resolve_vector_store(cfg.vector_store_name)
    backend = open_vector_store_writer(resolved)
    try:
        imported = embedding_bundle.import_bundle(
            backend=backend,
            h5_path=bundle_file,
            force=force,
            batch_size=batch_size,
            rebuild_index=rebuild_index,
        )
    except (RuntimeError, embedding_bundle.BundleCorruptionError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    suffix = " and rebuilt its index" if rebuild_index else ""
    typer.echo(f"Imported {imported} concepts from '{bundle_file}'{suffix}.")
