"""Diagnostics for omop-emb CLI."""

import logging

import sqlalchemy as sa
import typer
from oa_configurator import Resolver

from omop_emb.backends import open_vector_store_reader
from omop_emb.config import OmopEmbConfig, resolve_omop_cdm_engine
from omop_emb.utils.errors import MissingStorageTableError

logger = logging.getLogger(__name__)
app = typer.Typer(help="Diagnostics for embedding storage and retrieval.")


@app.command(
    name="health-check", help="Verify backend connectivity and list registered models."
)
def health_check():
    cfg = OmopEmbConfig.get_config()
    resolved = Resolver.from_active_config().resolve_vector_store(cfg.vector_store_name)
    with open_vector_store_reader(resolved) as store:
        typer.echo(f"Backend: {resolved.backend_type} | connected.")

        # CDM connectivity is optional for the health check
        try:
            omop_cdm_engine = resolve_omop_cdm_engine()
            with omop_cdm_engine.connect() as conn:
                conn.execute(sa.text("SELECT 1"))
            typer.echo(f"CDM engine: {omop_cdm_engine.url} | connected.")
        except RuntimeError as exc:
            typer.echo(f"CDM engine: not configured ({exc})")
        except Exception as exc:
            typer.echo(f"CDM engine: connection failed. {exc}")

        records = store.get_registered_models()
        if not records:
            typer.echo("No registered models found.")
            return

        typer.echo(f"\n{len(records)} registered model(s):")
        typer.echo(
            f"  {'Model':<40} {'Provider':<10} {'Metric':<8} {'Index':<6} {'Dims':<6} {'Table'}"
        )
        typer.echo("  " + "-" * 95)
        for r in records:
            index_str = r.index_type.value if r.index_type else "none"
            metric_str = r.metric_type.value if r.metric_type else "any"
            provider_str = r.provider_type if r.provider_type else "-"
            typer.echo(
                f"  {r.model_name:<40} {provider_str:<10} {metric_str:<8} "
                f"{index_str:<6} {r.dimensions:<6} {r.storage_identifier}"
            )
            try:
                has_emb = str(store.has_any_embeddings(model_name=r.model_name))
            except MissingStorageTableError:
                has_emb = "storage table missing"
            typer.echo(f"    embeddings present: {has_emb}")
