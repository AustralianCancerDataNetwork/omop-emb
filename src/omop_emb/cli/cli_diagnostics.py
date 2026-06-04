"""Diagnostics for omop-emb CLI."""

import logging

import sqlalchemy as sa
import typer

from omop_emb.backends import resolve_backend
from omop_emb.config import MetricType, resolve_omop_cdm_engine
from omop_emb.interface import list_registered_models

logger = logging.getLogger(__name__)
app = typer.Typer(help="Diagnostics for embedding storage and retrieval.")


@app.command(name="health-check", help="Verify backend connectivity and list registered models.")
def health_check():
    backend = resolve_backend()
    typer.echo(f"Backend: {backend.backend_type.value} | connected.")

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

    records = list_registered_models(backend=backend)
    if not records:
        typer.echo("No registered models found.")
        return

    typer.echo(f"\n{len(records)} registered model(s):")
    typer.echo(f"  {'Model':<40} {'Provider':<10} {'Metric':<8} {'Index':<6} {'Dims':<6} {'Table'}")
    typer.echo("  " + "-" * 95)
    for r in records:
        index_str = r.index_type.value if r.index_type else "none"
        metric_str = r.metric_type.value if r.metric_type else "any"
        provider_str = r.provider_type.value if r.provider_type else "-"
        typer.echo(
            f"  {r.model_name:<40} {provider_str:<10} {metric_str:<8} "
            f"{index_str:<6} {r.dimensions:<6} {r.storage_identifier}"
        )
        probe_metric = r.metric_type or MetricType.COSINE
        has_emb = backend.has_any_embeddings(
            model_name=r.model_name,
            metric_type=probe_metric,
        )
        typer.echo(f"    embeddings present: {has_emb}")
