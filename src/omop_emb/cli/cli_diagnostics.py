"""Diagnostics for omop-emb CLI."""

import logging
from typing import Annotated

import sqlalchemy as sa
import typer
from dotenv import load_dotenv

from .utils import configure_logging_level, resolve_backend, resolve_omop_cdm_engine
from omop_emb.config import MetricType
from omop_emb.interface import list_registered_models

logger = logging.getLogger(__name__)
app = typer.Typer(help="Diagnostics for embedding storage and retrieval.")


@app.command(name="health-check", help="Verify backend connectivity and list registered models.")
def health_check(
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()

    backend = resolve_backend()
    typer.echo(f"Backend: {backend.backend_type.value} — connected.")

    # CDM connectivity is optional for the health check
    try:
        omop_cdm_engine = resolve_omop_cdm_engine()
        with omop_cdm_engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        typer.echo(f"CDM engine: {omop_cdm_engine.url} — connected.")
    except RuntimeError as exc:
        typer.echo(f"CDM engine: not configured ({exc})")
    except Exception as exc:
        typer.echo(f"CDM engine: connection failed — {exc}")

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
        typer.echo(
            f"  {r.model_name:<40} {r.provider_type:<10} {metric_str:<8} "
            f"{index_str:<6} {r.dimensions:<6} {r.storage_identifier}"
        )
        # FLAT models have metric_type=None; use COSINE for the diagnostic call
        # (FLAT accepts any backend-supported metric).
        probe_metric = r.metric_type or MetricType.COSINE
        has_emb = backend.has_any_embeddings(
            model_name=r.model_name,
            provider_type=r.provider_type,
            metric_type=probe_metric,
        )
        typer.echo(f"    embeddings present: {has_emb}")
