"""Diagnostics for omop-emb CLI."""

from typing import Annotated
from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)
import sqlalchemy as sa
import typer
app = typer.Typer(help="Diagnostics for embedding storage and retrieval.")

from .utils import configure_logging_level, resolve_omop_cdm_engine, resolve_emb_engine
from omop_emb.interface import list_registered_models, EmbeddingReaderInterface
from omop_emb.config import IndexType


@app.command(name="health-check", help="List all registered models and verify connectivity to both engines.")
def health_check(
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()

    emb_engine = resolve_emb_engine()
    omop_cdm_engine = resolve_omop_cdm_engine()

    with emb_engine.connect() as conn:
        conn.execute(sa.text("SELECT 1"))
    typer.echo(f"EMB engine: {emb_engine.url} — connected.")

    with omop_cdm_engine.connect() as conn:
        conn.execute(sa.text("SELECT 1"))
    typer.echo(f"CDM engine: {omop_cdm_engine.url} — connected.")

    records = list_registered_models(emb_engine=emb_engine)
    if not records:
        typer.echo("No registered models found in the pgvector registry.")
        return

    typer.echo(f"\n{len(records)} registered model(s):")
    typer.echo(f"  {'Model':<40} {'Provider':<10} {'Index':<6} {'Dims':<6} {'Table'}")
    typer.echo("  " + "-" * 85)
    for r in records:
        typer.echo(
            f"  {r.model_name:<40} {r.provider_type:<10} {r.index_type:<6} "
            f"{r.dimensions:<6} {r.storage_identifier}"
        )

        reader = EmbeddingReaderInterface(
            emb_engine=emb_engine,
            omop_cdm_engine=omop_cdm_engine,
            canonical_model_name=r.model_name,
            provider_name_or_type=r.provider_type,
        )
        try:
            idx = IndexType(r.index_type)
        except ValueError:
            typer.echo(f"    [WARN] Unknown index_type '{r.index_type}' — skipping table check.")
            continue

        has_emb = reader.has_any_embeddings(index_type=idx)
        typer.echo(f"    embeddings present: {has_emb}")
