from typing import Annotated

import typer

from omop_emb.config import configure_logging
from . import (
    cli_embeddings as embeddings,
    cli_legacy as legacy,
    cli_maintenance as maintenance,
    cli_diagnostics as diagnostics,
)

app = typer.Typer(
    rich_markup_mode="rich",
    help="CLI Manager for OMOP embeddings and utilities."
)


@app.callback()
def _main(
    verbose: Annotated[
        int,
        typer.Option("--verbose", "-v", count=True, help="Increase log verbosity (-v INFO, -vv DEBUG)."),
    ] = 0,
) -> None:
    configure_logging(verbosity=verbose)


app.add_typer(embeddings.app, name="embeddings")
app.add_typer(maintenance.app, name="maintenance")
app.add_typer(diagnostics.app, name="diagnostics")
app.add_typer(legacy.app, name="legacy")

if __name__ == "__main__":
    app()
