import typer
from . import (
    cli_embeddings as embeddings,
    cli_maintenance as maintenance,
    cli_diagnostics as diagnostics,
)


app = typer.Typer(
    rich_markup_mode="rich",
    help="CLI Manager for OMOP embeddings and utilities."
)

app.add_typer(embeddings.app, name="embeddings")
app.add_typer(maintenance.app, name="maintenance")
app.add_typer(diagnostics.app, name="diagnostics")

if __name__ == "__main__":
    app()
