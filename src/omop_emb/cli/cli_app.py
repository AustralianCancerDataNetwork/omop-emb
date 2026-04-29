import typer
from . import (
    cli_embeddings as embeddings,
    cli_maintenance as maintenance,
)


app = typer.Typer(
    rich_markup_mode="rich",
    help="CLI Manager for OMOP embeddings and utilities."
)

app.add_typer(embeddings.app, name="embeddings")
app.add_typer(maintenance.app, name="maintenance")

if __name__ == "__main__":
    app()