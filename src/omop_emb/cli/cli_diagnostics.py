from typing import (
    Annotated,
    Optional
)
from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)
import typer
app = typer.Typer(help="Diagnostics for embedding storage and retrieval")

from omop_emb.interface import (
    EmbeddingReaderInterface,
)
from omop_emb.config import (
    BackendType,
    ProviderType,
    IndexType,
)
from omop_emb.model_registry.model_registry_manager import ModelRegistryManager
from omop_emb.backends.base_backend import EmbeddingBackend
from .utils import configure_logging_level, resolve_omop_cdm_engine

@app.command(help="Check the health of the embedding storage and retrieval system")
def health_check(
    registry_db_name: Annotated[str, typer.Option(
        "--registry-db-name",
        help="Filename for the local metadata registry SQLite database. Defaults to 'metadata.db'.",
    )] = "metadata.db",
    storage_base_dir: Annotated[Optional[str], typer.Option(
        "--storage-base-dir",
        help="Optional base directory for omop-emb metadata registry (metadata.db). Reverts to `OMOP_EMB_BASE_STORAGE_DIR` if not provided, or defaults to ./.omop_emb in the current working directory.",
        rich_help_panel="Storage Options",
    )] = None,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)"
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()
    omop_cdm_engine = resolve_omop_cdm_engine()

    # This is to circumvent the interface, which expects an index and backend type
    resolved_storage_dir = str(EmbeddingBackend._resolve_storage_path(storage_base_dir))
    model_registry_manager = ModelRegistryManager(
        base_dir=resolved_storage_dir,
        db_file=registry_db_name,
    )

    for registered_model in model_registry_manager.get_registered_models_from_db():
        logger.info(f"Registered model: {registered_model}")

        reader = EmbeddingReaderInterface(
            omop_cdm_engine=omop_cdm_engine,
            backend_name_or_type=registered_model.backend_type,
            provider_name_or_type=registered_model.provider_type,
            canonical_model_name=registered_model.model_name,
            storage_base_dir=resolved_storage_dir,
            registry_db_name=registry_db_name
        )

        # We can assume the model is registered

