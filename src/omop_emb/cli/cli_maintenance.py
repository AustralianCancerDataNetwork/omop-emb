"""Deprecated utilities for the command line interface
to support older versions of omop-emb.
These are not extensively tested and may be removed in future 
versions."""

from typing import (
    Annotated,
    Optional,
    cast
)

from dotenv import load_dotenv
import h5py
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, Session
import logging
logger = logging.getLogger(__name__)
import os
from tqdm import tqdm
import typer
app = typer.Typer(help="Legacy commands for omop-emb CLI. These commands are deprecated and may be removed in future versions. Use with caution.")

from .utils import configure_logging_level, resolve_omop_cdm_engine
from omop_emb.config import (
    IndexType,
    BackendType,
    ProviderType,
    MetricType,
    ENV_OMOP_EMB_BACKEND,
    SUPPORTED_INDICES_AND_METRICS_PER_BACKEND,
)
from omop_emb.interface import migrate_legacy_registry_row, EmbeddingWriterInterface
from omop_emb.embeddings import EmbeddingClient
from omop_emb.backends import get_embedding_backend
from omop_emb.backends.index_config import index_config_from_index_type
from omop_emb.backends.faiss.storage_manager import EmbeddingStorageManager

def _load_legacy_rows(engine: sa.Engine, legacy_table: str) -> list[dict[str, object]]:
    inspector = sa.inspect(engine)
    if not inspector.has_table(legacy_table):
        raise RuntimeError(f"Legacy table '{legacy_table}' does not exist in source database.")

    query = sa.text(f'SELECT * FROM "{legacy_table}"')
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    return [dict(row) for row in rows]


def _legacy_row_fields(row: dict[str, object]) -> tuple[str, int, IndexType, str, dict[str, object]]:
    model_name_raw = row.get("model_name")
    dimensions_raw = row.get("dimensions")
    if model_name_raw is None or dimensions_raw is None:
        raise ValueError("Legacy row missing required fields 'model_name' and/or 'dimensions'.")

    index_raw = row.get("index_type") or row.get("index_method") or IndexType.FLAT.value
    index_type = IndexType(str(index_raw))

    storage_identifier_raw = row.get("storage_identifier") or row.get("table_name")
    if storage_identifier_raw is None:
        raise ValueError("Legacy row missing required storage field ('storage_identifier' or 'table_name').")
    storage_identifier = str(storage_identifier_raw)

    details_raw = row.get("details") or row.get("metadata") or {}
    if isinstance(details_raw, dict):
        metadata = dict(details_raw)
    else:
        metadata = {}

    model_name = str(model_name_raw)
    dimensions = int(str(dimensions_raw))

    return model_name, dimensions, index_type, storage_identifier, metadata


@app.command(deprecated=True, help="Migrate legacy pgvector registry rows into local metadata.db registry. Use with caution.")
def migrate_legacy_pgvector_registry(
    storage_base_dir: Annotated[Optional[str], typer.Option(
        "--storage-base-dir",
        help="Optional base directory for omop-emb metadata registry (metadata.db). Reverts to `OMOP_EMB_BASE_STORAGE_DIR` if not provided, or defaults to ./.omop_emb in the current working directory. Paths with `~` are expanded.",
    )] = None,
    provider_type: Annotated[ProviderType, typer.Option(
        "--provider-type",
        help="Provider type for the migrated models.",
    )] = ProviderType.OLLAMA,
    source_database_url: Annotated[Optional[str], typer.Option(
        "--source-database-url",
        help="Source database URL containing the legacy model_registry table. Defaults to OMOP_DATABASE_URL.",
    )] = None,
    legacy_table: Annotated[str, typer.Option(
        "--legacy-table",
        help="Name of the legacy table to migrate from.",
    )] = "model_registry",
    dry_run: Annotated[bool, typer.Option(
        "--dry-run",
        help="If set, report rows that would be migrated without writing to local metadata.",
    )] = False,
    drop_legacy_registry: Annotated[bool, typer.Option(
        "--drop-legacy-registry",
        help="If set, drop the legacy table after successful migration.",
    )] = False,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)"
    )] = 0,
):
    """Migrate legacy pgvector registry rows into local metadata.db registry."""
    configure_logging_level(verbosity)
    load_dotenv()

    omop_cdm_url = source_database_url or os.getenv("OMOP_DATABASE_URL")
    if omop_cdm_url is None:
        raise RuntimeError("OMOP_DATABASE_URL is not set. Provide --source-database-url.")

    omop_cdm_engine = sa.create_engine(omop_cdm_url, future=True, echo=False)

    legacy_rows = _load_legacy_rows(omop_cdm_engine, legacy_table=legacy_table)
    if not legacy_rows:
        logger.info("No legacy registry rows found. Nothing to migrate.")
        return

    logger.info(f"Found {len(legacy_rows)} legacy registry rows in '{legacy_table}'.")

    migrated = 0
    for row in legacy_rows:
        model_name, dimensions, index_type, storage_identifier, metadata = _legacy_row_fields(row)

        if dry_run:
            logger.info(
                f"[DRY RUN] Would migrate model={model_name}, index={index_type.value}, "
                f"dimensions={dimensions}, storage_identifier={storage_identifier}"
            )
            migrated += 1
            continue

        migrate_legacy_registry_row(
            omop_cdm_engine=omop_cdm_engine,
            backend_type=BackendType.PGVECTOR,
            provider_type=provider_type,
            model_name=model_name,
            dimensions=dimensions,
            index_type=index_type,
            metadata=metadata,
            storage_identifier=storage_identifier,
            storage_base_dir=storage_base_dir,
        )
        migrated += 1

    logger.info(f"Migrated {migrated} legacy registry rows into local metadata registry.")

    if drop_legacy_registry and not dry_run:
        with omop_cdm_engine.begin() as conn:
            conn.execute(sa.text(f'DROP TABLE IF EXISTS "{legacy_table}"'))
        logger.info(f"Dropped legacy registry table '{legacy_table}'.")


@app.command(help="Load local embeddings from HDF5 files into respective embedding backends and register in local metadata registry.")
def load_embeddings_from_hdf5(
    hdf5_file: Annotated[str, typer.Option(
        "--hdf5-file",
        help="Path to the HDF5 file containing the embeddings.",
    )],
    model: Annotated[str, typer.Option(
        "--model",
        help="Name of the embedding model to register the indices under. Has to be canonical naming scheme.",
    )],
    api_base: Annotated[str, typer.Option(
        "--api-base",
        help="Base URL for the API to use for generating embeddings.",
        rich_help_panel="Embedding API Options",
    )],
    api_key: Annotated[str, typer.Option(
        "--api-key",
        help="API key for the embedding API.",
        rich_help_panel="Embedding API Options",
    )],
    index_type: Annotated[IndexType, typer.Option(
        "--index-type",
        help="Backend-specific index type for newly registered models and how it should be stored.",
        rich_help_panel="Index Options",
    )] = IndexType.FLAT,
    metric_type: Annotated[Optional[MetricType], typer.Option(
        "--metric-type",
        help="Metric type for the embeddings. Required for some index types (e.g., HNSW) and used for FAISS index configuration.",
        rich_help_panel="Index Options",
    )] = MetricType.COSINE,
    backend_type: Annotated[BackendType, typer.Option(
        "--backend",
        help=f"Embedding backend to use. Can be replaced by the `{ENV_OMOP_EMB_BACKEND}` environment variable.",
        rich_help_panel="Storage Options",
    )] = BackendType.FAISS,
    provider_type: Annotated[ProviderType, typer.Option(
        "--provider-type",
        help="Provider type for the migrated model.",
    )] = ProviderType.OLLAMA,
    storage_base_dir: Annotated[Optional[str], typer.Option(
        "--storage-base-dir",
        help="Optional base directory for omop-emb metadata registry (metadata.db). Reverts to `OMOP_EMB_BASE_STORAGE_DIR` if not provided, or defaults to ./.omop_emb in the current working directory. Paths with `~` are expanded.",
    )] = None,
    registry_db_name: Annotated[str, typer.Option(
        "--registry-db-name",
        help="Filename for the local metadata registry SQLite database. Defaults to 'metadata.db'.",
    )] = "metadata.db",
    index_hnsw_num_neighbors: Annotated[Optional[int], typer.Option(
        "--index-hnsw-num-neighbors",
        help="HNSW: number of neighbors per graph node. Required when --index-type is HNSW.",
        rich_help_panel="Index Options",
    )] = None,
    index_hnsw_ef_search: Annotated[Optional[int], typer.Option(
        "--index-hnsw-ef-search",
        help="HNSW: ef parameter controlling recall during search. Required when --index-type is HNSW.",
        rich_help_panel="Index Options",
    )] = None,
    index_hnsw_ef_construction: Annotated[Optional[int], typer.Option(
        "--index-ef-construction",
        help="HNSW: ef parameter controlling graph quality during construction. Required when --index-type is HNSW.",
        rich_help_panel="Index Options",
    )] = None,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size to use when streaming embeddings from HDF5 file and upserting into the backend.",
    )] = 500,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)"
    )] = 0,
    force_delete: Annotated[bool, typer.Option(
        "--force-delete",
        help="If set, force deletion of existing model registration and storage if a model with the same name, provider, and index type already exists in the local metadata registry. Use with caution as this will result in data loss for the existing model.",
    )] = False,
):
    configure_logging_level(verbosity=verbosity)
    load_dotenv()

    embedding_client = EmbeddingClient(
        model=model,
        api_base=api_base,
        api_key=api_key,
    )
    engine = resolve_omop_cdm_engine()
    writer = EmbeddingWriterInterface(
        omop_cdm_engine=engine,
        embedding_client=embedding_client,
        backend_name_or_type=backend_type,
        storage_base_dir=storage_base_dir,
        registry_db_name=registry_db_name,
    )

    index_kwargs = {
        'num_neighbors': index_hnsw_num_neighbors,
        'ef_search': index_hnsw_ef_search,
        'ef_construction': index_hnsw_ef_construction,
    }

    index_config = index_config_from_index_type(
        index_type,
        **index_kwargs
    )

    if writer.is_model_registered(index_type=index_type) and not force_delete:
        raise RuntimeError(f"Model '{model}' with provider '{provider_type.value}' and index type '{index_type.value}' is already registered in the local metadata registry. Please choose a different model name or clean up the existing registration before loading new indices.")

    writer.delete_model(index_type=index_type)

    # Ensure OMOP metadata tables exist, then initialize the embedding store.
    writer.register_model(
        index_config=index_config,
    )

    with h5py.File(hdf5_file, "r") as loaded_file:
        if EmbeddingStorageManager.HDF5_DATASET_NAME_EMBEDDINGS not in loaded_file:
            raise RuntimeError(f"HDF5 file '{hdf5_file}' does not contain the expected dataset '{EmbeddingStorageManager.HDF5_DATASET_NAME_EMBEDDINGS}'.")
        if EmbeddingStorageManager.HDF5_DATASET_NAME_CONCEPT_IDS not in loaded_file:
            raise RuntimeError(f"HDF5 file '{hdf5_file}' does not contain the expected dataset '{EmbeddingStorageManager.HDF5_DATASET_NAME_CONCEPT_IDS}'.")
        
        embeddings: h5py.Dataset = cast(h5py.Dataset, loaded_file[EmbeddingStorageManager.HDF5_DATASET_NAME_EMBEDDINGS])
        concept_ids: h5py.Dataset = cast(h5py.Dataset, loaded_file[EmbeddingStorageManager.HDF5_DATASET_NAME_CONCEPT_IDS])

        if len(embeddings) != len(concept_ids):
            raise RuntimeError(f"HDF5 file '{hdf5_file}' contains mismatched number of embeddings and concept IDs: {len(embeddings)} embeddings vs {len(concept_ids)} concept IDs.")
        
        dimensions = embeddings.shape[1]
        if dimensions != embedding_client.embedding_dim:
            raise RuntimeError(f"Embedding dimension in HDF5 file '{hdf5_file}' ({dimensions}) does not match expected embedding dimension for model '{model}' ({embedding_client.embedding_dim}).")
        
        
        n_batches = (len(embeddings) + batch_size - 1) // batch_size

        def _batches():
            for i in range(0, len(embeddings), batch_size):
                yield concept_ids[i : i + batch_size], embeddings[i : i + batch_size]

        writer.bulk_upsert_concept_embeddings(
            index_type=index_type,
            batches=tqdm(_batches(), total=n_batches, desc="Loading embeddings", unit="batch"),
            metric_type=metric_type,
        )

@app.command(help="Rebuild indexes for a registered model")
def rebuild_index(
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Registered embedding model name to rebuild indexes for.",
    )],
    backend_type: Annotated[BackendType, typer.Option(
        "--backend",
        help=f"Embedding backend to use. Can be replaced by the `{ENV_OMOP_EMB_BACKEND}` environment variable.",
        rich_help_panel="Storage Options",
    )] = BackendType.FAISS,
    provider_type: Annotated[ProviderType, typer.Option(
        "--provider-type",
        help="Provider type the model was registered with.",
    )] = ProviderType.OLLAMA,
    index_type: Annotated[IndexType, typer.Option(
        "--index-type",
        help="Index type the model was registered with.",
        rich_help_panel="Index Options",
    )] = IndexType.FLAT,
    storage_base_dir: Annotated[Optional[str], typer.Option(
        "--storage-base-dir",
        help="Optional base directory for omop-emb metadata registry (metadata.db). Reverts to `OMOP_EMB_BASE_STORAGE_DIR` if not provided, or defaults to ./.omop_emb in the current working directory.",
        rich_help_panel="Storage Options",
    )] = None,
    registry_db_name: Annotated[str, typer.Option(
        "--registry-db-name",
        help="Filename for the local metadata registry SQLite database.",
        rich_help_panel="Storage Options",
    )] = "metadata.db",
    metric_types: Annotated[Optional[list[MetricType]], typer.Option(
        "--metric-type",
        help="Metric(s) to rebuild. Repeat to rebuild multiple metrics. Defaults to all metrics supported by the registered index type.",
        rich_help_panel="Index Options",
    )] = None,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size when streaming embeddings from storage during rebuild. Meaningful for FAISS; ignored by pgvector.",
    )] = 100_000,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()

    engine = resolve_omop_cdm_engine()
    backend = get_embedding_backend(
        omop_cdm_engine=engine,
        backend_name_or_type=backend_type,
        storage_base_dir=storage_base_dir,
        registry_db_name=registry_db_name,
    )

    resolved_metric_types: tuple[MetricType, ...]
    if metric_types:
        resolved_metric_types = tuple(metric_types)
    else:
        resolved_metric_types = SUPPORTED_INDICES_AND_METRICS_PER_BACKEND.get(backend_type, {}).get(index_type, ())

    backend.rebuild_model_indexes(
        model,
        provider_type,
        index_type,
        engine=engine,
        metric_types=resolved_metric_types,
        batch_size=batch_size,
    )
    logger.info("Completed index rebuild for model '%s'.", model)


@app.command(help="Switch index type for a registered model.")
def switch_index_type(
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Registered embedding model name to add the new index for.",
    )],
    new_index_type: Annotated[IndexType, typer.Option(
        "--new-index-type",
        help="New index type to register and build for the model.",
        rich_help_panel="Index Options",
    )] = IndexType.HNSW,
    backend_type: Annotated[BackendType, typer.Option(
        "--backend",
        help=f"Embedding backend to use. Can be replaced by the `{ENV_OMOP_EMB_BACKEND}` environment variable.",
        rich_help_panel="Storage Options",
    )] = BackendType.FAISS,
    provider_type: Annotated[ProviderType, typer.Option(
        "--provider-type",
        help="Provider type the model was registered with.",
    )] = ProviderType.OLLAMA,
    storage_base_dir: Annotated[Optional[str], typer.Option(
        "--storage-base-dir",
        help="Optional base directory for omop-emb metadata registry (metadata.db). Reverts to `OMOP_EMB_BASE_STORAGE_DIR` if not provided, or defaults to ./.omop_emb in the current working directory.",
        rich_help_panel="Storage Options",
    )] = None,
    registry_db_name: Annotated[str, typer.Option(
        "--registry-db-name",
        help="Filename for the local metadata registry SQLite database.",
        rich_help_panel="Storage Options",
    )] = "metadata.db",
    metric_types: Annotated[Optional[list[MetricType]], typer.Option(
        "--metric-type",
        help="Metric(s) to build after registering. Defaults to all metrics supported by the new index type.",
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
        help="Batch size when streaming embeddings from storage during rebuild. Meaningful for FAISS; ignored by pgvector.",
    )] = 100_000,
    rebuild: Annotated[bool, typer.Option(
        "--rebuild/--no-rebuild",
        help="Build the new index immediately after registration. Pass --no-rebuild to defer.",
    )] = True,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()

    omop_cdm_engine = resolve_omop_cdm_engine()
    backend = get_embedding_backend(
        omop_cdm_engine=omop_cdm_engine,
        backend_name_or_type=backend_type,
        storage_base_dir=storage_base_dir,
        registry_db_name=registry_db_name,
    )

    existing = backend.get_registered_models(model_name=model, provider_type=provider_type)
    if not existing:
        raise RuntimeError(
            f"No registration found for model '{model}' with provider '{provider_type.value}'. "
            "Register the model first via the main embedding CLI."
        )

    new_index_config = index_config_from_index_type(
        new_index_type,
        num_neighbors=index_hnsw_num_neighbors,
        ef_search=index_hnsw_ef_search,
        ef_construction=index_hnsw_ef_construction,
    )
    backend.register_model(
        model_name=model,
        dimensions=existing[0].dimensions,
        provider_type=provider_type,
        index_config=new_index_config,
    )
    logger.info(f"Registered model '{model}' with index_type={new_index_type.value}.")

    if rebuild:
        resolved_metric_types: tuple[MetricType, ...]
        if metric_types:
            resolved_metric_types = tuple(metric_types)
        else:
            resolved_metric_types = SUPPORTED_INDICES_AND_METRICS_PER_BACKEND.get(
                backend_type, {}
            ).get(new_index_type, ())
        backend.rebuild_model_indexes(
            model_name=model,
            provider_type=provider_type,
            index_type=new_index_type,
            metric_types=resolved_metric_types,
            batch_size=batch_size,
        )

    logger.info(
        f"Registered model '{model}' with index_type={new_index_type.value}."
        + (" Built indexes." if rebuild else "")
    )