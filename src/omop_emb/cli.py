import sqlalchemy as sa
from sqlalchemy.orm import Session

from orm_loader.helpers import get_logger, create_db

from typing import Annotated, Optional
from pathlib import Path
import csv
import json
import os
import logging
from dotenv import load_dotenv
from tqdm import tqdm
import typer

from omop_emb.embeddings import EmbeddingClient
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter
from omop_emb.interface import (
    EmbeddingWriterInterface, 
    EmbeddingReaderInterface, 
    migrate_legacy_registry_row,
)
from omop_emb.config import IndexType, BackendType, ProviderType

app = typer.Typer()
logger = get_logger(__name__)

SNAPSHOT_MANIFEST_NAME = "manifest.json"


def configure_logging_level(verbosity: int) -> None:
    """Configure global logging based on CLI verbosity flags."""
    level_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    log_level = level_map.get(min(verbosity, 2), logging.DEBUG)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


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


def _resolve_engine() -> sa.Engine:
    engine_string = os.getenv('OMOP_DATABASE_URL')
    if engine_string is None:
        raise RuntimeError("OMOP_DATABASE_URL environment variable not set. Please set it in your .env file to point to your database.")

    engine = sa.create_engine(engine_string, future=True, echo=False)
    if engine.dialect.name != "postgresql":
        raise RuntimeError("Only PostgreSQL databases are supported for embedding storage with the current backends. Please check your `OMOP_DATABASE_URL` environment variable and ensure it points to a PostgreSQL database.")
    return engine


def _build_pgvector_reader(
    canonical_model_name: str,
    storage_base_dir: Optional[str], 
    provider_type: ProviderType = ProviderType.OLLAMA
) -> EmbeddingReaderInterface:
    reader = EmbeddingReaderInterface(
        canonical_model_name=canonical_model_name,
        backend_name_or_type=BackendType.PGVECTOR,
        provider_name_or_type=provider_type,
        storage_base_dir=storage_base_dir,
    )
    if reader.backend_type != BackendType.PGVECTOR:
        raise RuntimeError("Resolved embedding backend is not pgvector. Set --storage-base-dir and/or OMOP_EMB_BACKEND appropriately.")
    return reader


@app.command()
def add_embeddings(
    api_base: Annotated[str, typer.Option(
        "--api-base",
        help="Base URL for the API to use for generating embeddings.",
    )],
    api_key: Annotated[str, typer.Option(
        "--api-key",
        help="API key for the embedding API.")],
    index_type: Annotated[IndexType, typer.Option(
        "--index-type",
        help="Backend-specific index type for newly registered models and how it should be stored.",
    )] = IndexType.FLAT,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size to use when generating and inserting embeddings. Adjust based on your system's memory capacity.")] = 100,
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Name of the embedding model to use for generating concept embeddings (e.g., 'text-embedding-3-small'). If not provided, embeddings will not be generated.")] = "text-embedding-3-small",
    backend_name: Annotated[Optional[str], typer.Option(
        "--backend",
        help="Embedding backend to use. Can be replaced by the `OMOP_EMB_BACKEND` environment variable."
    )] = None,
    storage_base_dir: Annotated[Optional[str], typer.Option(
        "--storage-base-dir",
        help="Optional base directory for embedding backend storage. Reverts to `OMOP_EMB_BASE_STORAGE_DIR` if not provided, or defaults to ./.omop_emb in the current working directory. Paths with `~` are expanded."
    )] = None,
    standard_only: Annotated[bool, typer.Option(
        "--standard-only",
        help="If set, only generate embeddings for OMOP standard concepts (standard_concept = 'S')."
    )] = False,
    vocabularies: Annotated[Optional[list[str]], typer.Option(
        "--vocabulary",
        help="Optional vocabulary filter. Repeat the option to embed concepts only from specific OMOP vocabularies."
    )] = None,
    domains: Annotated[Optional[list[str]], typer.Option(
        "--domain",
        help="Optional domain filter. Repeat the option to embed concepts only from specific OMOP domains."
    )] = None,
    num_embeddings: Annotated[Optional[int], typer.Option(
        "--num-embeddings", "-n",
        help="If set, limits the number of concepts for which embeddings are generated. Useful for testing and development to speed up the embedding generation step.")] = None,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)"
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()

    engine = _resolve_engine()

    embedding_client = EmbeddingClient(
        model=model,
        api_base=api_base,
        api_key=api_key,
        embedding_batch_size=batch_size
    )
    embedding_writer = EmbeddingWriterInterface(
        embedding_client=embedding_client,
        backend_name_or_type=backend_name,
        storage_base_dir=storage_base_dir,
    )

    concept_filter = EmbeddingConceptFilter(
        require_standard=standard_only,
        domains=tuple(domains) if domains else None,
        vocabularies=tuple(vocabularies) if vocabularies else None,
    )

    # Ensure OMOP metadata tables exist, then initialize the embedding store.
    create_db(engine)
    embedding_writer.register_model(
        engine=engine,
        index_config=index_config,
    )

    with Session(engine) as reader, Session(engine) as writer:
        total_concepts_missing_concepts = embedding_writer.get_concepts_without_embedding_count(
            session=reader,
            concept_filter=concept_filter,
            index_type=index_type,
        )
        total_concepts = min(total_concepts_missing_concepts, num_embeddings) if num_embeddings is not None else total_concepts_missing_concepts

        concepts_without_embedding = embedding_writer.q_get_concepts_without_embedding(
            concept_filter=concept_filter,
            limit=total_concepts,
            index_type=index_type
        )

        logger.info(f"Total concepts to process: {total_concepts}")
        with tqdm(total=total_concepts, desc="Processing", unit="concept") as pbar:
            result = reader.execute(concepts_without_embedding)

            for row_chunk in result.partitions(batch_size):
                batch_concepts = {row.concept_id: row.concept_name for row in row_chunk}

                embedding_writer.embed_and_upsert_concepts(
                    session=writer,
                    concept_ids=tuple(batch_concepts.keys()),
                    concept_texts=tuple(batch_concepts.values()),
                    batch_size=batch_size,
                    index_type=index_type
                )

                pbar.update(len(batch_concepts))

    logger.info("Completed embedding generation and storage.")

# TODO: Import and Export routines
# Requires common format for exporting bits and pieces in the respective storage backends

@app.command()
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

    source_url = source_database_url or os.getenv("OMOP_DATABASE_URL")
    if source_url is None:
        raise RuntimeError("OMOP_DATABASE_URL is not set. Provide --source-database-url.")

    source_engine = sa.create_engine(source_url, future=True, echo=False)

    legacy_rows = _load_legacy_rows(source_engine, legacy_table=legacy_table)
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
        with source_engine.begin() as conn:
            conn.execute(sa.text(f'DROP TABLE IF EXISTS "{legacy_table}"'))
        logger.info(f"Dropped legacy registry table '{legacy_table}'.")


if __name__ == "__main__":
    app()
