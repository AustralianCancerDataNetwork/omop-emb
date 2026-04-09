from omop_llm import LLMClient

import sqlalchemy as sa
from sqlalchemy.orm import Session

from orm_loader.helpers import get_logger, configure_logging, create_db

from typing import Annotated, Optional
from pathlib import Path
import csv
import json
import os
from dotenv import load_dotenv
from tqdm import tqdm
import typer

from omop_emb.utils.embedding_utils import EmbeddingConceptFilter
from omop_emb.interface import EmbeddingInterface
from omop_emb.config import IndexType, BackendType

app = typer.Typer()
logger = get_logger(__name__)

SNAPSHOT_MANIFEST_NAME = "manifest.json"


def _resolve_engine() -> sa.Engine:
    engine_string = os.getenv('OMOP_DATABASE_URL')
    if engine_string is None:
        raise RuntimeError("OMOP_DATABASE_URL environment variable not set. Please set it in your .env file to point to your database.")

    engine = sa.create_engine(engine_string, future=True, echo=False)
    assert engine.dialect.name == "postgresql", "Only PostgreSQL databases are supported for embedding storage with the current backends. Please check your `OMOP_DATABASE_URL` environment variable and ensure it points to a PostgreSQL database."
    return engine


def _build_pgvector_interface(storage_base_dir: Optional[str]) -> EmbeddingInterface:
    interface = EmbeddingInterface.from_backend_name(
        backend_name=BackendType.PGVECTOR,
        storage_base_dir=storage_base_dir,
    )
    if interface.backend.backend_type != BackendType.PGVECTOR:
        raise RuntimeError("Resolved embedding backend is not pgvector. Set --storage-base-dir and/or OMOP_EMB_BACKEND appropriately.")
    return interface


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
    num_embeddings: Annotated[Optional[int], typer.Option(
        "--num-embeddings", "-n",
        help="If set, limits the number of concepts for which embeddings are generated. Useful for testing and development to speed up the embedding generation step.")] = None,
):
    configure_logging()
    load_dotenv()

    engine = _resolve_engine()

    interface = EmbeddingInterface.from_backend_name(
        backend_name=backend_name,
        storage_base_dir=storage_base_dir,
        embedding_client=LLMClient(
            model=model,
            api_base=api_base,
            api_key=api_key,
            embedding_batch_size=batch_size
        ),
    )
    embedding_dim = interface.embedding_dim
    assert embedding_dim is not None, "Embedding dimensions could not be determined from the embedding client."

    concept_filter = EmbeddingConceptFilter(
        require_standard=standard_only,
        vocabularies=tuple(vocabularies) if vocabularies else None,
    )
    
    # Ensure OMOP metadata tables exist, then initialize the embedding store.
    create_db(engine)
    interface.initialise_store(engine)

    with Session(engine) as reader, Session(engine) as writer:
        total_concepts = num_embeddings or interface.get_concepts_without_embedding_count(
            session=reader,
            model_name=model,
            concept_filter=concept_filter,
            index_type=index_type
        )
        concepts_without_embedding = interface.q_get_concepts_without_embedding(
            model_name=model,
            concept_filter=concept_filter,
            limit=num_embeddings,
            index_type=index_type
        )

        logger.info(f"Total concepts to process: {total_concepts}")
        with tqdm(total=total_concepts, desc="Processing", unit="concept") as pbar:
            result = reader.execute(concepts_without_embedding)
            
            for row_chunk in result.partitions(batch_size):
                batch_concepts = {row.concept_id: row.concept_name for row in row_chunk}
                
                interface.embed_and_upsert_concepts(
                    session=writer,
                    model_name=model,
                    concept_ids=tuple(batch_concepts.keys()),
                    concept_texts=tuple(batch_concepts.values()),
                    batch_size=batch_size,
                    index_type=index_type
                )
                
                pbar.update(len(batch_concepts))

    logger.info("Completed embedding generation and storage.")


@app.command()
def export_pgvector(
    output_dir: Annotated[str, typer.Option(
        "--output-dir", "-o",
        help="Directory where pgvector snapshot files (CSV + manifest) are written.",
    )],
    storage_base_dir: Annotated[Optional[str], typer.Option(
        "--storage-base-dir",
        help="Optional base directory for omop-emb metadata registry (metadata.db). Reverts to `OMOP_EMB_BASE_STORAGE_DIR` if not provided, or defaults to ./.omop_emb in the current working directory. Paths with `~` are expanded.",
    )] = None,
    model: Annotated[Optional[list[str]], typer.Option(
        "--model", "-m",
        help="Optional model-name filter. Repeat to export specific models only.",
    )] = None,
    index_type: Annotated[Optional[IndexType], typer.Option(
        "--index-type",
        help="Optional index-type filter.",
    )] = None,
):
    """Export pgvector embedding tables to file for checkpoint/restore workflows."""
    configure_logging()
    load_dotenv()

    engine = _resolve_engine()
    interface = _build_pgvector_interface(storage_base_dir=storage_base_dir)
    interface.initialise_store(engine)

    records = interface.backend.embedding_model_registry.get_registered_models_from_db(
        backend_type=BackendType.PGVECTOR
    )
    if records is None:
        raise RuntimeError("No pgvector models found in local registry metadata. Nothing to export.")

    model_filter = set(model) if model else None
    selected_records = [
        record for record in records
        if (model_filter is None or record.model_name in model_filter)
        and (index_type is None or record.index_type == index_type)
    ]
    if not selected_records:
        raise RuntimeError("No pgvector models matched the provided filters.")

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format_version": 1,
        "backend": BackendType.PGVECTOR.value,
        "tables": [],
    }

    with engine.connect() as conn:
        for record in selected_records:
            table_name = record.storage_identifier
            quoted_table_name = f'"{table_name}"'
            row_count = conn.execute(sa.text(f"SELECT COUNT(*) FROM {quoted_table_name}"))
            n_rows = int(row_count.scalar_one())

            csv_filename = f"{table_name}.csv"
            csv_path = output_path / csv_filename

            query = sa.text(
                f"SELECT concept_id, embedding::text AS embedding FROM {quoted_table_name} ORDER BY concept_id"
            )
            result = conn.execution_options(stream_results=True).execute(query)

            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["concept_id", "embedding"])
                for row in result:
                    writer.writerow([int(row.concept_id), str(row.embedding)])

            logger.info(f"Exported {n_rows} rows from {table_name} to {csv_path}")
            manifest["tables"].append(
                {
                    "model_name": record.model_name,
                    "index_type": record.index_type.value,
                    "dimensions": record.dimensions,
                    "storage_identifier": record.storage_identifier,
                    "metadata": dict(record.metadata),
                    "rows": n_rows,
                    "file": csv_filename,
                }
            )

    manifest_path = output_path / SNAPSHOT_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Wrote pgvector snapshot manifest to {manifest_path}")


@app.command()
def import_pgvector(
    input_dir: Annotated[str, typer.Option(
        "--input-dir", "-i",
        help="Directory containing pgvector snapshot files and manifest.json.",
    )],
    storage_base_dir: Annotated[Optional[str], typer.Option(
        "--storage-base-dir",
        help="Optional base directory for omop-emb metadata registry (metadata.db). Reverts to `OMOP_EMB_BASE_STORAGE_DIR` if not provided, or defaults to ./.omop_emb in the current working directory. Paths with `~` are expanded.",
    )] = None,
    replace: Annotated[bool, typer.Option(
        "--replace",
        help="If set, truncate each destination embedding table before import.",
    )] = False,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Number of rows per INSERT batch.",
    )] = 5000,
):
    """Import pgvector embedding tables from a previously exported snapshot."""
    configure_logging()
    load_dotenv()

    input_path = Path(input_dir).resolve()
    manifest_path = input_path / SNAPSHOT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Snapshot manifest not found at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("backend") != BackendType.PGVECTOR.value:
        raise ValueError("Snapshot manifest backend is not pgvector.")

    table_specs = manifest.get("tables")
    if not isinstance(table_specs, list) or not table_specs:
        raise ValueError("Snapshot manifest does not contain any tables.")

    engine = _resolve_engine()
    interface = _build_pgvector_interface(storage_base_dir=storage_base_dir)

    for table_spec in table_specs:
        model_name = str(table_spec["model_name"])
        index_type_enum = IndexType(str(table_spec["index_type"]))
        dimensions = int(table_spec["dimensions"])
        expected_table_name = str(table_spec["storage_identifier"])
        metadata = table_spec.get("metadata") or {}

        interface.register_model(
            engine=engine,
            model_name=model_name,
            dimensions=dimensions,
            index_type=index_type_enum,
            metadata=metadata,
        )
        actual_table_name = interface.get_model_table_name(
            model_name=model_name,
            index_type=index_type_enum,
        )
        if actual_table_name != expected_table_name:
            raise RuntimeError(
                f"Storage identifier mismatch for model '{model_name}': expected '{expected_table_name}', got '{actual_table_name}'."
            )

    interface.initialise_store(engine)

    with Session(engine) as session:
        for table_spec in table_specs:
            table_name = str(table_spec["storage_identifier"])
            csv_file = input_path / str(table_spec["file"])
            if not csv_file.is_file():
                raise FileNotFoundError(f"Missing snapshot table file: {csv_file}")

            quoted_table_name = f'"{table_name}"'
            if replace:
                session.execute(sa.text(f"TRUNCATE TABLE {quoted_table_name}"))
                session.commit()

            insert_sql = sa.text(
                f"""
                INSERT INTO {quoted_table_name} (concept_id, embedding)
                VALUES (:concept_id, CAST(:embedding AS vector))
                ON CONFLICT (concept_id) DO UPDATE
                SET embedding = EXCLUDED.embedding
                """
            )

            total_rows = 0
            batch: list[dict[str, object]] = []
            with csv_file.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    batch.append(
                        {
                            "concept_id": int(row["concept_id"]),
                            "embedding": row["embedding"],
                        }
                    )
                    if len(batch) >= batch_size:
                        session.execute(insert_sql, batch)
                        session.commit()
                        total_rows += len(batch)
                        batch = []

                if batch:
                    session.execute(insert_sql, batch)
                    session.commit()
                    total_rows += len(batch)

            logger.info(f"Imported {total_rows} rows into {table_name} from {csv_file}")


if __name__ == "__main__":
    app()
