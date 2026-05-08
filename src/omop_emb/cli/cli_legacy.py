"""Legacy CLI commands for backward compatibility. These are just to support older embeddings."""

import logging
from typing import Annotated

import numpy as np
import typer
from dotenv import load_dotenv
from tqdm import tqdm

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from .utils import configure_logging_level, resolve_backend
from omop_emb.backends.index_config import FlatIndexConfig
from omop_emb.config import MetricType, ProviderType
from omop_emb.interface import _fetch_cdm_concepts_for_ingestion
from omop_emb.utils.embedding_utils import ConceptEmbeddingRecord

logger = logging.getLogger(__name__)
app = typer.Typer(help="Legacy commands for importing pre-built embeddings.")


@app.command(name="add-embeddings-from-h5", help="Ingest raw embeddings from an HDF5 file into the embedding store.")
def add_embeddings_from_h5(
    h5_file: Annotated[str, typer.Option(
        "--h5-file",
        help="Path to the HDF5 file. Must contain 'concept_id' and 'embedding' datasets.",
    )],
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Canonical model name to register the embeddings under (e.g. 'nomic-embed-text:v1.5').",
    )],
    omop_cdm_db_url: Annotated[str, typer.Option(
        "--omop-cdm-db-url",
        help="Database URL for the OMOP CDM instance (e.g. postgresql://user:pass@host:port/db). Required to validate concept IDs and populate metadata.",
    )],
    provider_type: Annotated[ProviderType, typer.Option(
        "--provider-type",
        help="Embedding provider that produced these embeddings.",
    )] = ProviderType.OLLAMA,
    metric_type: Annotated[MetricType, typer.Option(
        "--metric-type",
        help="Distance metric to use when storing and searching embeddings.",
    )] = MetricType.COSINE,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Number of embeddings written to the backend per batch.",
    )] = 10_000,
    cdm_batch_size: Annotated[int, typer.Option(
        "--cdm-batch-size",
        help="Batch size for fetching concept metadata from the CDM during ingestion. Adjust if you encounter performance issues or database limits during ingestion.",
    )] = 50_000,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    """Ingest pre-built embeddings from an HDF5 file into the configured backend.

    The HDF5 file must contain two datasets:

    \b
      concept_id   — 1-D integer array of OMOP concept IDs
      embedding    — 2-D float array of shape (N, dimensions)

    Concept metadata (domain_id, vocabulary_id, is_standard, is_valid) is
    fetched from the OMOP CDM per batch and stored alongside the embeddings.
    Concept IDs not found in the CDM are stored with empty metadata.
    """
    configure_logging_level(verbosity)
    load_dotenv()

    try:
        import h5py
    except ImportError:
        typer.echo(
            "h5py is required for this command. Install it with: pip install h5py",
            err=True,
        )
        raise typer.Exit(1)

    from pathlib import Path
    h5_path = Path(h5_file).expanduser().resolve()
    if not h5_path.exists():
        typer.echo(f"HDF5 file not found: {h5_path}", err=True)
        raise typer.Exit(1)

    with h5py.File(h5_path, "r") as f:
        for _ds_name in ("concept_ids", "embeddings"):
            if _ds_name not in f or not isinstance(f[_ds_name], h5py.Dataset):
                typer.echo(f"HDF5 file is missing required dataset '{_ds_name}'.", err=True)
                raise typer.Exit(1)

        cid_ds = f["concept_ids"]
        emb_ds = f["embeddings"]
        assert isinstance(cid_ds, h5py.Dataset) and isinstance(emb_ds, h5py.Dataset)

        total: int = cid_ds.shape[0]
        emb_shape = emb_ds.shape

        if total == 0:
            typer.echo("HDF5 file contains no embeddings. Nothing to do.")
            raise typer.Exit(0)

        if emb_shape[0] != total:
            typer.echo(
                f"Mismatch: concept_ids has {total} entries but embeddings has {emb_shape[0]} rows.",
                err=True,
            )
            raise typer.Exit(1)

        if len(emb_shape) != 2:
            typer.echo(
                f"Embedding shape {emb_shape} is invalid. Expected 2D array (num_concepts, dimensions).",
                err=True,
            )
            raise typer.Exit(1)

        dimensions = emb_shape[1]

        backend = resolve_backend()
        backend.register_model(
            model_name=model,
            provider_type=provider_type,
            dimensions=dimensions,
            index_config=FlatIndexConfig(),
        )
        typer.echo(f"Registered model '{model}' ({dimensions}d, metric={metric_type.value}).")

        cdm_engine = sa.create_engine(omop_cdm_db_url, future=True, echo=False)
        cdm_factory = sessionmaker(cdm_engine)

        n_batches = (total + batch_size - 1) // batch_size
        typer.echo(f"Ingesting {total:,} embeddings in {n_batches} batch(es) of {batch_size:,}...")

        ingested = 0
        for start in tqdm(range(0, total, batch_size), desc="Ingesting embeddings"):
            end = min(start + batch_size, total)
            batch_cids: np.ndarray = np.asarray(cid_ds[start:end])
            batch_emb = np.asarray(emb_ds[start:end], dtype=np.float32)
            meta = _fetch_cdm_concepts_for_ingestion(
                {int(cid) for cid in batch_cids}, cdm_factory,
                batch_size=cdm_batch_size,
            )
            records = []
            for j in range(len(batch_cids)):
                cid = int(batch_cids[j])
                row = meta.get(cid)
                records.append(ConceptEmbeddingRecord(
                    concept_id=cid,
                    domain_id=row.domain_id if row else "",
                    vocabulary_id=row.vocabulary_id if row else "",
                    is_standard=row.standard_concept in ("S", "C") if row else False,
                    is_valid=row.invalid_reason not in ("D", "U") if row else True,
                ))
            backend.upsert_embeddings(
                model_name=model,
                metric_type=metric_type,
                records=records,
                embeddings=batch_emb,
            )
            ingested += end - start
            logger.info("Ingested %d / %d embeddings.", ingested, total)

    typer.echo(f"Done. Ingested {ingested:,} embeddings for '{model}'.")
