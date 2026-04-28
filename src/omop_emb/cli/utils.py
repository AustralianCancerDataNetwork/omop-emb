import logging
import os
import sqlalchemy as sa
from typing import Optional
from omop_emb.config import BackendType, ProviderType
from omop_emb.interface import EmbeddingReaderInterface


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

def resolve_engine() -> sa.Engine:
    engine_string = os.getenv('OMOP_DATABASE_URL')
    if engine_string is None:
        raise RuntimeError("OMOP_DATABASE_URL environment variable not set. Please set it in your .env file to point to your database.")

    engine = sa.create_engine(engine_string, future=True, echo=False)
    if engine.dialect.name != "postgresql":
        raise RuntimeError("Only PostgreSQL databases are supported for embedding storage with the current backends. Please check your `OMOP_DATABASE_URL` environment variable and ensure it points to a PostgreSQL database.")
    return engine

def build_pgvector_reader(
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