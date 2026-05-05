import logging
import os

import sqlalchemy as sa

from omop_emb.backends.base_backend import EmbeddingBackend
from omop_emb.config import (
    BackendType,
    ENV_CDM_DATABASE_URL,
    ENV_EMB_POSTGRES_URL,
    ENV_EMB_SQLITE_PATH,
    ENV_OMOP_EMB_BACKEND,
)


def configure_logging_level(verbosity: int) -> None:
    level_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    log_level = level_map.get(min(verbosity, 2), logging.DEBUG)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def resolve_backend() -> EmbeddingBackend:
    """Return the configured embedding backend.

    Reads OMOP_EMB_BACKEND (default: sqlitevec) to select the backend, then
    OMOP_EMB_SQLITE_PATH (sqlite-vec) or OMOP_EMB_POSTGRES_URL (pgvector) for
    the connection details.
    """
    backend_str = os.getenv(ENV_OMOP_EMB_BACKEND, BackendType.SQLITEVEC.value).lower()

    if backend_str == BackendType.SQLITEVEC.value:
        db_path = os.getenv(ENV_EMB_SQLITE_PATH)
        if db_path is None:
            raise RuntimeError(
                f"{ENV_EMB_SQLITE_PATH} is not set. "
                "Set it to the path of your sqlite-vec database file "
                "(e.g. OMOP_EMB_SQLITE_PATH=/data/omop_emb.db). "
                f"Use {ENV_OMOP_EMB_BACKEND}=pgvector to switch to the PostgreSQL backend."
            )
        from omop_emb.backends.sqlitevec import SQLiteVecBackend
        return SQLiteVecBackend.from_path(db_path)

    if backend_str == BackendType.PGVECTOR.value:
        url = os.getenv(ENV_EMB_POSTGRES_URL)
        if url is None:
            raise RuntimeError(
                f"{ENV_EMB_POSTGRES_URL} is not set. "
                "Set it to the connection URL of your pgvector Postgres instance "
                "(e.g. postgresql://user:pass@localhost:5433/omop_emb)."
            )
        engine = sa.create_engine(url, future=True, echo=False)
        if engine.dialect.name != "postgresql":
            raise RuntimeError(
                f"{ENV_EMB_POSTGRES_URL} must point to a PostgreSQL database "
                "(pgvector extension required)."
            )
        try:
            from omop_emb.backends.pgvector import PGVectorEmbeddingBackend
        except ImportError as exc:
            raise RuntimeError(
                "pgvector backend is not installed. "
                "Install it with: pip install omop-emb[pgvector]"
            ) from exc
        return PGVectorEmbeddingBackend(emb_engine=engine)

    raise RuntimeError(
        f"Unknown backend {backend_str!r} in {ENV_OMOP_EMB_BACKEND}. "
        f"Supported: {[b.value for b in BackendType]}."
    )


def resolve_omop_cdm_engine() -> sa.Engine:
    """CDM engine — any SQLAlchemy dialect, used read-only."""
    url = os.getenv(ENV_CDM_DATABASE_URL)
    if url is None:
        raise RuntimeError(
            f"{ENV_CDM_DATABASE_URL} is not set. "
            "Set it to the connection URL of your OMOP CDM database."
        )
    return sa.create_engine(url, future=True, echo=False)
