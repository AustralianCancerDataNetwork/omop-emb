import logging
import os

import sqlalchemy as sa

from omop_emb.backends.base_backend import EmbeddingBackend
from omop_emb.config import (
    BackendType,
    ENV_CDM_DATABASE_URL,
    ENV_OMOP_EMB_BACKEND,
    build_engine_string,
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

    Reads ``OMOP_EMB_BACKEND`` (default: ``sqlitevec``) to select the backend.
    Connection details are resolved via ``build_engine_string``:

    - ``sqlitevec``: requires ``OMOP_EMB_SQLITE_PATH`` (or ``OMOP_EMB_DB_URL``).
    - ``pgvector``: requires ``OMOP_EMB_DB_USER``, ``OMOP_EMB_DB_PASSWORD``,
      ``OMOP_EMB_DB_HOST``, ``OMOP_EMB_DB_NAME`` (and optionally
      ``OMOP_EMB_DB_PORT``, ``OMOP_EMB_DB_CONN``), or ``OMOP_EMB_DB_URL``.
    """
    backend_str = os.getenv(ENV_OMOP_EMB_BACKEND, BackendType.SQLITEVEC.value).lower()

    try:
        backend_type = BackendType(backend_str)
    except ValueError:
        raise RuntimeError(
            f"Unknown backend {backend_str!r} in {ENV_OMOP_EMB_BACKEND}. "
            f"Supported: {[b.value for b in BackendType]}."
        )

    url = build_engine_string(backend_type)

    if backend_type == BackendType.SQLITEVEC:
        from omop_emb.backends.sqlitevec import SQLiteVecBackend
        assert url.database is not None  # guaranteed by build_engine_string
        return SQLiteVecBackend.from_path(url.database)

    if backend_type == BackendType.PGVECTOR:
        engine = sa.create_engine(url, future=True, echo=False)
        if engine.dialect.name != "postgresql":
            raise RuntimeError(
                "The resolved URL must point to a PostgreSQL database "
                f"(pgvector extension required), got dialect: {engine.dialect.name!r}."
            )
        try:
            from omop_emb.backends.pgvector import PGVectorEmbeddingBackend
        except ImportError as exc:
            raise RuntimeError(
                "pgvector backend is not installed. "
                "Install it with: pip install omop-emb[pgvector]"
            ) from exc
        return PGVectorEmbeddingBackend(emb_engine=engine)

    raise RuntimeError(f"Implementation for {backend_type.value} is not available.")


def resolve_omop_cdm_engine() -> sa.Engine:
    """CDM engine — any SQLAlchemy dialect, used read-only."""
    url = os.getenv(ENV_CDM_DATABASE_URL)
    if url is None:
        raise RuntimeError(
            f"{ENV_CDM_DATABASE_URL} is not set. "
            "Set it to the connection URL of your OMOP CDM database."
        )
    return sa.create_engine(url, future=True, echo=False)
