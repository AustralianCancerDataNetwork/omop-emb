import logging
import os
import sqlalchemy as sa

from omop_emb.config import ENV_EMB_POSTGRES_URL, ENV_CDM_DATABASE_URL


def configure_logging_level(verbosity: int) -> None:
    level_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    log_level = level_map.get(min(verbosity, 2), logging.DEBUG)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def resolve_emb_engine() -> sa.Engine:
    """Embedding engine — must point to the dedicated pgvector Postgres instance."""
    url = os.getenv(ENV_EMB_POSTGRES_URL)
    if url is None:
        raise RuntimeError(
            f"{ENV_EMB_POSTGRES_URL} environment variable not set. "
            "Set it to the connection URL of your dedicated pgvector Postgres instance "
            "(e.g. postgresql://omop_emb:omop_emb@localhost:5433/omop_emb)."
        )
    engine = sa.create_engine(url, future=True, echo=False)
    if engine.dialect.name != "postgresql":
        raise RuntimeError(
            f"{ENV_EMB_POSTGRES_URL} must point to a PostgreSQL database "
            "(pgvector extension required)."
        )
    return engine


def resolve_omop_cdm_engine() -> sa.Engine:
    """CDM engine — any SQLAlchemy dialect, used read-only."""
    url = os.getenv(ENV_CDM_DATABASE_URL)
    if url is None:
        raise RuntimeError(
            f"{ENV_CDM_DATABASE_URL} environment variable not set. "
            "Set it to the connection URL of your OMOP CDM database."
        )
    return sa.create_engine(url, future=True, echo=False)
