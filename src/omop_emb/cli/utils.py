import logging
import os
from typing import Union, Optional

import sqlalchemy as sa

from omop_emb.backends.base_backend import EmbeddingBackend
from omop_emb.config import (
    BackendType,
    ENV_OMOP_CDM_DATABASE_URL,
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


def resolve_omop_cdm_engine() -> sa.Engine:
    """Resolve CDM engine with any SQLAlchemy dialect, used read-only."""
    url = os.getenv(ENV_OMOP_CDM_DATABASE_URL)
    if url is None:
        raise RuntimeError(
            f"{ENV_OMOP_CDM_DATABASE_URL} is not set. "
            "Set it to the connection URL of your OMOP CDM database."
        )
    return sa.create_engine(url, future=True, echo=False)
