"""OMOP CDM utilities. Since embeddings and OMOP CDM are separate, we have this helper utility in case we need to query the OMOP CDM."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Iterator, Optional

from sqlalchemy import Engine, Row, Select, select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, sessionmaker

from omop_alchemy.cdm.model.vocabulary import Concept
from omop_emb.utils.embedding_utils import CDMConceptFilter

logger = logging.getLogger(__name__)


def streamed(stmt: Select, batch_size: int) -> Select:
    """Return *stmt* with server-side cursor streaming enabled.
    """
    return stmt.execution_options(stream_results=True, yield_per=batch_size)


def concept_embedding_projection() -> Select:
    """Select CDM fields and canonical derived flags used for embeddings."""

    return select(
        Concept.concept_id,
        Concept.concept_name,
        Concept.domain_id,
        Concept.vocabulary_id,
        Concept.standard_concept,
        Concept.invalid_reason,
        Concept.is_standard_expr().label("is_standard"),
        Concept.is_valid_expr().label("is_valid"),
    )


@contextmanager
def cdm_session(cdm_engine: Engine) -> Generator[Session, None, None]:
    """Context manager yielding a single CDM session from *cdm_engine*."""
    with sessionmaker(cdm_engine)() as session:
        yield session


def check_concept_cdm(cdm_engine: Engine) -> None:
    """Verify the OMOP CDM Concept table is reachable.

    Raises RuntimeError with a human-friendly message when the schema is
    missing, so callers can fail fast before expensive setup (e.g. model
    registration).
    """
    try:
        with cdm_session(cdm_engine) as session:
            session.execute(select(Concept.concept_id).limit(1))
    except DBAPIError as e:
        error_msg = str(e).lower()
        if "does not exist" in error_msg or "no such table" in error_msg:
            logger.error(
                "Database schema is missing! Did you forget to run the ingestion CLI?"
            )
            raise RuntimeError("Database not initialized.") from e
        raise


def fetch_cdm_concepts_for_filter(
    concept_filter: Optional[CDMConceptFilter],
    cdm_engine: Engine,
) -> dict[int, Row]:
    """Return CDM rows matching *concept_filter*, keyed by concept_id.

    Selects all columns needed for both concept name lookup and embedding
    metadata (domain_id, vocabulary_id, canonical is_standard/is_valid flags),
    while retaining the underlying OMOP flag columns for existing callers.
    """
    query = concept_embedding_projection()
    if concept_filter is not None:
        query = concept_filter.apply(query)
    with cdm_session(cdm_engine) as session:
        return {row.concept_id: row for row in session.execute(query)}


def iter_cdm_concepts_for_filter(
    concept_filter: Optional[CDMConceptFilter],
    cdm_engine: Engine,
    chunk_size: int = 5_000,
) -> Iterator[Row]:
    """Stream CDM concept rows matching *concept_filter*, server-side chunked.

    Uses ``yield_per`` so the database driver fetches *chunk_size* rows at a
    time instead of buffering the full result set.  The session is held open
    for the lifetime of the generator.
    """
    query = concept_embedding_projection()
    if concept_filter is not None:
        query = concept_filter.apply(query)
    with cdm_session(cdm_engine) as session:
        yield from session.execute(streamed(query, chunk_size))


def count_missing_concepts(
    concept_filter: Optional[CDMConceptFilter],
    cdm_engine: Engine,
    embedded_ids: set[int],
    chunk_size: int = 10_000,
) -> int:
    """Return how many CDM concepts match *concept_filter* but lack an embedding.

    Streams only ``concept_id`` (one integer column) and checks each against
    *embedded_ids* via O(1) set lookup, far cheaper than fetching full rows.
    """
    query = select(Concept.concept_id)
    if concept_filter is not None:
        query = concept_filter.apply(query)
    count = 0
    with cdm_session(cdm_engine) as session:
        for row in session.execute(streamed(query, chunk_size)):
            if row.concept_id not in embedded_ids:
                count += 1
    return count
