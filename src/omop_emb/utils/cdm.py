"""OMOP CDM utilities. Since embeddings and OMOP CDM are separate, we have this helper utility in case we need to query the OMOP CDM."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import Engine, Row, select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, sessionmaker

from omop_alchemy.cdm.model.vocabulary import Concept
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter

logger = logging.getLogger(__name__)

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
            logger.error("Database schema is missing! Did you forget to run the ingestion CLI?")
            raise RuntimeError("Database not initialized.") from e
        raise


def fetch_cdm_concepts_for_filter(
    concept_filter: Optional[EmbeddingConceptFilter],
    cdm_engine: Engine,
) -> dict[int, Row]:
    """Return CDM rows matching *concept_filter*, keyed by concept_id.

    Selects all columns needed for both concept name lookup and embedding
    metadata (domain_id, vocabulary_id, standard_concept, invalid_reason),
    so callers do not need a second CDM round-trip.
    """
    query = select(
        Concept.concept_id,
        Concept.concept_name,
        Concept.domain_id,
        Concept.vocabulary_id,
        Concept.standard_concept,
        Concept.invalid_reason,
    )
    if concept_filter is not None:
        query = concept_filter.apply(query, Concept)
    with cdm_session(cdm_engine) as session:
        return {row.concept_id: row for row in session.execute(query)}


def fetch_cdm_concepts_for_ingestion(
    concept_ids: set[int],
    cdm_engine: Engine,
    batch_size: int = 50_000,
) -> dict[int, Row]:
    """Return CDM rows needed to build ``ConceptEmbeddingRecord`` filter columns.

    Sub-batches to avoid bind-parameter limits on large concept sets.
    Fetches ``domain_id``, ``vocabulary_id``, ``standard_concept``, and
    ``invalid_reason`` for each concept_id.
    """
    if not concept_ids:
        return {}
    id_list = list(concept_ids)
    result: dict[int, Row] = {}
    for start in range(0, len(id_list), batch_size):
        chunk = id_list[start : start + batch_size]
        query = select(
            Concept.concept_id,
            Concept.domain_id,
            Concept.vocabulary_id,
            Concept.standard_concept,
            Concept.invalid_reason,
        ).where(Concept.concept_id.in_(chunk))
        with cdm_session(cdm_engine) as session:
            result.update({row.concept_id: row for row in session.execute(query)})
    return result
