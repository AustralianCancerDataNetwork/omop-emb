"""Query texts: sampled from the CDM, read from a file, and embedded."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from omop_alchemy.cdm.model.vocabulary import Concept, Concept_Synonym
from omop_llm import EmbeddingRole
from sqlalchemy import func, literal, select

from benchmarks.recall.scenarios import Query
from omop_emb.interface import EmbeddingReaderInterface
from omop_emb.utils.cdm import CDMSessionFactory


def sample_queries(
    cdm_sessions: CDMSessionFactory, domains: Sequence[str], per_domain: int, seed: str
) -> list[Query]:
    """Sample concept synonyms per domain, deterministically for a seed.

    A synonym is a realistic query whose text differs from the concept name,
    so the concept it belongs to is known and nearby but not an exact match.
    Synonyms are distinct within a domain and ordered by a hash of the seed.
    """
    queries: list[Query] = []
    with cdm_sessions() as session:
        for domain in domains:
            name = Concept_Synonym.concept_synonym_name
            statement = (
                select(name)
                .join(Concept, Concept.concept_id == Concept_Synonym.concept_id)
                .where(
                    Concept.domain_id == domain,
                    name.is_distinct_from(Concept.concept_name),
                    func.btrim(name) != "",
                )
                .group_by(name)
                .order_by(func.md5(literal(f"{seed}:{domain}:") + name), name)
                .limit(per_domain)
            )
            for text in session.scalars(statement):
                queries.append(Query(id=len(queries), text=str(text), domain=domain))
    return queries


def read_queries(path: Path) -> list[Query]:
    """Read a JSON list of ``{"q" or "text": ..., "domain": ...}`` objects."""
    items = json.loads(path.read_text(encoding="utf-8"))
    return [
        Query(id=index, text=str(item.get("q", item.get("text"))), domain=str(item["domain"]))
        for index, item in enumerate(items)
    ]


def embed_queries(model_backend, queries: Sequence[Query], batch_size: int = 32) -> np.ndarray:
    """Embed query texts with the store's model as float32."""
    return np.asarray(
        EmbeddingReaderInterface.generate_embeddings(
            model_backend,
            [q.text for q in queries],
            role=EmbeddingRole.QUERY,
            batch_size=batch_size,
        ),
        dtype=np.float32,
    )
