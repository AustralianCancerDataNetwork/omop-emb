from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import Select, func
from sqlalchemy.sql.elements import ColumnElement
from omop_alchemy.cdm.model.vocabulary import Concept
from ..config import MetricType 


@dataclass(frozen=True)
class EmbeddingConceptFilter:
    """
    Constraints applied during embedding retrieval.

    This mirrors the current OMOP grounding needs without importing
    ``omop_graph`` or its search-constraint objects into ``omop_emb``.

    The `limit` field determines the number of nearest neighbors returned by embedding search operations. If not set, a backend default may be used.
    """

    concept_ids: Optional[tuple[int, ...]] = None
    domains: Optional[tuple[str, ...]] = None
    vocabularies: Optional[tuple[str, ...]] = None
    require_standard: bool = False
    limit: Optional[int] = None

    def __post_init__(self) -> None:
        if self.limit is not None and self.limit <= 0:
            raise ValueError(
                f"EmbeddingConceptFilter.limit must be a positive integer, got {self.limit}."
            )

    def apply(self, query: Select) -> Select:
        if self.concept_ids is not None:
            query = query.where(Concept.concept_id.in_(self.concept_ids))

        if self.domains is not None:
            query = query.where(Concept.domain_id.in_(self.domains))

        if self.vocabularies is not None:
            query = query.where(Concept.vocabulary_id.in_(self.vocabularies))

        if self.require_standard:
            query = query.where(Concept.standard_concept.in_(["S", "C"]))

        return query.limit(self.limit)
    
    def is_empty(self) -> bool:
        return (
            self.concept_ids is None and
            self.domains is None and
            self.vocabularies is None and
            not self.require_standard and
            self.limit is None
        )


@dataclass(frozen=True)
class NearestConceptMatch:
    """
    Backend-neutral nearest-neighbor payload returned to callers.

    The current resolver layer in ``omop-graph`` needs these fields to build
    ``LabelMatch`` objects and to explain whether a retrieved concept is
    standard and active.
    """

    concept_id: int
    concept_name: str
    similarity: float
    is_standard: bool
    is_active: bool


def get_similarity_from_distance(distance_col: float | ColumnElement, metric: MetricType) -> float | ColumnElement:
    """
    Map distance values to a similarity score in [0.0, 1.0].
    """
    if metric == MetricType.COSINE:
        # Cosine distance is typically in [0, 2], so this maps to [0, 1].
        similarity = 1.0 - (distance_col / 2.0)
    elif metric == MetricType.L2:
        similarity = 1.0 / (1.0 + distance_col)
    elif metric == MetricType.L1:
        similarity = 1.0 / (1.0 + distance_col)
    elif metric == MetricType.HAMMING:
        raise NotImplementedError()
    elif metric == MetricType.JACCARD:
        similarity = 1.0 - distance_col
    else:
        raise ValueError(f"Unsupported metric type: {metric.value}")

    if isinstance(similarity, ColumnElement):
        return func.least(func.greatest(similarity, 0.0), 1.0)
    else:
        return min(1.0, max(0.0, similarity))
