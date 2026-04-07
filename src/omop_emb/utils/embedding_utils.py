from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Union, Type, TypeVar, Generic, Dict, Any, Callable, Tuple

from sqlalchemy import Select
from omop_alchemy.cdm.model.vocabulary import Concept
from ..config import BackendType, SUPPORTED_INDICES_AND_METRICS_PER_BACKEND, IndexType, MetricType 


@dataclass(frozen=True)
class EmbeddingConceptFilter:
    """
    Constraints applied during embedding retrieval.

    This mirrors the current OMOP grounding needs without importing
    ``omop_graph`` or its search-constraint objects into ``omop_emb``.
    """

    concept_ids: Optional[tuple[int, ...]] = None
    domains: Optional[tuple[str, ...]] = None
    vocabularies: Optional[tuple[str, ...]] = None
    require_standard: bool = False

    def apply(self, query: Select) -> Select:
        if self.concept_ids is not None:
            query = query.where(Concept.concept_id.in_(self.concept_ids))

        if self.domains is not None:
            query = query.where(Concept.domain_id.in_(self.domains))

        if self.vocabularies is not None:
            query = query.where(Concept.vocabulary_id.in_(self.vocabularies))

        if self.require_standard:
            query = query.where(Concept.standard_concept.in_(["S", "C"]))

        return query


@dataclass(frozen=True)
class NearestConceptMatch:
    """
    Backend-neutral nearest-neighbor payload returned to callers.

    The current resolver layer in ``omop-graph`` needs these fields to build
    ``LabelMatch`` objects and to explain whether a retrieved concept is
    standard and active.
    """

    concept_id: int
    concept_name: Optional[str]
    similarity: float
    is_standard: Optional[bool]
    is_active: Optional[bool]


def get_similarity_from_distance(distance_col, metric: MetricType):
    """
    Helper to map various distance metrics to a 0.0 - 1.0 similarity score.
    """
    if metric == MetricType.COSINE:
        return 1.0 - distance_col

    elif metric == MetricType.L2:
        return 1.0 / (1.0 + distance_col)

    elif metric == MetricType.L1:
        return 1.0 / (1.0 + distance_col)

    elif metric == MetricType.HAMMING:
        # Hamming distance is the number of differing bits.
        # To get similarity, we need to know total bits (dimensions)
        # Assuming you want a normalized score: 1 - (dist / dim)
        # Note: This requires passing 'dimensions' into the helper
        raise NotImplementedError()
        return 1.0 - (distance_col / dimensions)

    elif metric == MetricType.JACCARD:
        return 1.0 - distance_col

    else:
        raise ValueError(f"Unsupported metric type: {metric.value}")