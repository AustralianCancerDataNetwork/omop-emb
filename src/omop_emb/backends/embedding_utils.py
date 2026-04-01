from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Union, Type, TypeVar, Generic, Dict, Any, Callable, Tuple

from sqlalchemy import Select
from omop_alchemy.cdm.model.vocabulary import Concept
from .config import BackendType, SUPPORTED_INDICES_AND_METRICS_PER_BACKEND, IndexType, MetricType 


@dataclass(frozen=True)
class EmbeddingModelRecord:
    """
    Canonical description of a registered embedding model.

    ``storage_identifier`` is intentionally backend-specific. For example:
    - PostgreSQL backend: dynamic embedding table name
    - FAISS backend: on-disk index path or logical collection name
    """

    model_name: str
    dimensions: int
    backend_type: BackendType
    index_type: IndexType
    storage_identifier: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)


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
    concept_name: str
    similarity: float
    is_standard: bool
    is_active: bool


@dataclass(frozen=True)
class EmbeddingBackendCapabilities:
    """
    Capability flags for a backend implementation.

    These are not used by the current code yet, but they make backend
    differences explicit. For example, a FAISS backend might support nearest
    neighbor search but require explicit refreshes after bulk writes.
    """

    stores_embeddings: bool = True
    supports_incremental_upsert: bool = True
    supports_nearest_neighbor_search: bool = True
    supports_server_side_similarity: bool = True
    supports_filter_by_concept_ids: bool = True
    supports_filter_by_domain: bool = True
    supports_filter_by_vocabulary: bool = True
    supports_filter_by_standard_flag: bool = True
    requires_explicit_index_refresh: bool = False


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