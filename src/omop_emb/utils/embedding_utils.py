from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, overload
import logging

from sqlalchemy import func
from sqlalchemy.sql.elements import ColumnElement

from omop_alchemy.cdm.query import ConceptFilter as CDMConceptFilter  # noqa: F401
from omop_emb.config import (
    MetricType,
    PGVECTOR_HALFVEC_MAX_DIMENSIONS,
    PGVECTOR_VECTOR_MAX_DIMENSIONS,
    VectorColumnType,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingConceptFilter:
    """Search constraints applied during KNN retrieval.

    All fields are optional. Unset fields impose no constraint. 
    To limit the number of KNN results returned, pass `k` to 
    `get_nearest_concepts`/`get_similar_concepts`. 

    Notes
    -----
    Mirrors OMOP grounding needs without importing ``omop_graph`` or its
    search-constraint types into ``omop_emb``. The field shape deliberately
    tracks ``omop_alchemy.cdm.query.ConceptFilter``, but the two are not
    interchangeable: ``ConceptFilter.apply()`` targets the CDM ``concept``
    table, while this filter constrains the embedding sidecar, whose flags are
    materialised boolean columns. omop-graph's equivalent type was retired in
    favour of ``ConceptFilter`` under OMOP_Alchemy#11; this one survives for
    that reason.

    Attributes
    ----------
    concept_ids : tuple[int, ...], optional
        Restrict results to this set of concept IDs.
    domains : tuple[str, ...], optional
        Restrict results to concepts in these OMOP domains.
    vocabularies : tuple[str, ...], optional
        Restrict results to concepts from these vocabularies.
    require_standard : bool
        When ``True``, only concepts satisfying omop-alchemy's canonical
        ``Concept.is_standard`` rule — raw flag ``'S'`` — are returned.
        Default ``False``.
    include_classification : bool
        Widens ``require_standard`` to also admit classification (``'C'``)
        concepts, matching ``ConceptFilter.include_classification``. Only
        meaningful with ``require_standard``. Default ``False``.
    require_active : bool
        When ``True``, only concepts satisfying omop-alchemy's canonical
        ``Concept.is_valid`` rule are returned. Default ``False``.
    """

    concept_ids: Optional[tuple[int, ...]] = None
    domains: Optional[tuple[str, ...]] = None
    vocabularies: Optional[tuple[str, ...]] = None
    require_standard: bool = False
    include_classification: bool = False
    require_active: bool = False

    def is_empty(self) -> bool:
        """Return ``True`` if no constraints are set."""
        return (
            self.concept_ids is None
            and self.domains is None
            and self.vocabularies is None
            and not self.require_standard
            and not self.require_active
        )


@dataclass(frozen=True)
class NearestConceptMatch:
    """Single nearest-neighbour result as returned to callers.

    ``concept_id`` and ``similarity`` are always populated by the backend.
    ``domain_id``, ``vocabulary_id``, ``is_standard``, and ``is_active`` are
    also populated by the backend directly from the embedding table's filter
    columns (see :class:`~omop_emb.backends.embedding_table.ConceptEmbeddingMixin`),
    so they are available regardless of whether a CDM engine is configured.
    ``concept_name`` is the only field enriched by the interface layer from
    the OMOP CDM, and only when an ``omop_cdm_engine`` is provided.

    Attributes
    ----------
    concept_id : int
        OMOP concept ID of the matched concept.
    similarity : float
        Similarity score in ``[0.0, 1.0]``. Higher is more similar.
    concept_name : str, optional
        Human-readable concept name. ``None`` when no CDM engine is provided.
    domain_id : str, optional
        OMOP domain (e.g. ``'Condition'``, ``'Drug'``), taken from the
        embedding table. ``None`` only if the backend could not resolve it.
    vocabulary_id : str, optional
        Source vocabulary (e.g. ``'SNOMED'``, ``'RxNorm'``), taken from the
        embedding table. ``None`` only if the backend could not resolve it.
    is_classification : bool, optional
        Whether the concept is a classification ('C') concept — a hierarchy
        node, not a valid mapping target.
    is_standard : bool, optional
        ``True`` if ``standard_concept`` is ``'S'`` or ``'C'``, taken from
        the embedding table. ``None`` only if the backend could not resolve
        it.
    is_active : bool, optional
        ``True`` if ``invalid_reason`` is not ``'D'`` or ``'U'``, taken from
        the embedding table. ``None`` only if the backend could not resolve
        it.
    """

    concept_id: int
    similarity: float
    concept_name: Optional[str] = None
    domain_id: Optional[str] = None
    vocabulary_id: Optional[str] = None
    is_standard: Optional[bool] = None
    is_classification: Optional[bool] = None
    is_active: Optional[bool] = None

    def to_dict(self) -> dict:
        """Convert to a dictionary for serialization."""
        return asdict(self)


@overload
def get_similarity_from_distance(
    distance_col: float,
    metric: MetricType,
) -> float: ...


@overload
def get_similarity_from_distance(
    distance_col: ColumnElement,
    metric: MetricType,
) -> ColumnElement: ...


def get_similarity_from_distance(
    distance_col: float | ColumnElement,
    metric: MetricType,
) -> float | ColumnElement:
    """Convert a raw distance value to a similarity score in ``[0.0, 1.0]``.

    Parameters
    ----------
    distance_col : float | ColumnElement
        Raw distance value or SQLAlchemy column expression.
    metric : MetricType
        Distance metric that produced ``distance_col``.

    Returns
    -------
    float | ColumnElement
        Similarity in ``[0.0, 1.0]``. When ``distance_col`` is a
        ``ColumnElement`` the result is also a column expression with
        ``LEAST``/``GREATEST`` clamping applied.

    Notes
    -----
    Conversion formulas:

    * ``COSINE``: distance in ``[0, 2]``, so ``similarity = 1 - dist/2``.
    * ``L2``: ``similarity = 1 / (1 + dist)``.
    * ``L1``: ``similarity = 1 / (1 + dist)``.
    * ``JACCARD``: ``similarity = 1 - dist``.
    * ``HAMMING``: not implemented.
    """
    if metric == MetricType.COSINE:
        similarity = 1.0 - (distance_col / 2.0)
    elif metric == MetricType.L2:
        similarity = 1.0 / (1.0 + distance_col)
    elif metric == MetricType.L1:
        similarity = 1.0 / (1.0 + distance_col)
    elif metric == MetricType.HAMMING:
        raise ValueError(
            "HAMMING distance has no similarity conversion formula for 'vector' columns. "
            "It requires a 'bit' column type which is not currently supported."
        )
    elif metric == MetricType.JACCARD:
        similarity = 1.0 - distance_col
    else:
        raise ValueError(f"Unsupported metric type: {metric.value}")

    if isinstance(similarity, ColumnElement):
        return func.least(func.greatest(similarity, 0.0), 1.0)
    else:
        return min(1.0, max(0.0, similarity))


def vector_column_type_for_dimensions(dimensions: int) -> VectorColumnType:
    """Return the appropriate PostgreSQL column type for a given dimensionality.

    Parameters
    ----------
    dimensions : int
        Number of dimensions in the embedding vector.

    Returns
    -------
    VectorColumnType
        ``VECTOR`` for dimensions up to 2 000, ``HALFVEC`` for up to 4 000.

    Raises
    ------
    ValueError
        If ``dimensions`` exceeds the halfvec limit of 4 000.
    """
    if dimensions <= PGVECTOR_VECTOR_MAX_DIMENSIONS:
        return VectorColumnType.VECTOR
    if dimensions <= PGVECTOR_HALFVEC_MAX_DIMENSIONS:
        logger.warning(
            f"Using {VectorColumnType.HALFVEC} for {dimensions} dimensions. This uses float16 quantization which may reduce accuracy. "
            f"Consider reducing dimensionality to {PGVECTOR_VECTOR_MAX_DIMENSIONS} or less to use the full float32 precision of {VectorColumnType.VECTOR}."
        )
        return VectorColumnType.HALFVEC
    raise ValueError(
        f"pgvector supports at most {PGVECTOR_HALFVEC_MAX_DIMENSIONS:,} dimensions "
        f"(halfvec), but model requests {dimensions:,}."
    )
