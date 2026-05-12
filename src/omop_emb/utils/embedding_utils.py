from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, overload
import logging
logger = logging.getLogger(__name__)

from sqlalchemy import Select, func
from sqlalchemy.sql.elements import ColumnElement
from omop_alchemy.cdm.model.vocabulary import Concept

from omop_emb.config import (
    MetricType,
    PGVECTOR_HALFVEC_MAX_DIMENSIONS,
    PGVECTOR_VECTOR_MAX_DIMENSIONS,
    VectorColumnType
)


@dataclass(frozen=True)
class EmbeddingConceptFilter:
    """Search constraints applied during KNN retrieval.

    All fields are optional. Unset fields impose no constraint. ``limit``
    maps directly to the ``k`` nearest neighbours returned.

    Notes
    -----
    Mirrors OMOP grounding needs without importing ``omop_graph`` or its
    search-constraint types into ``omop_emb``.

    Attributes
    ----------
    concept_ids : tuple[int, ...], optional
        Restrict results to this set of concept IDs.
    domains : tuple[str, ...], optional
        Restrict results to concepts in these OMOP domains.
    vocabularies : tuple[str, ...], optional
        Restrict results to concepts from these vocabularies.
    require_standard : bool
        When ``True``, only standard concepts (``standard_concept`` in
        ``('S', 'C')``) are returned. Default ``False``.
    require_active : bool
        When ``True``, only active concepts (``invalid_reason`` not in 
        ``('D', 'U')``) are returned. Default ``False``.
    limit : int, optional
        Maximum number of nearest neighbours to return. If not set, the
        backend default is used.
    """

    concept_ids: Optional[tuple[int, ...]] = None
    domains: Optional[tuple[str, ...]] = None
    vocabularies: Optional[tuple[str, ...]] = None
    require_standard: bool = False
    require_active: bool = False
    limit: Optional[int] = None

    def __post_init__(self) -> None:
        if self.limit is not None and self.limit <= 0:
            raise ValueError(
                f"EmbeddingConceptFilter.limit must be a positive integer, got {self.limit}."
            )

    def apply(self, query: Select, table: type) -> Select:
        """Apply filter constraints to a CDM-backed SQLAlchemy select.

        .. warning::
            CDM use only.  This method generates ``IN (…)`` bind parameters and
            is safe only when list fields (``concept_ids``, ``domains``,
            ``vocabularies``) are small.  Embedding backend queries must use
            :func:`omop_emb.backends.db_utils.setup_concept_filter_temps` with
            subquery-based WHERE clauses instead.

        Parameters
        ----------
        query : Select
            Base select statement targeting the OMOP CDM ``concept`` table.

        Returns
        -------
        Select
            Query with all active constraints and ``limit`` applied.
        """
        if self.concept_ids is not None:
            query = query.where(table.concept_id.in_(self.concept_ids))

        if self.domains is not None:
            query = query.where(table.domain_id.in_(self.domains))

        if self.vocabularies is not None:
            query = query.where(table.vocabulary_id.in_(self.vocabularies))

        if self.require_standard:
            if hasattr(table, "is_standard"):
                query = query.where(table.is_standard == True)  # noqa: E712
            else:
                query = query.where(table.standard_concept.in_(["S", "C"]))

        if self.require_active:
            if hasattr(table, "is_valid"):
                query = query.where(table.is_valid == True)  # noqa: E712
            else:
                query = query.where(table.invalid_reason.not_in(["D", "U"]))

        if self.limit is not None:
            query = query.limit(self.limit)
        return query

    def is_empty(self) -> bool:
        """Return ``True`` if no constraints are set."""
        return (
            self.concept_ids is None and
            self.domains is None and
            self.vocabularies is None and
            not self.require_standard and
            not self.require_active and
            self.limit is None
        )


@dataclass(frozen=True)
class NearestConceptMatch:
    """Single nearest-neighbour result as returned to callers.

    ``concept_id`` and ``similarity`` are always populated by the backend.
    The remaining fields are optionally enriched by the interface layer from
    the OMOP CDM when an ``omop_cdm_engine`` is provided.

    Attributes
    ----------
    concept_id : int
        OMOP concept ID of the matched concept.
    similarity : float
        Similarity score in ``[0.0, 1.0]``. Higher is more similar.
    concept_name : str, optional
        Human-readable concept name. ``None`` when no CDM engine is provided.
    is_standard : bool, optional
        ``True`` if ``standard_concept`` is ``'S'`` or ``'C'``. ``None``
        when no CDM engine is provided.
    is_active : bool, optional
        ``True`` if ``invalid_reason`` is not ``'D'`` or ``'U'``. ``None``
        when no CDM engine is provided.
    """

    concept_id: int
    similarity: float
    concept_name: Optional[str] = None
    is_standard: Optional[bool] = None
    is_active: Optional[bool] = None

    def to_dict(self) -> dict:
        """Convert to a dictionary for serialization."""
        return asdict(self)


@dataclass(frozen=True)
class ConceptEmbeddingRecord:
    """Concept metadata for a single embedding upsert row.

    Populated from the OMOP CDM by the caller (interface layer) before being
    passed to the backend.

    Attributes
    ----------
    concept_id : int
        OMOP concept ID.
    domain_id : str
        OMOP domain (e.g. ``'Condition'``, ``'Drug'``).
    vocabulary_id : str
        Source vocabulary (e.g. ``'SNOMED'``, ``'RxNorm'``).
    is_standard : bool
        ``True`` if ``standard_concept`` is ``'S'`` or ``'C'``.
    """

    concept_id: int
    domain_id: str
    vocabulary_id: str
    is_standard: bool
    is_valid: bool = True


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

    * ``COSINE`` -- distance in ``[0, 2]``, so ``similarity = 1 - dist/2``.
    * ``L2``     -- ``similarity = 1 / (1 + dist)``.
    * ``L1``     -- ``similarity = 1 / (1 + dist)``.
    * ``JACCARD`` -- ``similarity = 1 - dist``.
    * ``HAMMING`` -- not implemented.
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