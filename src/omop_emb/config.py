from enum import StrEnum
from typing import Dict, Tuple

ENV_DOCUMENT_EMBEDDING_PREFIX = "OMOP_EMB_DOCUMENT_EMBEDDING_PREFIX"
ENV_QUERY_EMBEDDING_PREFIX = "OMOP_EMB_QUERY_EMBEDDING_PREFIX"
ENV_EMBEDDING_DIM = "OMOP_EMB_EMBEDDING_DIM"

# Kept for CLI / legacy migration tooling that may still reference it.
ENV_OMOP_EMB_BACKEND = "OMOP_EMB_BACKEND"
# Removed: ENV_BASE_STORAGE_DIR — storage is now entirely in Postgres.

ENV_EMB_POSTGRES_URL = "OMOP_EMB_POSTGRES_URL"
ENV_CDM_DATABASE_URL = "OMOP_DATABASE_URL"


class ProviderType(StrEnum):
    """Enum for supported embedding model providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"


class BackendType(StrEnum):
    """Embedding storage backend.  Currently only pgvector is supported as a
    production backend; the enum is preserved so that ``EmbeddingModelRecord``
    and the registry schema remain stable.
    """
    PGVECTOR = "pgvector"


class VectorColumnType(StrEnum):
    """Postgres column type used to store embedding vectors.

    ``VECTOR``  – ``pgvector`` ``vector`` type  (float32, ≤ 2 000 dims).
    ``HALFVEC`` – ``pgvector`` ``halfvec`` type (float16, ≤ 4 000 dims).

    Selected automatically by the backend based on dimensionality:
    dimensions > 2 000 → HALFVEC, otherwise VECTOR.
    """
    VECTOR = "vector"
    HALFVEC = "halfvec"


_VECTOR_MAX_DIMENSIONS = 2_000
_HALFVEC_MAX_DIMENSIONS = 4_000


def vector_column_type_for_dimensions(dimensions: int) -> VectorColumnType:
    """Return the appropriate Postgres column type for *dimensions*."""
    if dimensions <= _VECTOR_MAX_DIMENSIONS:
        return VectorColumnType.VECTOR
    if dimensions <= _HALFVEC_MAX_DIMENSIONS:
        return VectorColumnType.HALFVEC
    raise ValueError(
        f"pgvector supports at most {_HALFVEC_MAX_DIMENSIONS:,} dimensions "
        f"(halfvec), but model requests {dimensions:,}."
    )


class IndexType(StrEnum):
    """Enum for index types used for nearest neighbor search."""
    FLAT = "flat"
    HNSW = "hnsw"


class MetricType(StrEnum):
    """Defines the distance metric used for nearest neighbor search."""
    L2 = "l2"
    COSINE = "cosine"
    L1 = "l1"
    HAMMING = "hamming"
    JACCARD = "jaccard"


def parse_backend_type(value: str | BackendType) -> BackendType:
    if isinstance(value, BackendType):
        return value
    try:
        return BackendType(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid backend type {value!r}. Expected one of "
            f"{[member.value for member in BackendType]}."
        ) from exc


def parse_index_type(value: str | IndexType) -> IndexType:
    if isinstance(value, IndexType):
        return value
    try:
        return IndexType(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid index type {value!r}. Expected one of "
            f"{[member.value for member in IndexType]}."
        ) from exc


def parse_metric_type(value: str | MetricType) -> MetricType:
    if isinstance(value, MetricType):
        return value
    try:
        return MetricType(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid metric type {value!r}. Expected one of "
            f"{[member.value for member in MetricType]}."
        ) from exc


SUPPORTED_INDICES_AND_METRICS_PER_BACKEND: Dict[BackendType, Dict[IndexType, Tuple[MetricType, ...]]] = {
    # https://github.com/pgvector/pgvector?tab=readme-ov-file#querying
    BackendType.PGVECTOR: {
        IndexType.FLAT: (MetricType.L2, MetricType.COSINE, MetricType.L1, MetricType.HAMMING, MetricType.JACCARD),
        IndexType.HNSW: (MetricType.L2, MetricType.COSINE, MetricType.L1),
    },
}

# FAISS cache supports the same metrics as pgvector FLAT/HNSW for L2 and COSINE.
SUPPORTED_FAISS_CACHE_METRICS: Tuple[MetricType, ...] = (MetricType.L2, MetricType.COSINE)


def is_supported_index_metric_combination_for_backend(backend: BackendType, index: IndexType, metric: MetricType) -> bool:
    supported = SUPPORTED_INDICES_AND_METRICS_PER_BACKEND.get(backend, {})
    return metric in supported.get(index, ())


def is_index_type_supported_for_backend(backend: BackendType, index: IndexType) -> bool:
    return index in SUPPORTED_INDICES_AND_METRICS_PER_BACKEND.get(backend, {})


def get_supported_index_types_for_backend(backend: BackendType) -> Tuple[IndexType, ...]:
    return tuple(SUPPORTED_INDICES_AND_METRICS_PER_BACKEND.get(backend, {}).keys())
