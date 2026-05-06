from enum import StrEnum
from typing import Dict, Tuple

ENV_DOCUMENT_EMBEDDING_PREFIX = "OMOP_EMB_DOCUMENT_EMBEDDING_PREFIX"
ENV_QUERY_EMBEDDING_PREFIX = "OMOP_EMB_QUERY_EMBEDDING_PREFIX"
ENV_EMBEDDING_DIM = "OMOP_EMB_EMBEDDING_DIM"

ENV_OMOP_EMB_BACKEND = "OMOP_EMB_BACKEND"

# sqlite-vec backend (default)
ENV_EMB_SQLITE_PATH = "OMOP_EMB_SQLITE_PATH"

# pgvector backend (optional)
ENV_EMB_POSTGRES_URL = "OMOP_EMB_POSTGRES_URL"

# OMOP CDM (always required for concept ingestion in CLI)
ENV_CDM_DATABASE_URL = "OMOP_DATABASE_URL"


class ProviderType(StrEnum):
    """Embedding model provider.

    Members
    -------
    OLLAMA
        Self-hosted models served via the Ollama runtime.
    OPENAI
        OpenAI and OpenAI-compatible API endpoints.
    """

    OLLAMA = "ollama"
    OPENAI = "openai"


class BackendType(StrEnum):
    """Embedding storage backend.

    Members
    -------
    SQLITEVEC
        Default backend. Requires no external database server.
    PGVECTOR
        Optional backend. Requires a PostgreSQL instance with the pgvector
        extension (``pip install omop-emb[pgvector]``).
    FAISS
        Sidecar read-acceleration layer on top of any primary backend.
    """

    SQLITEVEC = "sqlitevec"
    PGVECTOR = "pgvector"
    FAISS = "faiss"


class VectorColumnType(StrEnum):
    """PostgreSQL column type used to store embedding vectors (pgvector only).

    Selected automatically by the pgvector backend based on dimensionality.
    Not applicable to the sqlite-vec backend.

    Members
    -------
    VECTOR
        ``pgvector`` ``vector`` type (float32, up to 2 000 dims).
    HALFVEC
        ``pgvector`` ``halfvec`` type (float16, up to 4 000 dims).

    Notes
    -----
    .. TODO: Include the storage options also for sqlite-vec
       https://alexgarcia.xyz/sqlite-vec/guides/scalar-quant.html
    """

    VECTOR = "vector"
    HALFVEC = "halfvec"


PGVECTOR_VECTOR_MAX_DIMENSIONS = 2_000
PGVECTOR_HALFVEC_MAX_DIMENSIONS = 4_000


class IndexType(StrEnum):
    """Index structure built on an embedding table for nearest-neighbor search.

    Members
    -------
    FLAT
        Exact sequential scan. Always correct, slower at scale.
    HNSW
        Approximate nearest-neighbor index. Supported by pgvector and FAISS
        only, not by sqlite-vec.
    """

    FLAT = "flat"
    HNSW = "hnsw"


class MetricType(StrEnum):
    """Distance metric used for nearest-neighbor search.

    Members
    -------
    L2
        Euclidean (L2) distance.
    COSINE
        Cosine distance.
    L1
        Manhattan (L1) distance.
    HAMMING
        Hamming distance (bit vectors).
    JACCARD
        Jaccard distance (bit vectors).
    """

    L2 = "l2"
    COSINE = "cosine"
    L1 = "l1"
    HAMMING = "hamming"
    JACCARD = "jaccard"


def parse_backend_type(value: str | BackendType) -> BackendType:
    """Parse a string or ``BackendType`` into a ``BackendType``.

    Parameters
    ----------
    value : str | BackendType
        Backend identifier string or enum member.

    Returns
    -------
    BackendType

    Raises
    ------
    ValueError
        If ``value`` is not a recognised backend type.
    """
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
    """Parse a string or ``IndexType`` into an ``IndexType``.

    Parameters
    ----------
    value : str | IndexType
        Index type identifier string or enum member.

    Returns
    -------
    IndexType

    Raises
    ------
    ValueError
        If ``value`` is not a recognised index type.
    """
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
    """Parse a string or ``MetricType`` into a ``MetricType``.

    Parameters
    ----------
    value : str | MetricType
        Metric type identifier string or enum member.

    Returns
    -------
    MetricType

    Raises
    ------
    ValueError
        If ``value`` is not a recognised metric type.
    """
    if isinstance(value, MetricType):
        return value
    try:
        return MetricType(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid metric type {value!r}. Expected one of "
            f"{[member.value for member in MetricType]}."
        ) from exc


# Supported index + metric combinations per backend.
SUPPORTED_INDICES_AND_METRICS_PER_BACKEND: Dict[BackendType, Dict[IndexType, Tuple[MetricType, ...]]] = {
    # https://alexgarcia.xyz/sqlite-vec/features/vec0.html
    BackendType.SQLITEVEC: {
        IndexType.FLAT: (MetricType.L2, MetricType.COSINE),
    },
    # https://github.com/pgvector/pgvector?tab=readme-ov-file#querying
    BackendType.PGVECTOR: {
        IndexType.FLAT: (MetricType.L2, MetricType.COSINE, MetricType.L1),
        IndexType.HNSW: (MetricType.L2, MetricType.COSINE, MetricType.L1),
    },
    BackendType.FAISS: {
        IndexType.FLAT: (MetricType.L2, MetricType.COSINE),
        IndexType.HNSW: (MetricType.L2, MetricType.COSINE),
    },
}


def is_supported_index_metric_combination_for_backend(
    backend: BackendType, index: IndexType, metric: MetricType
) -> bool:
    """Return whether a backend supports a given index and metric combination.

    Parameters
    ----------
    backend : BackendType
    index : IndexType
    metric : MetricType

    Returns
    -------
    bool
    """
    supported = SUPPORTED_INDICES_AND_METRICS_PER_BACKEND.get(backend, {})
    return metric in supported.get(index, ())


def is_index_type_supported_for_backend(backend: BackendType, index: IndexType) -> bool:
    """Return whether a backend supports a given index type.

    Parameters
    ----------
    backend : BackendType
    index : IndexType

    Returns
    -------
    bool
    """
    return index in SUPPORTED_INDICES_AND_METRICS_PER_BACKEND.get(backend, {})


def get_supported_index_types_for_backend(backend: BackendType) -> Tuple[IndexType, ...]:
    """Return all index types supported by a backend.

    Parameters
    ----------
    backend : BackendType

    Returns
    -------
    tuple[IndexType, ...]
    """
    return tuple(SUPPORTED_INDICES_AND_METRICS_PER_BACKEND.get(backend, {}).keys())


def get_supported_metrics_for_backend(backend: BackendType) -> Tuple[MetricType, ...]:
    """Return all metrics supported by any index type for a backend.

    Parameters
    ----------
    backend : BackendType

    Returns
    -------
    tuple[MetricType, ...]
    """
    seen: set[MetricType] = set()
    for metrics in SUPPORTED_INDICES_AND_METRICS_PER_BACKEND.get(backend, {}).values():
        seen.update(metrics)
    return tuple(seen)
