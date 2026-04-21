from enum import StrEnum
from typing import Dict, Tuple

ENV_OMOP_EMB_BACKEND = "OMOP_EMB_BACKEND"
ENV_BASE_STORAGE_DIR = "OMOP_EMB_BASE_STORAGE_DIR"  # For backends that use file-based storage, e.g. FAISS with on-disk indices or registry metadata storage

class ProviderType(StrEnum):
    """Enum for supported embedding model providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"

class BackendType(StrEnum):
    """Enum for supported embedding backends."""

    PGVECTOR = "pgvector"
    FAISS = "faiss"

class IndexType(StrEnum):
    """Enum for index types used for nearest neighbor search.
    Index types are specific to backends and have different performance characteristics and supported metrics.

    
    Notes
    -----
    Not each index type is supported by each backend
    """
    FLAT = "flat"  # Exact search, no index
    HNSW = "hnsw"  # Hierarchical Navigable Small World graph
    IVF = "ivf" # Inverted File Index
    IVF_PQ = "ivf_pq" # IVF with Product Quantization

class MetricType(StrEnum):
    """Defines the distance type used for nearest neighbor search. 

    Notes
    -----
    Not all metrics are supported for all index types and backends.
    
    """
    L2 = "l2"
    COSINE = "cosine"
    L1 = "l1"
    HAMMING = "hamming"
    JACCARD = "jaccard"


def parse_backend_type(value: str | BackendType) -> BackendType:
    """Normalize a backend value to ``BackendType`` with a clear error message."""
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
    """Normalize an index value to ``IndexType`` with a clear error message."""
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
    """Normalize a metric value to ``MetricType`` with a clear error message."""
    if isinstance(value, MetricType):
        return value
    try:
        return MetricType(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid metric type {value!r}. Expected one of "
            f"{[member.value for member in MetricType]}."
        ) from exc

# TODO: Support non-flat indices in the future
SUPPORTED_INDICES_AND_METRICS_PER_BACKEND: Dict[BackendType, Dict[IndexType, Tuple[MetricType, ...]]] = {
    #https://github.com/pgvector/pgvector?tab=readme-ov-file#querying
    BackendType.PGVECTOR: {
        IndexType.FLAT: (MetricType.L2, MetricType.COSINE, MetricType.L1, MetricType.HAMMING, MetricType.JACCARD),
    },
    # Check here: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    BackendType.FAISS: {
        IndexType.FLAT: (MetricType.L2, MetricType.COSINE),
        IndexType.HNSW: (MetricType.L2, MetricType.COSINE),
    }
}

def is_supported_index_metric_combination_for_backend(backend: BackendType, index: IndexType, metric: MetricType) -> bool:
    supported_indices_and_metrics = SUPPORTED_INDICES_AND_METRICS_PER_BACKEND.get(backend, {})
    supported_metrics = supported_indices_and_metrics.get(index, ())
    return metric in supported_metrics

def is_index_type_supported_for_backend(backend: BackendType, index: IndexType) -> bool:
    supported_indices_and_metrics = SUPPORTED_INDICES_AND_METRICS_PER_BACKEND.get(backend, {})
    return index in supported_indices_and_metrics

def get_supported_index_types_for_backend(backend: BackendType) -> Tuple[IndexType, ...]:
    supported_indices_and_metrics = SUPPORTED_INDICES_AND_METRICS_PER_BACKEND.get(backend, {})
    return tuple(supported_indices_and_metrics.keys())

def get_supported_metrics_for_backend_index(backend: BackendType, index: IndexType) -> Tuple[MetricType, ...]:
    supported_indices_and_metrics = SUPPORTED_INDICES_AND_METRICS_PER_BACKEND.get(backend, {})
    return supported_indices_and_metrics.get(index, ())
