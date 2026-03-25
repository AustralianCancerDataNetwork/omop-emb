from .base import (
    EmbeddingBackend,
    EmbeddingBackendCapabilities,
    EmbeddingConceptFilter,
    EmbeddingIndexConfig,
    EmbeddingModelRecord,
    NearestConceptMatch,
)
from .errors import (
    EmbeddingBackendConfigurationError,
    EmbeddingBackendDependencyError,
    EmbeddingBackendError,
    UnknownEmbeddingBackendError,
)
from .factory import (
    get_embedding_backend,
    normalize_backend_name,
)

__all__ = [
    "EmbeddingBackend",
    "EmbeddingBackendCapabilities",
    "EmbeddingConceptFilter",
    "EmbeddingIndexConfig",
    "EmbeddingModelRecord",
    "NearestConceptMatch",
    "EmbeddingBackendConfigurationError",
    "EmbeddingBackendDependencyError",
    "EmbeddingBackendError",
    "UnknownEmbeddingBackendError",
    "get_embedding_backend",
    "normalize_backend_name",
    "FaissEmbeddingBackend",
    "PGVectorEmbeddingBackend",
]


def __getattr__(name: str):
    if name == "FaissEmbeddingBackend":
        from .faiss import FaissEmbeddingBackend

        return FaissEmbeddingBackend
    if name == "PGVectorEmbeddingBackend":
        from .pgvector import PGVectorEmbeddingBackend

        return PGVectorEmbeddingBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
