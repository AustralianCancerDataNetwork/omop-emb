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
    DEFAULT_BACKEND,
    SUPPORTED_BACKENDS,
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
    "DEFAULT_BACKEND",
    "SUPPORTED_BACKENDS",
    "get_embedding_backend",
    "normalize_backend_name",
    "FaissEmbeddingBackend",
    "PostgresEmbeddingBackend",
]


def __getattr__(name: str):
    if name == "FaissEmbeddingBackend":
        from .faiss import FaissEmbeddingBackend

        return FaissEmbeddingBackend
    if name == "PostgresEmbeddingBackend":
        from .postgres import PostgresEmbeddingBackend

        return PostgresEmbeddingBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
