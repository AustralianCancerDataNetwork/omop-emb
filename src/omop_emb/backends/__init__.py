from .base_backend import (
    EmbeddingBackend,
)
from .factory import (
    get_embedding_backend,
    normalize_backend_name,
)
from .index_config import (
    IndexConfig,
    FlatIndexConfig,
    HNSWIndexConfig,
)

__all__ = [
    "EmbeddingBackend",
    "get_embedding_backend",
    "normalize_backend_name",
    "IndexConfig",
    "FlatIndexConfig",
    "HNSWIndexConfig",
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
