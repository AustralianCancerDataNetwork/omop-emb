from __future__ import annotations

from typing import TYPE_CHECKING

from .base_backend import EmbeddingBackend, resolve_backend, resolve_backend_from_resolved_vector_store
from .read_only import (
    ReadOnlyEmbeddingStore,
    StoredEmbedding,
    initialize_resolved_vector_store,
    inspect_resolved_vector_store,
)
from .sqlitevec import SQLiteVecEmbeddingBackend

if TYPE_CHECKING:
    from .pgvector import PGVectorEmbeddingBackend

__all__ = [
    "EmbeddingBackend",
    "resolve_backend",
    "resolve_backend_from_resolved_vector_store",
    "ReadOnlyEmbeddingStore",
    "StoredEmbedding",
    "initialize_resolved_vector_store",
    "inspect_resolved_vector_store",
    "SQLiteVecEmbeddingBackend",
    "PGVectorEmbeddingBackend",
]


def __getattr__(name: str):
    if name == "PGVectorEmbeddingBackend":
        from .pgvector import (
            PGVectorEmbeddingBackend,
        )  # raises ImportError with install hint if absent

        globals()[name] = PGVectorEmbeddingBackend
        return PGVectorEmbeddingBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
