from __future__ import annotations

from typing import TYPE_CHECKING

from .base_backend import (
    EmbeddingBackend,
    EmbeddingStoreReader,
    open_vector_store_reader,
    open_vector_store_writer,
)
from .sqlitevec import SQLiteVecEmbeddingBackend

if TYPE_CHECKING:
    from .duckdbvss import DuckDBVSSEmbeddingBackend
    from .pgvector import PGVectorEmbeddingBackend

__all__ = [
    "DuckDBVSSEmbeddingBackend",
    "EmbeddingBackend",
    "EmbeddingStoreReader",
    "PGVectorEmbeddingBackend",
    "SQLiteVecEmbeddingBackend",
    "open_vector_store_reader",
    "open_vector_store_writer",
]


def __getattr__(name: str):
    if name == "PGVectorEmbeddingBackend":
        from .pgvector import (
            PGVectorEmbeddingBackend,
        )  # raises ImportError with install hint if absent

        globals()[name] = PGVectorEmbeddingBackend
        return PGVectorEmbeddingBackend
    if name == "DuckDBVSSEmbeddingBackend":
        from .duckdbvss import (
            DuckDBVSSEmbeddingBackend,
        )  # raises ImportError with install hint if absent

        globals()[name] = DuckDBVSSEmbeddingBackend
        return DuckDBVSSEmbeddingBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
