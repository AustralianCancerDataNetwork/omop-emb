"""sqlite-vec backend (default, no external dependencies)."""

from omop_emb.backends.sqlitevec.sqlitevec_backend import (
    SQLiteVecEmbeddingBackend,
)

__all__ = [
    "SQLiteVecEmbeddingBackend",
]
