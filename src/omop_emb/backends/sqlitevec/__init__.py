"""sqlite-vec backend (default, no external dependencies)."""
from omop_emb.backends.sqlitevec.sqlitevec_backend import (
    SQLiteVecEmbeddingBackend,
    create_sqlitevec_engine,
)

__all__ = [
    "SQLiteVecEmbeddingBackend",
    "create_sqlitevec_engine",
]
