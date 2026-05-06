"""sqlite-vec backend (default, no external dependencies)."""
from omop_emb.backends.sqlitevec.sqlitevec_backend import (
    SQLiteVecBackend,
    create_sqlitevec_engine,
)

__all__ = [
    "SQLiteVecBackend",
    "create_sqlitevec_engine",
]
