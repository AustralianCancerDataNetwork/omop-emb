"""sqlite-vec backend (default, no external dependencies)."""
from omop_emb.backends.sqlitevec.sqlitevec_backend import (
    SQLiteVecBackend,
    SQLiteVecTableSpec,
    create_sqlitevec_engine,
)

__all__ = [
    "SQLiteVecBackend",
    "SQLiteVecTableSpec",
    "create_sqlitevec_engine",
]
