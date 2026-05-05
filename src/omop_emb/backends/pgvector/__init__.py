"""pgvector backend (optional).

Requires: ``pip install omop-emb[pgvector]``
"""
try:
    from omop_emb.backends.pgvector.pg_backend import PGVectorEmbeddingBackend
    from omop_emb.backends.pgvector.pg_index_manager import (
        PGVectorFlatIndexManager,
        PGVectorHNSWIndexManager,
    )
except ImportError as _e:
    raise ImportError(
        "The pgvector backend requires additional dependencies.  "
        "Install with: pip install omop-emb[pgvector]"
    ) from _e

__all__ = [
    "PGVectorEmbeddingBackend",
    "PGVectorFlatIndexManager",
    "PGVectorHNSWIndexManager",
]
