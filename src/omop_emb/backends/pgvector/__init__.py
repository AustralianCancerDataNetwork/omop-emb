"""pgvector backend (optional).

Requires: ``pip install omop-emb[pgvector]``

The :exc:`ImportError` raised when pgvector is absent originates in
``pg_backend.py`` and includes the install hint.  It is intentionally not
re-wrapped here so that both direct and package-level imports surface the
same message.
"""
from omop_emb.backends.pgvector.pg_backend import PGVectorEmbeddingBackend
from omop_emb.backends.pgvector.pg_index_manager import (
    PGVectorFlatIndexManager,
    PGVectorHNSWIndexManager,
)

__all__ = [
    "PGVectorEmbeddingBackend",
    "PGVectorFlatIndexManager",
    "PGVectorHNSWIndexManager",
]
