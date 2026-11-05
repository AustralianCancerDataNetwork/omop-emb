"""DuckDB VSS backend (optional).

Requires: ``pip install omop-emb[duckdbvss]``
"""

from omop_emb.backends.duckdbvss.duckdbvss_backend import DuckDBVSSEmbeddingBackend

__all__ = [
    "DuckDBVSSEmbeddingBackend",
]
