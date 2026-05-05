"""Optional FAISS sidecar cache.

FAISS and h5py are optional dependencies.  This package is importable without
them installed; ``ImportError`` is raised only when
:class:`~omop_emb.storage.faiss.faiss_index_store.FAISSCache` is instantiated.
"""
from omop_emb.storage.faiss.faiss_index_store import FAISSCache

__all__ = ["FAISSCache"]
