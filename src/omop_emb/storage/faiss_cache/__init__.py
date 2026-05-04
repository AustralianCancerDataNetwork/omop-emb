"""Optional FAISS sidecar cache.

FAISS and h5py are optional dependencies.  This package is importable without
them installed; ``ImportError`` is raised only when
:class:`~omop_emb.storage.faiss_cache.faiss_cache.FAISSCache` is instantiated.
"""
from omop_emb.storage.faiss_cache.faiss_cache import FAISSCache

__all__ = ["FAISSCache"]
