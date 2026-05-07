"""Optional FAISS sidecar cache.

``faiss-cpu`` is an optional dependency.  This package is importable without
it installed; ``ImportError`` is raised only when
:class:`~omop_emb.storage.faiss.faiss_cache.FAISSCache` is instantiated.
"""
from omop_emb.storage.faiss.faiss_cache import FAISSCache, IVFFlatIndexConfig, IVFPQIndexConfig

__all__ = ["FAISSCache", "IVFFlatIndexConfig", "IVFPQIndexConfig"]
