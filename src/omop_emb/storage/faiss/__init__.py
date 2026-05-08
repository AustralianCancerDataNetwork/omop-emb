"""Optional FAISS sidecar cache.

Requires: ``pip install omop-emb[faiss-cpu]``
"""
from omop_emb.storage.faiss.faiss_cache import FAISSCache, IVFFlatIndexConfig, IVFPQIndexConfig

__all__ = ["FAISSCache", "IVFFlatIndexConfig", "IVFPQIndexConfig"]
