"""Optional FAISS sidecar cache.

Requires: ``pip install omop-emb[faiss-cpu]``
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omop_emb.storage.faiss.faiss_cache import FAISSCache, IVFFlatIndexConfig, IVFPQIndexConfig

__all__ = ["FAISSCache", "IVFFlatIndexConfig", "IVFPQIndexConfig"]

_FAISS_EXPORTS = frozenset(__all__)


def __getattr__(name: str):
    if name in _FAISS_EXPORTS:
        from omop_emb.storage.faiss.faiss_cache import (  # raises ImportError with install hint if absent
            FAISSCache,
            IVFFlatIndexConfig,
            IVFPQIndexConfig,
        )
        g = globals()
        g["FAISSCache"] = FAISSCache
        g["IVFFlatIndexConfig"] = IVFFlatIndexConfig
        g["IVFPQIndexConfig"] = IVFPQIndexConfig
        return g[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
