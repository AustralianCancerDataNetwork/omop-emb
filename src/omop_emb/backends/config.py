from enum import StrEnum
from typing import Dict, Tuple

class BackendType(StrEnum):
    PGVECTOR = "pgvector"
    FAISS = "faiss"

class IndexType(StrEnum):
    FLAT = "flat"
    HNSW = "hnsw"
    IVF_FLAT = "ivf_flat"
    IVF_PQ = "ivf_pq"
    DISKANN = "diskann"

BACKEND_SUPPORTED_INDICES: Dict[BackendType, Tuple[IndexType, ...]] = {
    BackendType.PGVECTOR: (IndexType.HNSW, IndexType.FLAT, IndexType.DISKANN),
    BackendType.FAISS: (IndexType.FLAT, IndexType.IVF_FLAT, IndexType.IVF_PQ, IndexType.HNSW)
}