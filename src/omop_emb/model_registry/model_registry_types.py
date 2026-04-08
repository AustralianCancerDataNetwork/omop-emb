from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Mapping

from omop_emb.config import (
    IndexType, 
    BackendType
)

@dataclass(frozen=True)
class EmbeddingModelRecord:
    """
    Canonical description of a registered embedding model.

    ``storage_identifier`` is intentionally backend-specific. For example:
    - PostgreSQL backend: dynamic embedding table name
    - FAISS backend: on-disk index path or logical collection name
    """

    model_name: str
    dimensions: int
    backend_type: BackendType
    index_type: IndexType
    storage_identifier: str
    metadata: Mapping[str, object] = field(default_factory=dict)
