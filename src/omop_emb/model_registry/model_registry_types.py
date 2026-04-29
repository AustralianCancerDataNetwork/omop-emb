from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Mapping

from omop_emb.config import (
    IndexType,
    BackendType,
    ProviderType
)
from omop_emb.backends.index_config import (
    IndexConfig,
    index_config_from_index_type_and_metadata
)

@dataclass(frozen=True)
class EmbeddingModelRecord:
    """
    Canonical description of a registered embedding model.

    ``storage_identifier`` is intentionally backend-specific. For example:
    - PostgreSQL backend: dynamic embedding table name
    - FAISS backend: on-disk index path or logical collection name

    ``provider_type`` identifies which embedding provider was used to register
    the model (e.g. "OllamaProvider", "OpenAICompatProvider").
    """

    model_name: str
    provider_type: ProviderType
    dimensions: int
    backend_type: BackendType
    index_type: IndexType
    storage_identifier: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def index_config(self) -> IndexConfig:
        return index_config_from_index_type_and_metadata(self.index_type, self.metadata)