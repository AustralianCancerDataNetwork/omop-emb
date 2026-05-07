from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Optional

from omop_emb.config import BackendType, IndexType, MetricType, ProviderType
from omop_emb.backends.index_config import IndexConfig


@dataclass(frozen=True)
class EmbeddingModelRecord:
    """Immutable snapshot of one registered embedding model.

    Each record maps to exactly one row in the ``model_registry`` table,
    identified by ``(model_name, backend_type)``.

    Attributes
    ----------
    model_name : str
        Canonical model name including tag (e.g. ``'nomic-embed-text:v1.5'``).
    provider_type : ProviderType
        Provider that serves the model.
    backend_type : BackendType
        Embedding storage backend.
    index_config : IndexConfig
        Active index configuration. ``index_type`` and ``metric_type`` are
        derived from this field via properties.
    dimensions : int
        Embedding vector dimensionality.
    storage_identifier : str
        Physical table name. Unique per model; does not encode metric type.
    metadata : Mapping[str, object]
        Free-form operational data (e.g. FAISS cache info, user extras).
    created_at : datetime, optional
        Row creation timestamp (UTC).
    updated_at : datetime, optional
        Row last-updated timestamp (UTC).

    Notes
    -----
    ``metric_type`` is ``None`` when the model is registered with a FLAT
    index (exact scan). In that case any backend-supported metric is valid
    at query time. When ``metric_type`` is set (HNSW), queries must use
    that metric exactly.
    """

    model_name: str
    provider_type: ProviderType
    backend_type: BackendType
    index_config: IndexConfig
    dimensions: int
    storage_identifier: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def index_type(self) -> IndexType:
        """Index type derived from ``index_config``."""
        return self.index_config.index_type

    @property
    def metric_type(self) -> Optional[MetricType]:
        """Distance metric locked to the index, or ``None`` for FLAT."""
        return self.index_config.metric_type
