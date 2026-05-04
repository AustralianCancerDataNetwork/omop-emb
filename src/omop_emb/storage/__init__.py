"""omop_emb storage layer.

The production backend is ``PGVectorEmbeddingBackend``.  FAISS is an optional
sidecar cache, not an alternative backend.
"""
from omop_emb.storage.base import EmbeddingBackend, ConceptIDEmbeddingBase, require_registered_model
from omop_emb.storage.index_config import (
    IndexConfig,
    FlatIndexConfig,
    HNSWIndexConfig,
    index_config_from_index_type,
    index_config_from_index_type_and_metadata,
    INDEX_CONFIG_METADATA_KEY,
    FAISS_CACHE_METADATA_KEY,
)
from omop_emb.storage.postgres.pg_backend import PGVectorEmbeddingBackend
from omop_emb.storage.postgres.pg_registry import EmbeddingModelRecord, PostgresRegistryManager

__all__ = [
    "EmbeddingBackend",
    "ConceptIDEmbeddingBase",
    "require_registered_model",
    "IndexConfig",
    "FlatIndexConfig",
    "HNSWIndexConfig",
    "index_config_from_index_type",
    "index_config_from_index_type_and_metadata",
    "INDEX_CONFIG_METADATA_KEY",
    "FAISS_CACHE_METADATA_KEY",
    "PGVectorEmbeddingBackend",
    "EmbeddingModelRecord",
    "PostgresRegistryManager",
]
