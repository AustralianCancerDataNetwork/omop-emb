from .interface import EmbeddingWriterInterface, EmbeddingReaderInterface, list_registered_models
from .storage.base import EmbeddingBackend, ConceptIDEmbeddingBase
from .storage.faiss_cache import FAISSCache
from .embeddings import (
    EmbeddingClient,
    EmbeddingProvider,
    OllamaProvider,
    OpenAIProvider,
    get_provider_for_api_base,
)
from .config import (
    BackendType,
    IndexType,
    MetricType,
    ProviderType,
    VectorColumnType,
    parse_backend_type,
    parse_index_type,
    parse_metric_type,
    vector_column_type_for_dimensions,
)
from .storage import (
    IndexConfig,
    FlatIndexConfig,
    HNSWIndexConfig,
    PGVectorEmbeddingBackend,
    EmbeddingModelRecord,
)
from .utils.embedding_utils import EmbeddingConceptFilter, NearestConceptMatch

__all__ = [
    # Interfaces
    "EmbeddingWriterInterface",
    "EmbeddingReaderInterface",
    "list_registered_models",
    # Storage
    "EmbeddingBackend",
    "ConceptIDEmbeddingBase",
    "PGVectorEmbeddingBackend",
    "FAISSCache",
    # Index config
    "IndexConfig",
    "FlatIndexConfig",
    "HNSWIndexConfig",
    # Registry
    "EmbeddingModelRecord",
    # Embeddings
    "EmbeddingClient",
    "EmbeddingConceptFilter",
    "NearestConceptMatch",
    # Providers
    "EmbeddingProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "get_provider_for_api_base",
    # Enums & parsers
    "BackendType",
    "IndexType",
    "MetricType",
    "ProviderType",
    "VectorColumnType",
    "parse_backend_type",
    "parse_index_type",
    "parse_metric_type",
    "vector_column_type_for_dimensions",
]
