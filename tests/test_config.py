"""Tests for backend configuration and factory."""

import pytest
from sqlalchemy.orm import configure_mappers

from omop_emb.backends.factory import normalize_backend_name, get_embedding_backend
from omop_emb.backends.config import BackendType, IndexType, MetricType
from omop_emb.backends.registry import ModelRegistry


@pytest.mark.unit
class TestBackendConfig:
    """Test backend configuration."""
    
    def test_normalize_backend_name(self):
        """Test backend name normalization."""
        assert normalize_backend_name("faiss") == BackendType.FAISS
        assert normalize_backend_name("pgvector") == BackendType.PGVECTOR
    
    def test_get_faiss_backend(self, tmp_path):
        """Test getting FAISS backend."""
        backend = get_embedding_backend("faiss", faiss_base_dir=str(tmp_path))
        assert backend.backend_type == BackendType.FAISS
    
    def test_faiss_supports_flat_index(self):
        """Test FAISS supports FLAT index."""
        from omop_emb.backends.config import is_index_type_supported_for_backend
        
        assert is_index_type_supported_for_backend(BackendType.FAISS, IndexType.FLAT)
        assert is_index_type_supported_for_backend(BackendType.FAISS, IndexType.HNSW)
    
    def test_faiss_supports_cosine_metric(self):
        """Test FAISS supports COSINE metric."""
        from omop_emb.backends.config import is_supported_index_metric_combination_for_backend
        
        assert is_supported_index_metric_combination_for_backend(
            BackendType.FAISS,
            IndexType.FLAT,
            MetricType.COSINE,
        )
        assert is_supported_index_metric_combination_for_backend(
            BackendType.FAISS,
            IndexType.HNSW,
            MetricType.COSINE,
        )

    def test_model_registry_accepts_supported_faiss_index_type(self):
        configure_mappers()
        row = ModelRegistry(
            model_name="test-model",
            dimensions=128,
            storage_identifier="faiss_test_model",
            index_type=IndexType.HNSW,
            backend_type=BackendType.FAISS,
            details={},
        )

        assert row.index_type == IndexType.HNSW
