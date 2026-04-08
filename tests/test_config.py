"""Tests for backend configuration and factory."""

import pytest

from omop_emb.backends.factory import normalize_backend_name, get_embedding_backend
from omop_emb.config import BackendType, IndexType, MetricType


@pytest.mark.unit
class TestBackendConfig:
    """Test backend configuration."""
    
    def test_normalize_backend_name(self):
        """Test backend name normalization."""
        assert normalize_backend_name("faiss") == BackendType.FAISS
        assert normalize_backend_name("pgvector") == BackendType.PGVECTOR
    
    def test_get_faiss_backend(self, tmp_path):
        """Test getting FAISS backend."""
        backend = get_embedding_backend("faiss", storage_base_dir=str(tmp_path))
        assert backend.backend_type == BackendType.FAISS
    
    def test_faiss_supports_flat_index(self):
        """Test FAISS supports FLAT index."""
        from omop_emb.config import is_index_type_supported_for_backend
        
        assert is_index_type_supported_for_backend(BackendType.FAISS, IndexType.FLAT)
    
    def test_faiss_supports_cosine_metric(self):
        """Test FAISS supports COSINE metric."""
        from omop_emb.config import is_supported_index_metric_combination_for_backend
        
        assert is_supported_index_metric_combination_for_backend(
            BackendType.FAISS,
            IndexType.FLAT,
            MetricType.COSINE,
        )
