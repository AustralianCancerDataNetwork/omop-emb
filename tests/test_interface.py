"""Tests for EmbeddingInterface."""

import pytest
import numpy as np
from unittest.mock import Mock

from omop_emb.interface import EmbeddingInterface
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter
from omop_emb.config import IndexType, MetricType
from omop_emb.backends.base import NearestConceptMatch
from .conftest import CONCEPTS, MODEL_NAME, EMBEDDING_DIM


@pytest.mark.unit
class TestInterface:
    """Test EmbeddingInterface core functionality."""
    
    def test_register_model(self, session, embedding_interface: EmbeddingInterface):
        """Test registering an embedding model."""
        model = embedding_interface.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=IndexType.FLAT,
        )
        
        assert model.model_name == MODEL_NAME
        assert model.dimensions == EMBEDDING_DIM
    
    def test_model_registration_idempotent(self, session, embedding_interface: EmbeddingInterface, index_type: IndexType = IndexType.FLAT):
        """Test registering same model twice returns existing."""
        m1 = embedding_interface.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=index_type,
        )
        
        m2 = embedding_interface.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=index_type,
        )
        
        assert m1.storage_identifier == m2.storage_identifier
    
    def test_is_model_registered(self, session, embedding_interface: EmbeddingInterface, index_type: IndexType = IndexType.FLAT):
        """Test checking model registration status."""
        assert not embedding_interface.is_model_registered(model_name=MODEL_NAME, index_type=index_type)
        
        embedding_interface.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=index_type,
        )
        
        assert embedding_interface.is_model_registered(model_name=MODEL_NAME, index_type=index_type)
    
    def test_embed_texts(self, embedding_interface: EmbeddingInterface):
        """Test embedding generation."""
        embeddings = embedding_interface.embed_texts(CONCEPTS["Hypertension"].concept_name)
        
        assert embeddings.shape == (1, EMBEDDING_DIM)
        assert embeddings.dtype == np.float32
    
    def test_embed_multiple_texts(self, embedding_interface: EmbeddingInterface):
        """Test embedding multiple texts."""
        texts = [c.concept_name for c in CONCEPTS.values()]
        embeddings = embedding_interface.embed_texts(texts)
        
        assert embeddings.shape == (len(texts), EMBEDDING_DIM)
    
    def test_embed_and_upsert(self, session, embedding_interface: EmbeddingInterface, index_type: IndexType = IndexType.FLAT):
        """Test embedding and upserting concepts."""
        embedding_interface.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=index_type,
        )
        
        test_concepts = [CONCEPTS["Hypertension"], CONCEPTS["Diabetes"]]
        concept_ids = tuple(c.concept_id for c in test_concepts)
        concept_texts = [c.concept_name for c in test_concepts]
        
        embeddings = embedding_interface.embed_and_upsert_concepts(
            session=session,
            model_name=MODEL_NAME,
            concept_ids=concept_ids,
            concept_texts=concept_texts,
            index_type=index_type,
        )
        
        assert embeddings.shape == (2, EMBEDDING_DIM)
        assert embedding_interface.has_any_embeddings(session, embedding_model_name=MODEL_NAME, index_type=index_type)
    
    def test_get_embeddings_by_concept_ids(self, session, embedding_interface: EmbeddingInterface, index_type: IndexType = IndexType.FLAT):
        """Test retrieving stored embeddings."""
        embedding_interface.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=index_type,
        )
        
        test_concepts = [CONCEPTS["Hypertension"], CONCEPTS["Diabetes"]]
        concept_ids = tuple(c.concept_id for c in test_concepts)
        concept_texts = [c.concept_name for c in test_concepts]
        
        embeddings = embedding_interface.embed_and_upsert_concepts(
            session=session,
            model_name=MODEL_NAME,
            index_type=index_type,
            concept_ids=concept_ids,
            concept_texts=concept_texts,
        )
        
        retrieved = embedding_interface.get_embeddings_by_concept_ids(
            session=session,
            embedding_model_name=MODEL_NAME,
            concept_ids=concept_ids,
            index_type=index_type,
        )
        
        assert len(retrieved) == 2
        assert 1 in retrieved and 2 in retrieved

    @pytest.mark.parametrize("backend_name", ["faiss", "pgvector"])
    def test_search_return_structure_for_backends(self, session, mock_llm_client, backend_name, index_type: IndexType = IndexType.FLAT):
        """Search returns tuple[dict[int, float], ...] for both backend modes."""
        mock_backend = Mock()
        mock_backend.get_nearest_concepts.return_value = (
            (
                NearestConceptMatch(
                    concept_id=101,
                    concept_name=f"{backend_name}-match-a",
                    similarity=0.91,
                    is_standard=True,
                    is_active=True,
                ),
                NearestConceptMatch(
                    concept_id=202,
                    concept_name=f"{backend_name}-match-b",
                    similarity=0.77,
                    is_standard=True,
                    is_active=True,
                ),
            ),
        )

        interface = EmbeddingInterface(
            embedding_client=mock_llm_client,
            backend=mock_backend,
        )

        query_embedding = np.zeros((1, EMBEDDING_DIM), dtype=np.float32)
        result = interface.get_nearest_concepts(
            session=session,
            model_name=MODEL_NAME,
            index_type=index_type,
            query_embedding=query_embedding,
            metric_type=MetricType.COSINE,
            concept_filter=EmbeddingConceptFilter(limit=2),
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert set(result[0].keys()) == {101, 202}
        assert all(isinstance(k, int) for k in result[0].keys())
        assert all(isinstance(v, float) for v in result[0].values())
