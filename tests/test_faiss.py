"""Tests for FAISS backend."""

import pytest
import numpy as np

from omop_emb.backends.config import IndexType, MetricType
from omop_emb.backends.embedding_utils import EmbeddingConceptFilter
from .conftest import CONCEPTS, MODEL_NAME, EMBEDDING_DIM, TEST_CONCEPT_EMB


@pytest.mark.faiss
@pytest.mark.unit
class TestFaissBackend:
    """Test FAISS embedding backend."""
    
    def test_backend_registration(self, session, faiss_backend):
        """Test registering model with FAISS backend."""
        model = faiss_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=IndexType.FLAT,
        )
        
        assert model.model_name == MODEL_NAME
        assert model.dimensions == EMBEDDING_DIM
    
    def test_get_registered_model(self, session, faiss_backend):
        """Test retrieving registered model."""
        faiss_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=IndexType.FLAT,
        )
        
        retrieved = faiss_backend.get_registered_model(session, MODEL_NAME)
        
        assert retrieved is not None
        assert retrieved.model_name == MODEL_NAME
    
    def test_upsert_embeddings(self, session, faiss_backend, mock_llm_client):
        """Test upserting embeddings."""
        faiss_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=IndexType.FLAT,
        )
        
        test_concept = CONCEPTS["Hypertension"]
        concept_ids = (test_concept.concept_id,)
        embeddings = mock_llm_client.embeddings(test_concept.concept_name)
        
        faiss_backend.upsert_embeddings(
            session=session,
            model_name=MODEL_NAME,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )
        
        assert faiss_backend.has_any_embeddings(session, MODEL_NAME)
    
    def test_get_embeddings_by_ids(self, session, faiss_backend, mock_llm_client):
        """Test retrieving embeddings by concept IDs."""
        faiss_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=IndexType.FLAT,
        )
        
        test_concepts = [CONCEPTS["Hypertension"], CONCEPTS["Diabetes"]]
        concept_ids = [c.concept_id for c in test_concepts]
        embeddings = mock_llm_client.embeddings([c.concept_name for c in test_concepts])
        
        faiss_backend.upsert_embeddings(
            session=session,
            model_name=MODEL_NAME,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )
        
        retrieved = faiss_backend.get_embeddings_by_concept_ids(
            session=session,
            model_name=MODEL_NAME,
            concept_ids=concept_ids,
        )
        
        assert len(retrieved) == 2
        assert set(retrieved.keys()) == set(concept_ids)
    
    def test_nearest_neighbor_search(self, session, faiss_backend, mock_llm_client):
        """Test nearest neighbor search returns exact top-1 for known query."""
        faiss_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=IndexType.FLAT,
        )
        
        test_concepts = list(CONCEPTS.values())
        concept_ids = [c.concept_id for c in test_concepts]
        embeddings = mock_llm_client.embeddings([c.concept_name for c in test_concepts])
        
        faiss_backend.upsert_embeddings(
            session=session,
            model_name=MODEL_NAME,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )
        
        # Query
        query_embeddings = mock_llm_client.embeddings("Hypertension")
        results = faiss_backend.get_nearest_concepts(
            session=session,
            model_name=MODEL_NAME,
            query_embeddings=query_embeddings,
            metric_type=MetricType.L2,
            k=1,
        )
        
        assert len(results) == 1  # One query vector
        assert len(results[0]) == 1  # One nearest neighbor
        assert results[0][0].concept_id == CONCEPTS["Hypertension"].concept_id
    
    def test_nearest_neighbor_with_domain_filter(self, session, faiss_backend, mock_llm_client):
        """Test nearest neighbor search with domain filter."""
        faiss_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=IndexType.FLAT,
        )
        
        test_concepts = list(CONCEPTS.values())
        concept_ids = [c.concept_id for c in test_concepts]
        embeddings = mock_llm_client.embeddings([c.concept_name for c in test_concepts])
        
        faiss_backend.upsert_embeddings(
            session=session,
            model_name=MODEL_NAME,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )
        
        # Query filtered to Condition domain
        results = faiss_backend.get_nearest_concepts(
            session=session,
            model_name=MODEL_NAME,
            query_embeddings=TEST_CONCEPT_EMB,
            concept_filter=EmbeddingConceptFilter(domains=("Condition",)),
            metric_type=MetricType.L2,
            k=10,
        )
        
        expected_ids = {CONCEPTS["Hypertension"].concept_id, CONCEPTS["Diabetes"].concept_id}
        assert len(results) == 1 # One query vector
        assert set(m.concept_id for m in results[0]) == expected_ids
    
    def test_nearest_neighbor_with_vocabulary_filter(self, session, faiss_backend, mock_llm_client):
        """Test nearest neighbor search with vocabulary filter."""
        faiss_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=IndexType.FLAT,
        )
        
        test_concepts = list(CONCEPTS.values())
        concept_ids = [c.concept_id for c in test_concepts]
        embeddings = mock_llm_client.embeddings([c.concept_name for c in test_concepts])
        
        faiss_backend.upsert_embeddings(
            session=session,
            model_name=MODEL_NAME,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )
        
        # Query filtered to RxNorm vocabulary
        results = faiss_backend.get_nearest_concepts(
            session=session,
            model_name=MODEL_NAME,
            query_embeddings=TEST_CONCEPT_EMB,
            concept_filter=EmbeddingConceptFilter(vocabularies=("RxNorm",)),
            metric_type=MetricType.L2,
            k=10,
        )
        assert len(results) == 1 # One query vector
        assert len(results[0]) == 1  # Only one match
        assert results[0][0].concept_id == CONCEPTS["Aspirin"].concept_id
    
    def test_get_concepts_without_embedding(self, session, faiss_backend, mock_llm_client):
        """Test retrieving concepts without embeddings."""
        faiss_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=IndexType.FLAT,
        )
        
        test_concept = CONCEPTS["Hypertension"]
        concept_ids = (test_concept.concept_id,)
        embeddings = mock_llm_client.embeddings(test_concept.concept_name)

        faiss_backend.upsert_embeddings(
            session=session,
            model_name=MODEL_NAME,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )
        
        # Get unembedded concepts
        unembedded = faiss_backend.get_concepts_without_embedding(
            session=session,
            model_name=MODEL_NAME,
        )
        
        assert CONCEPTS["Aspirin"].concept_id in unembedded
        assert CONCEPTS["Diabetes"].concept_id in unembedded
        assert CONCEPTS["Hypertension"].concept_id not in unembedded
