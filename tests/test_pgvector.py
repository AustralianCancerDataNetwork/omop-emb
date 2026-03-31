"""Tests for pgvector backend."""

import pytest
import numpy as np

from omop_emb.backends.config import IndexType, MetricType
from omop_emb.backends.embedding_utils import EmbeddingConceptFilter
from .conftest import CONCEPTS, MODEL_NAME, EMBEDDING_DIM, TEST_CONCEPT_EMB


@pytest.mark.pgvector
@pytest.mark.unit
class TestPGVectorBackend:
    """Test pgvector embedding backend."""

    def test_backend_registration(self, session, pgvector_backend):
        """Test registering model with pgvector backend."""
        model = pgvector_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=IndexType.FLAT,
        )

        assert model.model_name == MODEL_NAME
        assert model.dimensions == EMBEDDING_DIM

    def test_get_registered_model(self, session, pgvector_backend):
        """Test retrieving registered model."""
        pgvector_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=IndexType.FLAT,
        )

        retrieved = pgvector_backend.get_registered_model(session, MODEL_NAME)

        assert retrieved is not None
        assert retrieved.model_name == MODEL_NAME

    def test_upsert_embeddings(self, session, pgvector_backend, mock_llm_client):
        """Test upserting embeddings."""
        pgvector_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=IndexType.FLAT,
        )

        test_concept = CONCEPTS["Hypertension"]
        concept_ids = (test_concept.concept_id,)
        embeddings = mock_llm_client.embeddings(test_concept.concept_name)

        pgvector_backend.upsert_embeddings(
            session=session,
            model_name=MODEL_NAME,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )

        assert pgvector_backend.has_any_embeddings(session, MODEL_NAME)

    def test_get_embeddings_by_ids(self, session, pgvector_backend, mock_llm_client):
        """Test retrieving embeddings by concept IDs."""
        pgvector_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=IndexType.FLAT,
        )

        test_concepts = [CONCEPTS["Hypertension"], CONCEPTS["Diabetes"]]
        concept_ids = [c.concept_id for c in test_concepts]
        embeddings = mock_llm_client.embeddings([c.concept_name for c in test_concepts])

        pgvector_backend.upsert_embeddings(
            session=session,
            model_name=MODEL_NAME,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )

        retrieved = pgvector_backend.get_embeddings_by_concept_ids(
            session=session,
            model_name=MODEL_NAME,
            concept_ids=concept_ids,
        )

        assert len(retrieved) == 2
        assert set(retrieved.keys()) == set(concept_ids)

    def test_nearest_neighbor_search(self, session, pgvector_backend, mock_llm_client):
        """Test exact top-1 nearest-neighbor retrieval with deterministic embeddings."""
        pgvector_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=IndexType.FLAT,
        )

        test_concepts = list(CONCEPTS.values())
        concept_ids = [c.concept_id for c in test_concepts]
        embeddings = mock_llm_client.embeddings([c.concept_name for c in test_concepts])

        pgvector_backend.upsert_embeddings(
            session=session,
            model_name=MODEL_NAME,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )

        query_embeddings = mock_llm_client.embeddings("Hypertension")
        results = pgvector_backend.get_nearest_concepts(
            session=session,
            model_name=MODEL_NAME,
            query_embeddings=query_embeddings,
            metric_type=MetricType.L2,
            k=1,
        )

        assert len(results) == 1
        assert len(results[0]) == 1
        assert results[0][0].concept_id == CONCEPTS["Hypertension"].concept_id

    def test_nearest_neighbor_with_domain_filter(self, session, pgvector_backend, mock_llm_client):
        """Test nearest neighbor search with domain filter."""
        pgvector_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=IndexType.FLAT,
        )

        test_concepts = list(CONCEPTS.values())
        concept_ids = [c.concept_id for c in test_concepts]
        embeddings = mock_llm_client.embeddings([c.concept_name for c in test_concepts])

        pgvector_backend.upsert_embeddings(
            session=session,
            model_name=MODEL_NAME,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )

        results = pgvector_backend.get_nearest_concepts(
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

    def test_nearest_neighbor_with_vocabulary_filter(self, session, pgvector_backend, mock_llm_client):
        """Test nearest neighbor search with vocabulary filter."""
        pgvector_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=IndexType.FLAT,
        )

        test_concepts = list(CONCEPTS.values())
        concept_ids = [c.concept_id for c in test_concepts]
        embeddings = mock_llm_client.embeddings([c.concept_name for c in test_concepts])

        pgvector_backend.upsert_embeddings(
            session=session,
            model_name=MODEL_NAME,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )

        query_embeddings = mock_llm_client.embeddings("Hypertension")
        results = pgvector_backend.get_nearest_concepts(
            session=session,
            model_name=MODEL_NAME,
            query_embeddings=query_embeddings,
            concept_filter=EmbeddingConceptFilter(vocabularies=("RxNorm",)),
            metric_type=MetricType.L2,
            k=10,
        )

        assert len(results) == 1 # One query vector
        assert len(results[0]) == 1  # Only one match
        assert results[0][0].concept_id == CONCEPTS["Aspirin"].concept_id

    def test_get_concepts_without_embedding(self, session, pgvector_backend, mock_llm_client):
        """Test retrieving concepts without embeddings."""
        pgvector_backend.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=IndexType.FLAT,
        )

        test_concept = CONCEPTS["Hypertension"]
        concept_ids = (test_concept.concept_id,)
        embeddings = mock_llm_client.embeddings(test_concept.concept_name)

        pgvector_backend.upsert_embeddings(
            session=session,
            model_name=MODEL_NAME,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )

        unembedded = pgvector_backend.get_concepts_without_embedding(
            session=session,
            model_name=MODEL_NAME,
        )

        assert CONCEPTS["Aspirin"].concept_id in unembedded
        assert CONCEPTS["Diabetes"].concept_id in unembedded
        assert CONCEPTS["Hypertension"].concept_id not in unembedded
