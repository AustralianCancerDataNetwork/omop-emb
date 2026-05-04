"""Shared pgvector backend tests exercised via a generic ``backend`` fixture."""

import pytest
import numpy as np

from omop_emb.config import IndexType, MetricType
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter
from omop_emb.utils.errors import ModelRegistrationConflictError
from omop_emb.storage.base import EmbeddingBackend
from omop_emb.storage.index_config import index_config_from_index_type
from omop_emb.embeddings import EmbeddingRole
from .conftest import CONCEPTS, MODEL_NAME, EMBEDDING_DIM, TEST_CONCEPT_EMB, PROVIDER_TYPE


class SharedBackendTests:
    """Base class containing backend-parity tests via a generic ``backend`` fixture."""

    def test_backend_registration(self, session, backend: EmbeddingBackend, index_type: IndexType = IndexType.FLAT):
        model = backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=index_config_from_index_type(index_type),
            dimensions=EMBEDDING_DIM,
        )
        assert model.model_name == MODEL_NAME
        assert model.dimensions == EMBEDDING_DIM

    def test_get_registered_model(self, session, backend: EmbeddingBackend, index_type: IndexType = IndexType.FLAT):
        backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=index_config_from_index_type(index_type),
            dimensions=EMBEDDING_DIM,
        )
        retrieved = backend.get_registered_model(
            model_name=MODEL_NAME, index_type=index_type, provider_type=PROVIDER_TYPE
        )
        assert retrieved is not None
        assert retrieved.model_name == MODEL_NAME

    def test_duplicate_registration_identical_success(self, session, backend: EmbeddingBackend, index_type: IndexType = IndexType.FLAT):
        params = dict(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            dimensions=EMBEDDING_DIM,
            index_config=index_config_from_index_type(index_type),
            metadata={"version": "1.0"},
        )
        model1 = backend.register_model(**params)
        model2 = backend.register_model(**params)
        assert model1.storage_identifier == model2.storage_identifier
        assert model2.metadata["version"] == "1.0"

    def test_registration_metadata_conflict_raises_error(self, session, backend: EmbeddingBackend, index_type: IndexType = IndexType.FLAT):
        params = dict(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            dimensions=EMBEDDING_DIM,
            index_config=index_config_from_index_type(index_type),
            metadata={"version": "1.0"},
        )
        backend.register_model(**params)
        with pytest.raises(ModelRegistrationConflictError) as excinfo:
            backend.register_model(**{**params, "metadata": {"version": "2.0"}})
        assert excinfo.value.conflict_field == "metadata"

    def test_upsert_embeddings(self, session, backend: EmbeddingBackend, mock_llm_client, index_type: IndexType = IndexType.FLAT):
        backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=index_config_from_index_type(index_type),
            dimensions=EMBEDDING_DIM,
        )
        test_concept = CONCEPTS["Hypertension"]
        embeddings = mock_llm_client.embeddings(
            test_concept.concept_name, embedding_role=EmbeddingRole.DOCUMENT
        )
        backend.upsert_embeddings(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=index_type,
            concept_ids=(test_concept.concept_id,),
            embeddings=embeddings,
        )
        assert backend.has_any_embeddings(
            model_name=MODEL_NAME, index_type=index_type, provider_type=PROVIDER_TYPE
        )

    def test_get_embeddings_by_ids(self, session, backend: EmbeddingBackend, mock_llm_client, index_type: IndexType = IndexType.FLAT):
        backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=index_config_from_index_type(index_type),
            dimensions=EMBEDDING_DIM,
        )
        test_concepts = [CONCEPTS["Hypertension"], CONCEPTS["Diabetes"]]
        concept_ids = [c.concept_id for c in test_concepts]
        embeddings = mock_llm_client.embeddings(
            [c.concept_name for c in test_concepts], embedding_role=EmbeddingRole.DOCUMENT
        )
        backend.upsert_embeddings(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=index_type,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )
        retrieved = backend.get_embeddings_by_concept_ids(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            concept_ids=tuple(concept_ids),
            index_type=index_type,
        )
        assert len(retrieved) == 2
        assert set(retrieved.keys()) == set(concept_ids)

    def test_nearest_neighbor_search(self, session, backend: EmbeddingBackend, mock_llm_client, index_type: IndexType = IndexType.FLAT):
        backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=index_config_from_index_type(index_type),
            dimensions=EMBEDDING_DIM,
        )
        test_concepts = list(CONCEPTS.values())
        concept_ids = [c.concept_id for c in test_concepts]
        embeddings = mock_llm_client.embeddings(
            [c.concept_name for c in test_concepts], embedding_role=EmbeddingRole.DOCUMENT
        )
        backend.upsert_embeddings(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=index_type,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )
        query_embeddings = mock_llm_client.embeddings(
            "Hypertension", embedding_role=EmbeddingRole.QUERY
        )
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=index_type,
            query_embeddings=query_embeddings,
            metric_type=MetricType.L2,
            concept_filter=EmbeddingConceptFilter(limit=1),
        )
        assert len(results) == 1
        assert len(results[0]) == 1
        assert results[0][0].concept_id == CONCEPTS["Hypertension"].concept_id

    def test_nearest_neighbor_with_domain_filter(self, session, backend: EmbeddingBackend, mock_llm_client, index_type: IndexType = IndexType.FLAT):
        backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=index_config_from_index_type(index_type),
            dimensions=EMBEDDING_DIM,
        )
        test_concepts = list(CONCEPTS.values())
        concept_ids = [c.concept_id for c in test_concepts]
        embeddings = mock_llm_client.embeddings(
            [c.concept_name for c in test_concepts], embedding_role=EmbeddingRole.DOCUMENT
        )
        backend.upsert_embeddings(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=index_type,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=index_type,
            query_embeddings=TEST_CONCEPT_EMB,
            concept_filter=EmbeddingConceptFilter(domains=("Condition",), limit=10),
            metric_type=MetricType.L2,
        )
        expected_ids = {CONCEPTS["Hypertension"].concept_id, CONCEPTS["Diabetes"].concept_id}
        assert len(results) == 1
        assert set(m.concept_id for m in results[0]) == expected_ids

    def test_nearest_neighbor_with_vocabulary_filter(self, session, backend: EmbeddingBackend, mock_llm_client, index_type: IndexType = IndexType.FLAT):
        backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=index_config_from_index_type(index_type),
            dimensions=EMBEDDING_DIM,
        )
        test_concepts = list(CONCEPTS.values())
        concept_ids = [c.concept_id for c in test_concepts]
        embeddings = mock_llm_client.embeddings(
            [c.concept_name for c in test_concepts], embedding_role=EmbeddingRole.DOCUMENT
        )
        backend.upsert_embeddings(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=index_type,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=index_type,
            query_embeddings=TEST_CONCEPT_EMB,
            concept_filter=EmbeddingConceptFilter(vocabularies=("RxNorm",), limit=10),
            metric_type=MetricType.L2,
        )
        assert len(results) == 1
        assert len(results[0]) == 1
        assert results[0][0].concept_id == CONCEPTS["Aspirin"].concept_id

    def test_get_concepts_without_embedding(self, session, backend: EmbeddingBackend, mock_llm_client, index_type: IndexType = IndexType.FLAT):
        backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=index_config_from_index_type(index_type),
            dimensions=EMBEDDING_DIM,
        )
        test_concept = CONCEPTS["Hypertension"]
        embeddings = mock_llm_client.embeddings(
            test_concept.concept_name, embedding_role=EmbeddingRole.DOCUMENT
        )
        backend.upsert_embeddings(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=index_type,
            concept_ids=(test_concept.concept_id,),
            embeddings=embeddings,
        )
        unembedded = backend.get_concepts_without_embedding(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=index_type,
        )
        assert CONCEPTS["Aspirin"].concept_id in unembedded
        assert CONCEPTS["Diabetes"].concept_id in unembedded
        assert CONCEPTS["Hypertension"].concept_id not in unembedded

    def test_l2_similarity_exact_values(self, session, backend: EmbeddingBackend, mock_llm_client, index_type: IndexType = IndexType.FLAT):
        """L2 distance checks with deterministic 1-D embeddings.

        TEST_CONCEPT_EMB = [-1.0]:
        - Hypertension = [-10.0] → L2 = 9.0 → sim = 1/(1+9) = 0.1
        - Diabetes     = [0.0]   → L2 = 1.0 → sim = 1/(1+1) = 0.5
        - Aspirin      = [10.0]  → L2 = 11.0 → sim = 1/(1+11) ≈ 0.0833
        """
        backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=index_config_from_index_type(index_type),
            dimensions=EMBEDDING_DIM,
        )
        test_concepts = list(CONCEPTS.values())
        concept_ids = [c.concept_id for c in test_concepts]
        embeddings = mock_llm_client.embeddings(
            [c.concept_name for c in test_concepts], embedding_role=EmbeddingRole.DOCUMENT
        )
        backend.upsert_embeddings(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=index_type,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=index_type,
            query_embeddings=TEST_CONCEPT_EMB,
            metric_type=MetricType.L2,
            concept_filter=EmbeddingConceptFilter(limit=10),
        )
        expected = {
            CONCEPTS["Hypertension"].concept_id: 0.1,
            CONCEPTS["Diabetes"].concept_id: 0.5,
            CONCEPTS["Aspirin"].concept_id: 1.0 / 12.0,
        }
        for match in results[0]:
            assert match.concept_id in expected
            assert np.isclose(match.similarity, expected[match.concept_id], rtol=1e-5), (
                f"concept_id {match.concept_id}: got {match.similarity}, expected {expected[match.concept_id]}"
            )

    def test_cosine_similarity_exact_values(self, session, backend: EmbeddingBackend, mock_llm_client, index_type: IndexType = IndexType.FLAT):
        """Cosine similarity checks with deterministic 1-D embeddings.

        Query [-1.0] normalized → [-1.0]:
        - Hypertension [-10.0] → normalized [-1.0] → cos_sim = 1.0 → sim = 1.0
        - Aspirin [10.0]       → normalized [1.0]  → cos_sim = -1.0 → sim = 0.0
        """
        backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=index_config_from_index_type(index_type),
            dimensions=EMBEDDING_DIM,
        )
        test_concepts = list(CONCEPTS.values())
        concept_ids = [c.concept_id for c in test_concepts]
        embeddings = mock_llm_client.embeddings(
            [c.concept_name for c in test_concepts], embedding_role=EmbeddingRole.DOCUMENT
        )
        backend.upsert_embeddings(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=index_type,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_type=index_type,
            query_embeddings=TEST_CONCEPT_EMB,
            metric_type=MetricType.COSINE,
            concept_filter=EmbeddingConceptFilter(limit=10),
        )
        expected = {
            CONCEPTS["Hypertension"].concept_id: 1.0,
            CONCEPTS["Aspirin"].concept_id: 0.0,
        }
        for match in results[0]:
            if match.concept_id in expected:
                assert np.isclose(match.similarity, expected[match.concept_id], atol=1e-5)
