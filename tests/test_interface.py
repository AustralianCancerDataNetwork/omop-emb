"""Tests for EmbeddingInterface."""

import pytest
import numpy as np
from unittest.mock import Mock

from omop_emb.interface import EmbeddingInterface
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter
from omop_emb.config import BackendType, IndexType, MetricType
from omop_emb.backends.base import NearestConceptMatch
from omop_emb.model_registry import EmbeddingModelRecord
from omop_emb.utils.errors import ModelRegistrationConflictError
from .conftest import CONCEPTS, MODEL_NAME, EMBEDDING_DIM


class TestInterface:
    """Test EmbeddingInterface core functionality."""

    @pytest.mark.integration
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
    
    @pytest.mark.integration
    def test_model_registration_idempotent(self, session, embedding_interface: EmbeddingInterface):
        """Test registering same model twice returns existing."""
        index_type = IndexType.FLAT
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

    @pytest.mark.unit
    def test_model_registration_dimension_conflict_raises(self, mock_llm_client):
        backend = Mock()
        backend.get_registered_model.return_value = EmbeddingModelRecord(
            model_name=MODEL_NAME,
            dimensions=1024,
            backend_type=BackendType.FAISS,
            index_type=IndexType.FLAT,
            storage_identifier="faiss_test_model",
        )
        interface = EmbeddingInterface(
            embedding_client=mock_llm_client,
            backend=backend,
        )

        with pytest.raises(ModelRegistrationConflictError) as excinfo:
            interface.ensure_model_registered(
                engine=Mock(),
                session=Mock(),
                model_name=MODEL_NAME,
                dimensions=512,
                index_type=IndexType.FLAT,
            )

        assert excinfo.value.conflict_field == "dimensions"

    @pytest.mark.unit
    def test_model_registration_dimension_conflict_overwrites_when_requested(self, mock_llm_client):
        backend = Mock()
        backend.get_registered_model.return_value = EmbeddingModelRecord(
            model_name=MODEL_NAME,
            dimensions=1024,
            backend_type=BackendType.FAISS,
            index_type=IndexType.FLAT,
            storage_identifier="faiss_test_model",
        )
        backend.register_model.return_value = EmbeddingModelRecord(
            model_name=MODEL_NAME,
            dimensions=512,
            backend_type=BackendType.FAISS,
            index_type=IndexType.FLAT,
            storage_identifier="faiss_test_model",
        )
        interface = EmbeddingInterface(
            embedding_client=mock_llm_client,
            backend=backend,
        )
        engine = Mock()
        session = Mock()

        model = interface.ensure_model_registered(
            engine=engine,
            session=session,
            model_name=MODEL_NAME,
            dimensions=512,
            index_type=IndexType.FLAT,
            overwrite_existing_conflicts=True,
        )

        backend.delete_model.assert_called_once_with(
            engine=engine,
            session=session,
            model_name=MODEL_NAME,
        )
        backend.register_model.assert_called_once()
        assert model.dimensions == 512

    @pytest.mark.unit
    def test_model_registration_overwrite_rebuilds_even_without_conflict(self, mock_llm_client):
        backend = Mock()
        backend.register_model.return_value = EmbeddingModelRecord(
            model_name=MODEL_NAME,
            dimensions=512,
            backend_type=BackendType.FAISS,
            index_type=IndexType.FLAT,
            storage_identifier="faiss_test_model",
        )
        interface = EmbeddingInterface(
            embedding_client=mock_llm_client,
            backend=backend,
        )
        engine = Mock()
        session = Mock()

        model = interface.ensure_model_registered(
            engine=engine,
            session=session,
            model_name=MODEL_NAME,
            dimensions=512,
            index_type=IndexType.FLAT,
            overwrite_existing_conflicts=True,
        )

        backend.delete_model.assert_called_once_with(
            engine=engine,
            session=session,
            model_name=MODEL_NAME,
        )
        backend.get_registered_model.assert_not_called()
        backend.register_model.assert_called_once()
        assert model.dimensions == 512

    @pytest.mark.unit
    def test_model_registration_raises_for_stale_backend_artifacts(self, mock_llm_client):
        backend = Mock()
        backend.get_registered_model.return_value = None
        backend.has_stale_model_artifacts.return_value = True
        interface = EmbeddingInterface(
            embedding_client=mock_llm_client,
            backend=backend,
        )

        with pytest.raises(RuntimeError, match="overwrite-model-registration"):
            interface.ensure_model_registered(
                engine=Mock(),
                session=Mock(),
                model_name=MODEL_NAME,
                dimensions=512,
                index_type=IndexType.FLAT,
            )
    
    @pytest.mark.integration
    def test_is_model_registered(self, session, embedding_interface: EmbeddingInterface):
        """Test checking model registration status."""
        index_type = IndexType.FLAT
        assert not embedding_interface.is_model_registered(model_name=MODEL_NAME, index_type=index_type)
        
        embedding_interface.register_model(
            engine=session.bind,
            model_name=MODEL_NAME,
            dimensions=EMBEDDING_DIM,
            index_type=index_type,
        )
        
        assert embedding_interface.is_model_registered(model_name=MODEL_NAME, index_type=index_type)
    
    @pytest.mark.unit
    def test_embed_texts(self, mock_llm_client):
        """Test embedding generation."""
        embedding_interface = EmbeddingInterface(
            embedding_client=mock_llm_client,
            backend=Mock(),
        )
        embeddings = embedding_interface.embed_texts(CONCEPTS["Hypertension"].concept_name)
        
        assert embeddings.shape == (1, EMBEDDING_DIM)
        assert embeddings.dtype == np.float32
    
    @pytest.mark.unit
    def test_embed_multiple_texts(self, mock_llm_client):
        """Test embedding multiple texts."""
        embedding_interface = EmbeddingInterface(
            embedding_client=mock_llm_client,
            backend=Mock(),
        )
        texts = [c.concept_name for c in CONCEPTS.values()]
        embeddings = embedding_interface.embed_texts(texts)
        
        assert embeddings.shape == (len(texts), EMBEDDING_DIM)

    @pytest.mark.unit
    def test_embed_texts_applies_document_prefix_from_env(self, monkeypatch):
        captured: dict[str, object] = {}

        class FakeClient:
            embedding_dim = EMBEDDING_DIM

            def embeddings(self, texts, batch_size=None):
                captured["texts"] = texts
                return np.zeros((1, EMBEDDING_DIM), dtype=np.float32)

        monkeypatch.setenv(
            "OMOP_EMB_DOCUMENT_EMBEDDING_PREFIX",
            "Instruction: map concept. Passage: ",
        )
        embedding_interface = EmbeddingInterface(
            embedding_client=FakeClient(),
            backend=Mock(),
        )

        embedding_interface.embed_texts(CONCEPTS["Hypertension"].concept_name)

        assert captured["texts"] == "Instruction: map concept. Passage: Hypertension"

    @pytest.mark.unit
    def test_get_nearest_concepts_by_texts_applies_query_prefix_from_env(self, monkeypatch):
        captured: dict[str, object] = {}

        class FakeClient:
            embedding_dim = EMBEDDING_DIM

            def embeddings(self, texts, batch_size=None):
                captured["texts"] = texts
                return np.zeros((1, EMBEDDING_DIM), dtype=np.float32)

        backend = Mock()
        backend.get_nearest_concepts.return_value = ((),)
        embedding_interface = EmbeddingInterface(
            embedding_client=FakeClient(),
            backend=backend,
        )
        monkeypatch.setenv(
            "OMOP_EMB_QUERY_EMBEDDING_PREFIX",
            "Instruction: map concept. Query: ",
        )

        embedding_interface.get_nearest_concepts_by_texts(
            session=Mock(),
            embedding_model_name=MODEL_NAME,
            index_type=IndexType.FLAT,
            query_texts=CONCEPTS["Hypertension"].concept_name,
            metric_type=MetricType.COSINE,
        )

        assert captured["texts"] == ("Instruction: map concept. Query: Hypertension",)

    @pytest.mark.unit
    def test_rebuild_model_indexes_delegates_to_backend(self, mock_llm_client):
        backend = Mock()
        interface = EmbeddingInterface(
            embedding_client=mock_llm_client,
            backend=backend,
        )

        interface.rebuild_model_indexes(
            session=Mock(),
            model_name=MODEL_NAME,
            metric_types=(MetricType.COSINE,),
            batch_size=4096,
        )

        backend.rebuild_model_indexes.assert_called_once()
    
    @pytest.mark.integration
    def test_embed_and_upsert(self, session, embedding_interface: EmbeddingInterface):
        """Test embedding and upserting concepts."""
        index_type = IndexType.FLAT
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
    
    @pytest.mark.integration
    def test_get_embeddings_by_concept_ids(self, session, embedding_interface: EmbeddingInterface):
        """Test retrieving stored embeddings."""
        index_type = IndexType.FLAT
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

    @pytest.mark.unit
    @pytest.mark.parametrize("backend_name", ["faiss", "pgvector"])
    def test_search_return_structure_for_backends(self, mock_llm_client, backend_name):
        """Search returns tuple[dict[int, float], ...] for both backend modes."""
        index_type = IndexType.FLAT
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
            session=Mock(),
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
