"""Tests for EmbeddingInterface."""

import pytest
import numpy as np
from unittest.mock import Mock

from omop_emb.interface import EmbeddingWriterInterface
from omop_emb.embeddings import EmbeddingRole
from omop_emb.config import IndexType
from omop_emb.backends import FlatIndexConfig
from .conftest import CONCEPTS, MODEL_NAME, EMBEDDING_DIM


@pytest.mark.unit
class TestInterface:
    """Test EmbeddingInterface core functionality."""

    def test_register_model(self, session, embedding_writer_interface: EmbeddingWriterInterface):
        """Test registering an embedding model."""
        model = embedding_writer_interface.register_model(
            engine=session.bind,
            index_config=FlatIndexConfig()
        )

        assert model.model_name == MODEL_NAME
        assert model.dimensions == EMBEDDING_DIM

    def test_model_registration_idempotent(self, session, embedding_writer_interface: EmbeddingWriterInterface, index_type: IndexType = IndexType.FLAT):
        """Test registering same model twice returns existing."""
        m1 = embedding_writer_interface.register_model(
            engine=session.bind,
            index_config=FlatIndexConfig()
        )

        m2 = embedding_writer_interface.register_model(
            engine=session.bind,
            index_config=FlatIndexConfig()
        )

        assert m1.storage_identifier == m2.storage_identifier

    def test_is_model_registered(self, session, embedding_writer_interface: EmbeddingWriterInterface, index_type: IndexType = IndexType.FLAT):
        """Test checking model registration status."""
        assert not embedding_writer_interface.is_model_registered(index_type=index_type)

        embedding_writer_interface.register_model(
            engine=session.bind,
            index_config=FlatIndexConfig()
        )

        assert embedding_writer_interface.is_model_registered(index_type=index_type)

    def test_embed_texts(self, embedding_writer_interface: EmbeddingWriterInterface):
        """Test embedding generation."""
        embeddings = embedding_writer_interface.embed_texts(CONCEPTS["Hypertension"].concept_name, embedding_role=EmbeddingRole.DOCUMENT)

        assert embeddings.shape == (1, EMBEDDING_DIM)
        assert embeddings.dtype == np.float32

    def test_embed_multiple_texts(self, embedding_writer_interface: EmbeddingWriterInterface):
        """Test embedding multiple texts."""
        texts = [c.concept_name for c in CONCEPTS.values()]
        embeddings = embedding_writer_interface.embed_texts(texts, embedding_role=EmbeddingRole.DOCUMENT)

        assert embeddings.shape == (len(texts), EMBEDDING_DIM)

    def test_embed_and_upsert(self, session, embedding_writer_interface: EmbeddingWriterInterface, index_type: IndexType = IndexType.FLAT):
        """Test embedding and upserting concepts."""
        embedding_writer_interface.register_model(
            engine=session.bind,
            index_config=FlatIndexConfig()
        )

        test_concepts = [CONCEPTS["Hypertension"], CONCEPTS["Diabetes"]]
        concept_ids = tuple(c.concept_id for c in test_concepts)
        concept_texts = [c.concept_name for c in test_concepts]

        embeddings = embedding_writer_interface.embed_and_upsert_concepts(
            session=session,
            concept_ids=concept_ids,
            concept_texts=concept_texts,
            index_type=index_type,
        )

        assert embeddings.shape == (2, EMBEDDING_DIM)
        assert embedding_writer_interface.has_any_embeddings(session, index_type=index_type)

    def test_get_embeddings_by_concept_ids(self, session, embedding_writer_interface: EmbeddingWriterInterface, index_type: IndexType = IndexType.FLAT):
        """Test retrieving stored embeddings."""
        embedding_writer_interface.register_model(
            engine=session.bind,
            index_config=FlatIndexConfig()
        )

        test_concepts = [CONCEPTS["Hypertension"], CONCEPTS["Diabetes"]]
        concept_ids = tuple(c.concept_id for c in test_concepts)
        concept_texts = [c.concept_name for c in test_concepts]

        embeddings = embedding_writer_interface.embed_and_upsert_concepts(
            session=session,
            index_type=index_type,
            concept_ids=concept_ids,
            concept_texts=concept_texts,
        )

        retrieved = embedding_writer_interface.get_embeddings_by_concept_ids(
            session=session,
            concept_ids=concept_ids,
            index_type=index_type,
        )

        assert len(retrieved) == 2
        assert 1 in retrieved and 2 in retrieved
