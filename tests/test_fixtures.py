"""Verify fixtures are working correctly."""

import pytest
import sqlalchemy as sa
from omop_alchemy.cdm.model.vocabulary import Concept
from .conftest import CONCEPTS, EMBEDDING_DIM
from omop_emb.embeddings import EmbeddingRole


@pytest.mark.unit
class TestFixtures:
    """Test that fixtures and database setup work."""
    
    def test_pg_engine_creates_concept_table(self, pg_engine):
        """Verify Concept table is created in PostgreSQL test database."""
        inspector = sa.inspect(pg_engine)
        tables = inspector.get_table_names()
        
        assert "concept" in tables, f"Concept table not found. Tables: {tables}"
    
    def test_concept_table_has_required_columns(self, pg_engine):
        """Verify Concept table has all required columns."""
        inspector = sa.inspect(pg_engine)
        columns = {col["name"] for col in inspector.get_columns("concept")}
        
        required = {
            "concept_id", "concept_name", "domain_id", "vocabulary_id",
            "concept_class_id", "standard_concept", "concept_code",
            "valid_start_date", "valid_end_date"
        }
        
        assert required.issubset(columns), f"Missing columns: {required - columns}"
    
    def test_session_fixture_works(self, session):
        """Verify session fixture provides working database connection."""
        # Should be able to query an empty table
        result = session.query(Concept).all()
        assert isinstance(result, list)
    
    def test_add_concepts_to_db_fixture(self, session):
        """Verify add_concepts_to_db helper works and data persists."""
        concepts = session.query(Concept).all()
        assert len(concepts) == len(CONCEPTS)
        
        # Verify each concept
        for original in CONCEPTS.values():
            queried = session.query(Concept).filter_by(
                concept_id=original.concept_id
            ).first()
            
            assert queried is not None
            assert queried.concept_name == original.concept_name
            assert queried.domain_id == original.domain_id
            assert queried.vocabulary_id == original.vocabulary_id
    
    def test_mock_llm_client_generates_embeddings(self, mock_llm_client):
        """Verify mock LLM client generates consistent embeddings."""
        emb1 = mock_llm_client.embeddings(CONCEPTS["Hypertension"].concept_name, embedding_role=EmbeddingRole.DOCUMENT)
        emb2 = mock_llm_client.embeddings(CONCEPTS["Hypertension"].concept_name, embedding_role=EmbeddingRole.DOCUMENT)
        
        # Should be deterministic
        assert (emb1 == emb2).all()
        
        # Should have correct shape
        assert emb1.shape == (1, mock_llm_client.embedding_dim)
    
    
    def test_embedding_interface_initializes(self, embedding_writer_interface, session):
        """Verify EmbeddingInterface initializes with mocks."""
        assert embedding_writer_interface is not None
        assert embedding_writer_interface.embedding_dim == EMBEDDING_DIM
