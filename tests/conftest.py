"""Test configuration and shared fixtures."""

from __future__ import annotations

import os
import time
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any, Optional
from unittest.mock import Mock
from dataclasses import dataclass, field

import pytest
import numpy as np
import sqlalchemy as sa
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker


from omop_alchemy.cdm.model.vocabulary import Concept
from orm_loader.helpers import Base
from omop_llm import LLMClient

from omop_emb.backends.faiss import FaissEmbeddingBackend
from omop_emb.backends.pgvector import PGVectorEmbeddingBackend
from omop_emb.backends.registry import ensure_model_registry_schema, ModelRegistry
from omop_emb.interface import EmbeddingInterface


TEST_DB_NAME = os.getenv("TEST_DATABASE_NAME", "test_omop_emb")
TEST_USERNAME = os.getenv("TEST_DB_USERNAME", "test")
TEST_PASSWORD = os.getenv("TEST_DB_PASSWORD", "test")

DB_HOST = os.getenv("TEST_DB_HOST", None)
DB_PORT = os.getenv("TEST_DB_PORT", None)
DB_DRIVER = os.getenv("TEST_DB_DRIVER", "postgresql+psycopg2")
DB_ADMIN_USER = os.getenv("POSTGRES_USER", "postgres")
DB_ADMIN_PASS = os.getenv("POSTGRES_PASSWORD", "postgres")


# ================ Fixtures ================


def _create_test_url() -> sa.URL:
    """Construct the test database URL from environment variables."""
    if DB_HOST is None or not (
        DB_PORT is not None and DB_PORT.isdigit()
    ):
        raise RuntimeError(
            "TEST_DB_HOST and TEST_DB_PORT must be set in the environment. "
            "Example: TEST_DB_HOST=db-omop TEST_DB_PORT=5432"
        )
    return sa.URL.create(
        drivername=DB_DRIVER,
        username=TEST_USERNAME,
        password=TEST_PASSWORD,
        host=DB_HOST,
        port=int(DB_PORT),
        database=TEST_DB_NAME
    )

def _create_admin_url() -> sa.URL:
    """Construct the admin database URL for managing test DB."""
    if DB_HOST is None or not (
        DB_PORT is not None and DB_PORT.isdigit()
    ):
        raise RuntimeError(
            "TEST_DB_HOST and TEST_DB_PORT must be set in the environment. "
            "Example: TEST_DB_HOST=db-omop TEST_DB_PORT=5432"
        )
    return sa.URL.create(
        drivername=DB_DRIVER,
        username=DB_ADMIN_USER,
        password=DB_ADMIN_PASS,
        host=DB_HOST,
        port=int(DB_PORT),
        database="postgres"
    )

def _create_test_database() -> sa.URL:
    """Create a dedicated test database and return its URL."""
    
    admin_url = _create_admin_url()
    test_url = _create_test_url()
    admin_engine = sa.create_engine(admin_url, future=True)
    try:
        with admin_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(sa.text(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}"'))
            conn.execute(sa.text(f"DROP ROLE IF EXISTS {TEST_USERNAME}"))
            conn.execute(sa.text(f"CREATE USER {TEST_USERNAME} WITH PASSWORD '{TEST_PASSWORD}' SUPERUSER"))
            conn.execute(sa.text(f'CREATE DATABASE "{TEST_DB_NAME}" OWNER {TEST_USERNAME}'))
            conn.execute(sa.text(f"DROP TABLE IF EXISTS {ModelRegistry.__tablename__} CASCADE;"))
            

    finally:
        admin_engine.dispose()
    return test_url


def _drop_test_database() -> None:
    """Drop the test database and the associated test user."""
    admin_url = _create_admin_url()
    admin_engine = sa.create_engine(admin_url, future=True)
    try:
        with admin_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(
                sa.text(
                    "SELECT pg_terminate_backend(pid) "
                    "FROM pg_stat_activity "
                    "WHERE datname = :db_name "
                    "AND pid <> pg_backend_pid()"
                ),
                {"db_name": TEST_DB_NAME},
            )
            
            conn.execute(sa.text(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}"'))
            conn.execute(sa.text(f"DROP ROLE IF EXISTS {TEST_USERNAME};"))
    finally:
        admin_engine.dispose()


@pytest.fixture(scope="session")
def pg_engine():
    """Create PostgreSQL engine with retry logic and dedicated test DB."""
    test_db_url = _create_test_database()

    max_retries = 20
    for attempt in range(max_retries):
        try:
            engine = sa.create_engine(test_db_url, echo=False, future=True)
            with engine.connect() as conn:
                conn.execute(sa.text("SELECT 1"))
            print(f"\n--- PostgreSQL connection established to {TEST_DB_NAME} ---")
            
            # Create all tables
            Base.metadata.create_all(engine)
            ensure_model_registry_schema(engine)
            
            yield engine
            
            engine.dispose()
            _drop_test_database()
            return

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[{attempt + 1}/{max_retries}] PostgreSQL not ready: {e}")
                time.sleep(1)
            else:
                _drop_test_database()
                raise RuntimeError(f"PostgreSQL never became available after {max_retries} attempts: {e}")


@pytest.fixture
def session(pg_engine) -> Generator[Session, None, None]:
    """Provide clean session for each test with rollback."""
    # Clear tables before test
    with pg_engine.connect() as conn:
        with conn.begin():
            res = conn.execute(sa.text(f"SELECT table_name FROM {ModelRegistry.__tablename__}"))
            for (table_name,) in res:
                conn.execute(sa.text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
            conn.execute(sa.text(f"TRUNCATE TABLE concept, {ModelRegistry.__tablename__} CASCADE"))
    
    Session = sessionmaker(bind=pg_engine, future=True)
    db_session = Session()

    # Add data
    add_concepts_to_db(db_session)
    
    yield db_session

    db_session.close()


@pytest.fixture
def mock_llm_client() -> Mock:
    """Mock LLMClient with deterministic, low-dimensional embeddings."""
    client = Mock(spec=LLMClient)
    client.embedding_dim = EMBEDDING_DIM
    
    def create_embeddings(concept_names: list[str] | str, batch_size: Optional[int] = None) -> np.ndarray:
        if isinstance(concept_names, str):
            concept_names = [concept_names]
        else:
            concept_names = list(concept_names)
        
        embeddings: list[np.ndarray] = [CONCEPTS[name].embeddings for name in concept_names]
        return np.vstack(embeddings).astype(np.float32)
    
    client.embeddings = Mock(side_effect=create_embeddings)
    return client


@pytest.fixture
def temp_faiss_dir():
    """Temporary directory for FAISS indices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def faiss_backend(session, temp_faiss_dir) -> FaissEmbeddingBackend:
    """FAISS backend with model registry initialized."""
    backend = FaissEmbeddingBackend(base_dir=temp_faiss_dir)
    backend.initialise_store(session.bind)
    return backend


@pytest.fixture
def pgvector_backend(session) -> PGVectorEmbeddingBackend:
    """PGVector backend with vector extension and model registry initialized."""
    backend = PGVectorEmbeddingBackend()
    backend.initialise_store(session.bind)
    return backend


@pytest.fixture
def embedding_interface(session, mock_llm_client, faiss_backend) -> EmbeddingInterface:
    """Full embedding interface ready for testing."""
    interface = EmbeddingInterface(
        embedding_client=mock_llm_client,
        backend=faiss_backend,
    )
    interface.initialise_store(session.bind)
    return interface


# ================ Test Data ================
@dataclass
class TestConcept:
    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    concept_code: str
    standard_concept: str
    concept_class_id: str
    valid_start_date: str
    valid_end_date: str
    embeddings: np.ndarray

    def to_db(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "domain_id": self.domain_id,
            "vocabulary_id": self.vocabulary_id,
            "concept_code": self.concept_code,
            "standard_concept": self.standard_concept,
            "concept_class_id": self.concept_class_id,
            "valid_start_date": self.valid_start_date,
            "valid_end_date": self.valid_end_date,
        }


CONCEPTS: Dict[str, TestConcept] = {
    "Hypertension": TestConcept(
        concept_id=1, concept_name="Hypertension", domain_id="Condition", 
        vocabulary_id="SNOMED", concept_code="38341003", standard_concept="S", 
        concept_class_id="Clinical Finding",
        valid_start_date="2000-01-01", valid_end_date="2099-12-31",
        embeddings=np.array([[-10.0]], dtype=np.float32)
    ),
    "Diabetes": TestConcept(
        concept_id=2, concept_name="Diabetes", domain_id="Condition",
        vocabulary_id="SNOMED", concept_code="73211009", standard_concept="S", 
        concept_class_id="Clinical Finding",
        valid_start_date="2000-01-01", valid_end_date="2099-12-31",
        embeddings=np.array([[0.0]], dtype=np.float32)
    ),
    "Aspirin": TestConcept(
        concept_id=3, concept_name="Aspirin", domain_id="Drug",
        vocabulary_id="RxNorm", concept_code="1191", standard_concept="S", 
        concept_class_id="Ingredient",
        valid_start_date="2000-01-01", valid_end_date="2099-12-31",
        embeddings=np.array([[10.0]], dtype=np.float32)
    ),
}

TEST_CONCEPT_EMB = np.array([[-1.0]], dtype=np.float32)


MODEL_NAME = "test-model"
EMBEDDING_DIM = 1


def add_concepts_to_db(session: Session):
    """Helper to add test concepts to database (disables FK checks temporarily)."""
    try:
        session.execute(sa.text("SET session_replication_role = 'replica';"))
        
        for concept_data in CONCEPTS.values():
            concept = Concept(**concept_data.to_db())
            session.add(concept)
        session.commit()
    finally:
        # Re-enable triggers/constraints
        session.execute(sa.text("SET session_replication_role = 'origin';"))
        session.commit()