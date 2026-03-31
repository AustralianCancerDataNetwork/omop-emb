# OMOP EMB Test Suite

## Overview

Minimal, focused test suite for omop-emb with FAISS backend testing.

## Requirements

### PostgreSQL Database

Tests require a running PostgreSQL instance. By default, tests connect to:
- **Host**: localhost
- **Port**: 5432
- **Database**: omop_emb_test
- **User**: postgres
- **Password**: postgres

### Using Docker

Start a PostgreSQL container for testing:

```bash
docker run --name omop-emb-test-db \
  -e POSTGRES_DB=omop_emb_test \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  -d postgres:15
```

### Custom Connection String

Override the connection string via environment variable:

```bash
export TEST_DATABASE_URL="postgresql+psycopg2://user:password@host:port/database"
pytest tests/
```

## Running Tests

### All tests
```bash
pytest tests/ -v
```

### Unit tests only
```bash
pytest tests/ -m unit
```

### FAISS backend tests
```bash
pytest tests/ -m faiss
```

### pgvector backend tests
```bash
pytest tests/ -m pgvector
```

### Specific test file
```bash
pytest tests/test_interface.py -v
```

## Installation

```bash
pip install -e ".[faiss]"
pip install pytest
```

## Test Files

- `conftest.py` - Fixtures and database setup
- `test_fixtures.py` - Fixture validation tests
- `test_interface.py` - EmbeddingInterface tests
- `test_faiss.py` - FAISS backend tests
- `test_pgvector.py` - pgvector backend tests
- `test_config.py` - Configuration and factory tests

## Fixtures

### Core Fixtures
- `pg_engine` - PostgreSQL connection (session scope)
- `session` - Clean database session per test

### Mock Fixtures
- `mock_llm_client` - Mock LLMClient with deterministic embeddings

### Backend Fixtures
- `temp_faiss_dir` - Temporary FAISS index directory
- `faiss_backend` - Initialized FAISS backend
- `embedding_interface` - Full EmbeddingInterface

### Test Data Helper
- `add_concepts_to_db()` - Helper function to load test concepts

## Test Data

Three test concepts are available:
1. **Concept 1**: Hypertension (Condition, SNOMED, Standard)
2. **Concept 2**: Diabetes (Condition, SNOMED, Standard)
3. **Concept 3**: Aspirin (Drug, RxNorm, Standard)

Load them in tests via:
```python
def test_something(session):
    add_concepts_to_db(session)
    # concepts now available
```

## Notes

- Tests use PostgreSQL only (no SQLite)
- Foreign key constraints are enforced (required data must be complete)
- Each test gets a clean, isolated session
- Concept table is truncated before each test
