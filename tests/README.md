# OMOP EMB Test Suite

## Overview

Minimal, focused test suite for omop-emb with FAISS backend testing.

## Requirements

### PostgreSQL Database

Integration tests require a running PostgreSQL instance. By default, tests connect to:
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

### Existing Database Mode

If you do not want to provide admin credentials, run the PostgreSQL tests inside
a dedicated schema of an existing database:

```bash
export TEST_DB_HOST=localhost
export TEST_DB_PORT=5432
export TEST_DATABASE_NAME=my_existing_database
export TEST_DB_USERNAME=my_app_user
export TEST_DB_PASSWORD=my_app_password
export TEST_DB_USE_EXISTING=1
export TEST_DB_SCHEMA=omop_emb_test
```

In this mode the harness does not create or drop databases or roles. It creates
tables only inside `TEST_DB_SCHEMA` and sets `search_path` accordingly.

## Running Tests

### All tests
```bash
pytest tests/ -v
```

### Unit tests only
```bash
pytest tests/ -m unit
```

### Integration tests only
```bash
pytest tests/ -m integration
```

### FAISS backend integration tests
```bash
pytest tests/ -m "faiss and integration"
```

### pgvector backend integration tests
```bash
pytest tests/ -m "pgvector and integration"
```

### FAISS-only users
```bash
pytest tests/ -m "unit and not pgvector"
```

This is the recommended default for FAISS-only users who do not want to run the
PostgreSQL integration suite.

### Specific test file
```bash
pytest tests/test_interface.py -v
```

## Installation

```bash
pip install -e ".[faiss]"
pip install pytest
```

If you want to run pgvector integration tests as well:

```bash
pip install -e ".[faiss,pgvector]"
```

## Test Files

- `conftest.py` - Fixtures and database setup
- `test_fixtures.py` - PostgreSQL fixture validation tests
- `test_interface.py` - Mixed unit and integration tests for `EmbeddingInterface`
- `test_faiss.py` - FAISS backend integration tests
- `test_pgvector.py` - pgvector backend integration tests
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
- `embedding_interface` - Initialized `EmbeddingInterface` backed by the FAISS fixture

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

- Backend and fixture integration tests require PostgreSQL
- Pure unit tests can be run with `pytest tests/ -m unit`
- `pytest -m "not pgvector"` still includes FAISS integration tests; use
  `pytest -m unit` if you want to avoid all database-backed tests
- Foreign key constraints are enforced (required data must be complete)
- Each test gets a clean, isolated session
- In integration tests, the harness truncates the test schema's `concept` and
  `model_registry` tables before each test. In `TEST_DB_USE_EXISTING=1` mode,
  this happens in `TEST_DB_SCHEMA`, not in your application schemas such as
  `vocabulary` or `staging_vocabulary`
