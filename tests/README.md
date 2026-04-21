# OMOP EMB Test Suite

## Overview

Focused test suite for `omop-emb`. Unit tests run without any external services; integration tests require PostgreSQL.

## Requirements

### PostgreSQL (integration tests only)

Set these environment variables before running integration tests:

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_DB_HOST` | *(required)* | PostgreSQL host |
| `TEST_DB_PORT` | *(required)* | PostgreSQL port |
| `TEST_DB_USERNAME` | `test` | Test user |
| `TEST_DB_PASSWORD` | `test` | Test password |
| `TEST_DATABASE_NAME` | `test_omop_emb` | Test database name |
| `POSTGRES_USER` | `postgres` | Admin user (for DB creation) |
| `POSTGRES_PASSWORD` | `postgres` | Admin password |

### Using Docker

```bash
docker run --name omop-emb-test-db \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  -d postgres:15

export TEST_DB_HOST=localhost
export TEST_DB_PORT=5432
```

## Running Tests

```bash
# Unit tests only (no PostgreSQL needed)
uv run pytest tests/ -m unit -v

# All tests (requires PostgreSQL env vars)
uv run pytest tests/ -v

# FAISS backend tests
uv run pytest tests/ -m faiss -v

# pgvector backend tests
uv run pytest tests/ -m pgvector -v

# Single file
uv run pytest tests/test_embedding_client.py -v
```

## Test Files

| File | What it tests | External services |
|------|--------------|-------------------|
| `test_providers.py` | `OllamaProvider`, `OpenAIProvider`, `get_provider_for_api_base`, `get_provider_from_provider_type` | None |
| `test_embedding_client.py` | `EmbeddingClient` — construction, batching, similarity, euclidean distance | None (OpenAI mocked) |
| `test_ollama_provider_api.py` | `OllamaProvider.get_embedding_dim()` HTTP interaction | None (requests mocked) |
| `test_interface_validation.py` | `EmbeddingWriterInterface` input contracts, canonical name enforcement | None |
| `test_interface.py` | `EmbeddingWriterInterface` end-to-end write/read flows | PostgreSQL + FAISS |
| `test_config.py` | Backend/index/metric config factories | None |
| `test_fixtures.py` | Fixture health checks | PostgreSQL |
| `test_faiss.py` | FAISS backend (inherits `SharedBackendTests`) | PostgreSQL |
| `test_pgvector.py` | pgvector backend (inherits `SharedBackendTests`) | PostgreSQL |
| `shared_backend_tests.py` | Shared backend contract tests (not a test file itself) | — |

## Fixtures

### Database fixtures
- `pg_engine` *(session scope)* — creates a dedicated test PostgreSQL database; drops it on teardown
- `session` — clean SQLAlchemy session per test; truncates `concept` table before each test

### Mock fixtures
- `mock_llm_client` — `Mock(spec=EmbeddingClient)` with deterministic 1-D embeddings for the three test concepts; uses `ProviderType.OLLAMA`

### Backend fixtures
- `faiss_backend` — `FaissEmbeddingBackend` with model registry initialised
- `pgvector_backend` — `PGVectorEmbeddingBackend` with vector extension and model registry initialised
- `temp_storage_dir` — temporary directory for FAISS files and `metadata.db`

### Interface fixtures
- `embedding_writer_interface` — `EmbeddingWriterInterface` backed by FAISS, using `mock_llm_client`
- `registered_embedding_writer_interface` — writer interface with a pre-registered `FLAT` model
- `embedding_reader_interface` — `EmbeddingReaderInterface` pointing at the same FAISS registry

## Test Data

Three concepts are pre-loaded by the `session` fixture:

| ID | Name | Domain | Embedding |
|----|------|--------|-----------|
| 1 | Hypertension | Condition | `[[-10.0]]` |
| 2 | Diabetes | Condition | `[[0.0]]` |
| 3 | Aspirin | Drug | `[[10.0]]` |

Constants exported from `conftest.py`:

```python
MODEL_NAME     = "test-model:v1"
PROVIDER_TYPE  = ProviderType.OLLAMA
EMBEDDING_DIM  = 1
```
