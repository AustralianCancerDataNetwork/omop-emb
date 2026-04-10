# omop-emb
Embedding layer for OMOP CDM.

## Installation

`omop-emb` now exposes backend-specific optional dependencies so installation
can match the embedding backend you actually intend to use.

```bash
pip install "omop-emb[pgvector]"
pip install "omop-emb[faiss]"
pip install "omop-emb[all]"
```

Notes:

- `pgvector` installs the PostgreSQL/pgvector dependencies.
- `faiss` installs the FAISS-based backend dependencies. This currently only includes CPU support
- `all` installs both backend stacks for development or mixed environments.
- A plain `pip install omop-emb` installs the shared core package only.
- PostgreSQL-specific embedding dependencies are now optional, but `omop-emb`
  still requires some database backend for OMOP access and model registration.
- Non-PostgreSQL database backends have not yet been tested.

Extended documentation can be found [here](https://AustralianCancerDataNetwork.github.io/omop-emb).

## Quick Start

`omop-emb` reads OMOP concepts through SQLAlchemy and writes embedding metadata
through PostgreSQL even when you use the `faiss` backend for vector storage.

Example:

```bash
omop-emb add-embeddings \
  --api-base http://localhost:8000/v1 \
  --embedding-path /embeddings \
  --backend faiss \
  --index-type hnsw \
  --faiss-base-dir ./data \
  --model my-embedding-model \
  --embedding-dim 1024 \
  --vocabulary SNOMED \
  --num-embeddings 500
```

Important:

- `OMOP_DATABASE_URL` must point to the PostgreSQL database that exposes your
  OMOP vocabulary tables.
- `OMOP_EMB_METADATA_SCHEMA` controls where `omop-emb` creates its own
  `model_registry` and backend-specific metadata tables. The default is
  `public`.
- The CLI is intended to work against an existing OMOP database. It should not
  attempt to create the full OMOP schema.
- `--api-key` is optional. Use it only if your embedding service expects bearer-token authentication.
- `--api-base` should be the API base URL, for example `http://localhost:8000/v1`,
  not the full embeddings endpoint path. Use `--embedding-path` if your server
  expects a non-standard endpoint such as `/embed`.
- If you need to change the registered dimensions or other model configuration
  for an existing model name, use `--overwrite-model-registration`. This deletes
  the existing backend storage and model registry entry before re-registering it.
  For FAISS, this also deletes the model's on-disk directory so the vector store
  is rebuilt cleanly.
- The FAISS backend now supports `flat` and `hnsw` index types. `hnsw` is the
  better default for larger retrieval workloads.
- The code queries the OMOP `concept` table by ORM table name, not by a
  hard-coded schema-qualified path such as `vocabulary.concept`.
- PostgreSQL resolves that table through the connection `search_path`. If your
  OMOP vocabulary tables live in a non-default schema such as
  `staging_vocabulary`, your `search_path` must include that schema or the
  connection must otherwise resolve `concept` to the intended table.
- If you pass `--num-embeddings`, that is only a limit. The actual number of
  selected concepts may still be zero if the query resolves to the wrong table,
  the vocabulary filter matches nothing, or the model is already registered as
  embedded in the SQL registry.

Stored embeddings can also be queried after ingestion:

```bash
omop-emb search "type 2 diabetes" \
  --api-base http://localhost:8000/v1 \
  --embedding-path /embeddings \
  --model my-embedding-model \
  --backend faiss \
  --faiss-base-dir ./data \
  --metric-type cosine \
  --k 5
```

FAISS indexes can also be rebuilt explicitly from the stored HDF5 vectors:

```bash
omop-emb rebuild-index \
  --model my-embedding-model \
  --backend faiss \
  --faiss-base-dir ./data \
  --metric-type cosine \
  --metric-type l2
```

## DB Test Environment

The PostgreSQL-backed test suite reads these environment variables from
`tests/conftest.py`:

- `TEST_DB_HOST`: required, PostgreSQL host for the test database server.
- `TEST_DB_PORT`: required, PostgreSQL port.
- `TEST_DATABASE_NAME`: optional, defaults to `test_omop_emb`.
- `TEST_DB_USERNAME`: optional, defaults to `test`.
- `TEST_DB_PASSWORD`: optional, defaults to `test`.
- `TEST_DB_DRIVER`: optional, defaults to `postgresql+psycopg2`.
- `POSTGRES_USER`: optional, defaults to `postgres`; used for creating/dropping the test database.
- `POSTGRES_PASSWORD`: optional, defaults to `postgres`; used for creating/dropping the test database.

The tests create and drop a dedicated database and role, so the admin user must
have permission to create roles and databases.

For non-admin testing against an existing database, set:

- `TEST_DB_USE_EXISTING=1`
- `TEST_DB_SCHEMA=<dedicated_test_schema>`

In that mode, the suite does not create or drop databases or roles. It creates
its tables inside the specified schema and forces the engine `search_path` into
that schema so your existing OMOP tables are not touched.

To run only non-database unit tests:

```bash
pytest -m unit
```

To run FAISS integration tests against an existing PostgreSQL database/schema:

```bash
TEST_DB_USE_EXISTING=1 \
TEST_DB_SCHEMA=omop_emb_test \
pytest -m "faiss and integration"
```

# Project Roadmap

- [x] Interface for postgres storage of vectors
- [x] Interface for FAISS storage of embeddings
- [ ] Extensive unit testing
    - [ ] Backend testing
    - [ ] Corruption and restoration of DB testing
- [ ] Support non-Flat indices for each backend
- [ ] `faiss` GPU support
- [ ] [`pgvectorscale`](https://github.com/timescale/pgvectorscale) support
- [ ] Vector-quantisation for more efficient storage
