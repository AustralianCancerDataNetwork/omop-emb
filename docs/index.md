# OMOP Embeddings

`omop-emb` generates and retrieves vector embeddings for OMOP CDM concepts. It
works standalone out of the box (sqlite-vec, no external database required) and
optionally scales to PostgreSQL via the pgvector extension.

The package supports:

- dynamic embedding model registration — multiple models per backend, tracked in the embedding database
- embedding and lookup for OMOP concepts across configurable storage backends
- three storage backends:
  - **sqlite-vec** (default): zero-config, file-based or in-memory — no external service required
  - **pgvector**: PostgreSQL with the pgvector extension (FLAT sequential scan or HNSW SQL index)
  - **FAISS**: approximate nearest-neighbour sidecar on top of any primary backend
- CLI scripts to ingest OMOP CDM concepts and manage registered models

## Installation

Install the backend you want to use:

```bash
pip install omop-emb                  # sqlite-vec only (default backend)
pip install "omop-emb[pgvector]"      # adds PostgreSQL/pgvector support
pip install "omop-emb[faiss]"         # adds FAISS sidecar support
pip install "omop-emb[pgvector,faiss]"  # everything
```

## Environment Variables { data-toc-label="Environment Variables" }

### Backend selector

| Variable | Default | Description |
|---|---|---|
| `OMOP_EMB_BACKEND` | `sqlitevec` | Backend to use: `sqlitevec`, `pgvector`, or `faiss`. |

### sqlite-vec connection

| Variable | Description |
|---|---|
| `OMOP_EMB_SQLITE_PATH` | Path to the sqlite-vec database file. Use `:memory:` for an in-memory database. |

### pgvector connection (individual components)

| Variable | Default | Description |
|---|---|---|
| `OMOP_EMB_DB_HOST` | — | PostgreSQL host. |
| `OMOP_EMB_DB_PORT` | `5432` | PostgreSQL port. |
| `OMOP_EMB_DB_USER` | — | PostgreSQL user. |
| `OMOP_EMB_DB_PASSWORD` | — | PostgreSQL password. |
| `OMOP_EMB_DB_NAME` | — | PostgreSQL database name. |
| `OMOP_EMB_DB_DRIVER` | `postgresql+psycopg` | SQLAlchemy driver string. Override to use e.g. `psycopg2`. |
| `OMOP_EMB_DB_URL` | — | Full SQLAlchemy connection URL. Overrides all individual components above when set. |

### Embedding API (CLI concept ingestion)

| Variable | Description |
|---|---|
| `OMOP_CDM_DB_URL` | SQLAlchemy URL for the OMOP CDM database. Required only for concept ingestion commands. |
| `OMOP_EMB_DOCUMENT_EMBEDDING_PREFIX` | Task prefix prepended to concept texts at index time. |
| `OMOP_EMB_QUERY_EMBEDDING_PREFIX` | Task prefix prepended to search queries at query time. |

The prefix variables are optional and default to `""`. They are only needed for
asymmetric embedding models (e.g. nomic-embed-text, E5, BGE) that require
different task prefixes for indexing versus querying.

## Documentation overview

- [Installation](usage/installation.md)
- [Embedding storage backends](usage/backend-selection.md)
- [CLI Reference](usage/cli.md)
- [Asymmetric Embeddings](usage/asymmetric-embeddings.md)
