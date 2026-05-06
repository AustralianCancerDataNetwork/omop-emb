# Configuration Reference

All configuration is done via environment variables. `omop-emb` loads a `.env`
file from the working directory automatically when using the CLI. For library
usage, load it yourself with `python-dotenv` or set the variables in your
environment before importing.

---

## Backend selector

| Variable | Default | Values |
|---|---|---|
| `OMOP_EMB_BACKEND` | `sqlitevec` | `sqlitevec`, `pgvector` |

Controls which storage backend the CLI and `resolve_backend()` use.

---

## sqlite-vec connection

Required when `OMOP_EMB_BACKEND=sqlitevec` (or when the backend is left at its
default).

| Variable | Required | Description |
|---|---|---|
| `OMOP_EMB_SQLITE_PATH` | yes | Path to the sqlite-vec database file. |

Use the special value `:memory:` for a transient in-memory database (useful in
tests and short-lived scripts):

```bash
OMOP_EMB_SQLITE_PATH=/data/omop_emb.db      # persistent file
OMOP_EMB_SQLITE_PATH=:memory:               # in-memory (lost on process exit)
```

---

## pgvector connection

Required when `OMOP_EMB_BACKEND=pgvector`. You can either supply a full URL or
individual components — the full URL takes precedence when both are set.

### Individual components (recommended)

These match the variables used in the reference `docker-compose.yaml` and can
be committed safely to version-controlled `.env` files (without the password).

| Variable | Required | Default | Description |
|---|---|---|---|
| `OMOP_EMB_DB_HOST` | yes | — | PostgreSQL server hostname or IP. |
| `OMOP_EMB_DB_PORT` | no | `5432` | PostgreSQL server port. |
| `OMOP_EMB_DB_USER` | yes | — | Database user. |
| `OMOP_EMB_DB_PASSWORD` | yes | — | Database password. |
| `OMOP_EMB_DB_NAME` | yes | — | Database name. |
| `OMOP_EMB_DB_CONN` | no | `postgresql+psycopg` | SQLAlchemy driver string. |

**Example `.env`:**

```bash
OMOP_EMB_BACKEND=pgvector
OMOP_EMB_DB_HOST=omop-emb-db
OMOP_EMB_DB_PORT=5432
OMOP_EMB_DB_USER=omop_emb
OMOP_EMB_DB_PASSWORD=omop_emb
OMOP_EMB_DB_NAME=omop_emb
```

### Full URL override

| Variable | Required | Description |
|---|---|---|
| `OMOP_EMB_DB_URL` | no | Complete SQLAlchemy connection URL. Overrides all individual components above. |

```bash
OMOP_EMB_DB_URL=postgresql+psycopg://omop_emb:omop_emb@localhost:5432/omop_emb
```

Use this when the connection string is managed externally (e.g. injected by a
secrets manager or a container orchestrator) and the individual variables are
not available.

### URL composition

When `OMOP_EMB_DB_URL` is not set, `build_engine_string` assembles the
SQLAlchemy URL from the individual components at runtime:

```
{OMOP_EMB_DB_CONN}://{OMOP_EMB_DB_USER}:{OMOP_EMB_DB_PASSWORD}@{OMOP_EMB_DB_HOST}:{OMOP_EMB_DB_PORT}/{OMOP_EMB_DB_NAME}
```

With the defaults above this produces:

```
postgresql+psycopg://omop_emb:omop_emb@omop-emb-db:5432/omop_emb
```

### Driver string

The default driver is `postgresql+psycopg` (psycopg3). Override
`OMOP_EMB_DB_CONN` if you need a different driver:

| Driver string | Package | Notes |
|---|---|---|
| `postgresql+psycopg` | `psycopg[binary]>=3.1` | Default. Included in `omop-emb[pgvector]`. |
| `postgresql+psycopg2` | `psycopg2-binary` | Legacy. Install separately if needed. |
| `postgresql+asyncpg` | `asyncpg` | Async driver. Requires async SQLAlchemy setup. |

---

## Embedding model configuration

These are optional and apply to both backends.

| Variable | Default | Description |
|---|---|---|
| `OMOP_EMB_DOCUMENT_EMBEDDING_PREFIX` | `""` | Task prefix prepended to concept texts at index time. |
| `OMOP_EMB_QUERY_EMBEDDING_PREFIX` | `""` | Task prefix prepended to search queries at query time. |
| `OMOP_EMB_EMBEDDING_DIM` | — | Embedding dimensionality hint (rarely needed; usually auto-discovered). |

The prefix variables are only required for asymmetric embedding models that use
different task instructions for indexing versus querying:

| Model | Document prefix | Query prefix |
|---|---|---|
| `nomic-embed-text` | `search_document: ` | `search_query: ` |
| E5 family | `passage: ` | `query: ` |
| BGE family | `Represent this sentence for searching relevant passages: ` | `query: ` |

Symmetric models (e.g. `text-embedding-3-small`) do not need prefixes — leave
both variables unset or empty.

See [Asymmetric Embeddings](asymmetric-embeddings.md) for details.

---

## OMOP CDM access

Required only for the concept ingestion CLI commands (`add-embeddings`,
`add-embeddings-with-index`). Not needed for `list-models`, `rebuild-index`,
`delete-model`, or library usage.

| Variable | Required | Description |
|---|---|---|
| `OMOP_CDM_DB_URL` | for ingestion | SQLAlchemy URL for the OMOP CDM database (any dialect). |

```bash
OMOP_CDM_DB_URL=postgresql+psycopg://user:pass@localhost:5432/omop_cdm
```

---

## Complete example

A typical `.env` for local development with pgvector:

```bash
# Backend
OMOP_EMB_BACKEND=pgvector

# pgvector connection
OMOP_EMB_DB_HOST=localhost
OMOP_EMB_DB_PORT=5433       # mapped port in docker-compose
OMOP_EMB_DB_USER=omop_emb
OMOP_EMB_DB_PASSWORD=omop_emb
OMOP_EMB_DB_NAME=omop_emb

# CDM (only needed for ingestion commands)
OMOP_CDM_DB_URL=postgresql+psycopg://user:pass@localhost:5432/omop_cdm

# Asymmetric model prefixes (only needed for nomic-embed-text etc.)
OMOP_EMB_DOCUMENT_EMBEDDING_PREFIX=search_document:
OMOP_EMB_QUERY_EMBEDDING_PREFIX=search_query:
```

A typical `.env` for sqlite-vec (zero-config):

```bash
OMOP_EMB_BACKEND=sqlitevec
OMOP_EMB_SQLITE_PATH=/data/omop_emb.db

# CDM (only needed for ingestion commands)
OMOP_CDM_DB_URL=postgresql+psycopg://user:pass@localhost:5432/omop_cdm
```
