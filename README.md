# omop-emb

Vector embedding layer for OMOP CDM concepts.

`omop-emb` generates, stores, and retrieves embeddings for OMOP concepts. It
works out of the box with **sqlite-vec** (no external database required) and
scales to **PostgreSQL/pgvector** for larger deployments.

## Installation

```bash
pip install omop-emb                    # sqlite-vec backend (default, no extras needed)
pip install "omop-emb[pgvector]"        # adds PostgreSQL/pgvector support
pip install "omop-emb[faiss]"           # adds FAISS sidecar support
pip install "omop-emb[pgvector,faiss]"  # everything
```

## Quick start

**sqlite-vec (no external service):**

```bash
export OMOP_EMB_BACKEND=sqlitevec
export OMOP_EMB_SQLITE_PATH=/data/omop_emb.db

omop-emb add-embeddings --api-base http://localhost:11434/v1 --api-key ollama \
    --model nomic-embed-text:v1.5
```

**pgvector:**

```bash
export OMOP_EMB_BACKEND=pgvector
export OMOP_EMB_DB_HOST=localhost
export OMOP_EMB_DB_PORT=5432
export OMOP_EMB_DB_USER=omop_emb
export OMOP_EMB_DB_PASSWORD=omop_emb
export OMOP_EMB_DB_NAME=omop_emb

omop-emb add-embeddings --api-base http://localhost:11434/v1 --api-key ollama \
    --model nomic-embed-text:v1.5
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OMOP_EMB_BACKEND` | `sqlitevec` | Backend: `sqlitevec` or `pgvector`. |
| `OMOP_EMB_SQLITE_PATH` | — | sqlite-vec database file path (or `:memory:`). |
| `OMOP_EMB_DB_HOST` | — | pgvector: PostgreSQL host. |
| `OMOP_EMB_DB_PORT` | `5432` | pgvector: PostgreSQL port. |
| `OMOP_EMB_DB_USER` | — | pgvector: database user. |
| `OMOP_EMB_DB_PASSWORD` | — | pgvector: database password. |
| `OMOP_EMB_DB_NAME` | — | pgvector: database name. |
| `OMOP_EMB_DB_URL` | — | pgvector: full SQLAlchemy URL (overrides individual vars). |
| `OMOP_CDM_DB_URL` | — | OMOP CDM connection (required for ingestion commands only). |

See the [Configuration Reference](https://AustralianCancerDataNetwork.github.io/omop-emb/usage/configuration/)
for the complete list including asymmetric embedding prefixes and driver overrides.

## Documentation

Full documentation: <https://AustralianCancerDataNetwork.github.io/omop-emb>

- [Installation & backend setup](https://AustralianCancerDataNetwork.github.io/omop-emb/usage/installation/)
- [Configuration reference](https://AustralianCancerDataNetwork.github.io/omop-emb/usage/configuration/)
- [Backend selection & index types](https://AustralianCancerDataNetwork.github.io/omop-emb/usage/backend-selection/)
- [CLI reference](https://AustralianCancerDataNetwork.github.io/omop-emb/usage/cli/)

## Roadmap

- [x] sqlite-vec backend (default, zero-config)
- [x] pgvector backend (PostgreSQL)
- [x] FAISS sidecar (approximate nearest-neighbour acceleration)
- [x] HNSW index support for pgvector and FAISS
- [x] Extensive backend and registry testing
- [ ] Import/export of pre-computed embeddings
- [ ] `faiss` GPU support
- [ ] [`pgvectorscale`](https://github.com/timescale/pgvectorscale) support
- [ ] Vector quantisation for more efficient storage
