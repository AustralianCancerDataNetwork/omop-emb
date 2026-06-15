# omop-emb

Vector embedding layer for OMOP CDM concepts.

`omop-emb` generates, stores, and retrieves embeddings for OMOP concepts. It
works out of the box with **sqlite-vec** (no external database required) and
scales to **PostgreSQL/pgvector** for larger deployments. The database is the
source of truth — FAISS is an optional read-acceleration sidecar, not a primary
store.

## Installation

```bash
pip install omop-emb                         # sqlite-vec backend (default, no extras needed)
pip install "omop-emb[pgvector]"             # adds PostgreSQL/pgvector support
pip install "omop-emb[faiss-cpu]"            # adds FAISS sidecar support
pip install "omop-emb[pgvector,faiss-cpu]"   # everything
```

## Quick start

**Ingest concepts (sqlite-vec, no external service):**

```bash
export OMOP_EMB_BACKEND=sqlitevec
export OMOP_EMB_SQLITE_PATH=/data/omop_emb.db
export OMOP_CDM_DB_URL=postgresql+psycopg://user:pass@host:5432/omop_cdm

omop-emb embeddings add-embeddings --api-base http://localhost:11434/v1 --api-key ollama \
    --provider ollama --model nomic-embed-text:v1.5
```

**Search:**

```bash
omop-emb embeddings search --api-base http://localhost:11434/v1 --api-key ollama \
    --provider ollama --model nomic-embed-text:v1.5 \
    --query "hypertension" --query "type 2 diabetes" \
    --standard-only --domain Condition --k 5
```

**pgvector with HNSW index:**

```bash
export OMOP_EMB_BACKEND=pgvector
export OMOP_EMB_DB_HOST=localhost
export OMOP_EMB_DB_USER=omop_emb
export OMOP_EMB_DB_PASSWORD=omop_emb
export OMOP_EMB_DB_NAME=omop_emb

omop-emb embeddings add-embeddings --api-base http://localhost:11434/v1 --api-key ollama \
    --provider ollama --model nomic-embed-text:v1.5
omop-emb maintenance rebuild-index --model nomic-embed-text:v1.5 --index-type hnsw --metric-type cosine
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
| `OMOP_EMB_FAISS_CACHE_DIR` | — | Default FAISS cache directory (alternative to `--faiss-cache-dir`). |

See the [Configuration Reference](https://AustralianCancerDataNetwork.github.io/omop-emb/usage/configuration/)
for the complete list including asymmetric embedding prefixes and driver overrides.

## Documentation

Full documentation: <https://AustralianCancerDataNetwork.github.io/omop-emb>

- [Installation & backend setup](https://AustralianCancerDataNetwork.github.io/omop-emb/usage/installation/)
- [Configuration reference](https://AustralianCancerDataNetwork.github.io/omop-emb/usage/configuration/)
- [Backend selection & index types](https://AustralianCancerDataNetwork.github.io/omop-emb/usage/backend-selection/)
- [CLI reference](https://AustralianCancerDataNetwork.github.io/omop-emb/usage/cli/)
- [Interface guide](https://AustralianCancerDataNetwork.github.io/omop-emb/usage/interface-guide/)

## Roadmap

- [x] sqlite-vec backend (default, zero-config)
- [x] pgvector backend (PostgreSQL)
- [x] HNSW index support for pgvector
- [x] FAISS sidecar (approximate nearest-neighbour read acceleration)
- [x] FAISS export / import CLI (`export-faiss-cache`, `import-faiss-cache`)
- [x] In-DB concept filtering (domain, vocabulary, standard status, active status)
- [x] Transparent FAISS fast path in `EmbeddingReaderInterface`
- [x] Extensive backend and registry testing
- [ ] FAISS GPU support
- [ ] [`pgvectorscale`](https://github.com/timescale/pgvectorscale) support
- [ ] Vector quantisation for more efficient storage

---

## Configuration via oa-configurator

The database connection can also be configured via
[oa-configurator](https://github.com/AustralianCancerDataNetwork/oa-configurator),
which stores settings in `~/.config/omop/config.toml` and eliminates the need
for environment variables at runtime:

```bash
omop-config init
omop-config configure omop_alchemy   # CDM database (required for ingestion)
omop-config configure omop_emb       # embedding database
```

See [oa-configurator Setup](docs/getting-started/configuration.md) for details.

---

## Docker Compose

The included `docker-compose.yaml` provides both a CDM PostgreSQL database and a
pgvector embedding database, plus a Python container with all optional backends
pre-installed (`[pgvector,faiss-cpu]`). Default credentials work out of the box:

```bash
docker compose up
```

Include Ollama by adding the `standalone` profile:

```bash
docker compose --profile standalone up
```

The `python-emb` service runs `omop-config configure` at startup. To override
credentials:

```bash
cp .env.example .env
docker compose up
```
