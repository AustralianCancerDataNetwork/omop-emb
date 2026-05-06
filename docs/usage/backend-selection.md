# Embedding Storage Backends

`omop-emb` selects the storage backend at runtime via the `OMOP_EMB_BACKEND`
environment variable (default: `sqlitevec`).

## Supported backends

| Backend | Value | Requires | Notes |
|---|---|---|---|
| **sqlite-vec** | `sqlitevec` | nothing extra | Default. File or in-memory. |
| **pgvector** | `pgvector` | `omop-emb[pgvector]` + PostgreSQL | Scales to large corpora. |
| **FAISS** | `faiss` | `omop-emb[faiss]` | Read-acceleration sidecar only. |

## Runtime selection

```bash
export OMOP_EMB_BACKEND=sqlitevec   # default — set OMOP_EMB_SQLITE_PATH
export OMOP_EMB_BACKEND=pgvector    # set OMOP_EMB_DB_* vars or OMOP_EMB_DB_URL
```

See [Installation](installation.md) for the full list of connection variables
for each backend.

## Index types

Each primary backend supports index types controlled by an `IndexConfig` object.

!!! important "Registration always uses FLAT"
    Models must always be registered with a `FlatIndexConfig`. After data has
    been ingested, call `rebuild_index` (or the `rebuild-index` CLI command) to
    switch to an HNSW index. Registering directly with `HNSWIndexConfig` raises
    a `ValueError`.

### FLAT

Sequential scan — no index structure is built. Every query compares the query
vector against all stored embeddings. Always correct, requires no build step.

```python
from omop_emb.backends.index_config import FlatIndexConfig

FlatIndexConfig()  # no parameters
```

Use FLAT when the corpus is small (tens of thousands of concepts) or when exact
results are required.

### HNSW

Hierarchical Navigable Small World graph. Approximate nearest-neighbour search
with sub-linear query time. Supported by pgvector and FAISS; **not** supported
by sqlite-vec.

| Parameter | Default | Effect |
|---|---|---|
| `num_neighbors` | `16` | Graph connectivity (`M`). Higher = better recall, larger index. |
| `ef_construction` | `64` | Build quality. Higher = better recall at build time, slower build. |
| `ef_search` | `16` | Query recall. Higher = better recall at query time, slower query. |

```python
from omop_emb.backends.index_config import HNSWIndexConfig

HNSWIndexConfig(
    metric_type=MetricType.COSINE,  # locked in at build time for HNSW
    num_neighbors=16,
    ef_construction=64,
    ef_search=16,
)
```

The HNSW workflow is always: **register (FLAT) → ingest → rebuild index**:

```python
# 1. Register with FLAT — always
backend.register_model(model_name=..., provider_type=...,
                        index_config=FlatIndexConfig(), dimensions=768)

# 2. Ingest data
backend.upsert_embeddings(...)

# 3. Build HNSW index
backend.rebuild_index(model_name=..., provider_type=...,
                       index_config=HNSWIndexConfig(metric_type=MetricType.COSINE))
```

Or via the CLI:

```bash
omop-emb add-embeddings --api-base ... --api-key ... --model nomic-embed-text
omop-emb rebuild-index --model nomic-embed-text --index-type hnsw --metric-type cosine
```

!!! info "Backend differences"
    - **pgvector**: HNSW is a SQL `CREATE INDEX USING hnsw` object built by
      `rebuild_index`. Without it, pgvector falls back to a sequential scan
      automatically. `ef_search` is applied per session at query time.
    - **FAISS**: HNSW index is stored as a `.faiss` file. It is built from the
      stored data the first time it is needed or explicitly via `rebuild_index`.
    - **sqlite-vec**: HNSW is not supported. Only FLAT (exact scan) is available.

!!! warning "pgvector dimension limit"
    The pgvector `vector` column type supports **at most 2,000 dimensions**.
    Models with more than 2,000 dimensions automatically use the `halfvec`
    column type (up to 4,000 dimensions). Registering above 4,000 dimensions
    raises a `ValueError`.

## Metrics

Each backend supports a subset of distance metrics:

| Backend | FLAT | HNSW |
|---|---|---|
| sqlite-vec | L2, Cosine, L1 | — |
| pgvector | L2, Cosine, L1 | L2, Cosine, L1 |
| FAISS | L2, Cosine | L2, Cosine |

For FLAT models, the metric is supplied by the caller at query time. For HNSW
models, the metric is locked in at `rebuild_index` time and must be supplied
consistently at every query.
