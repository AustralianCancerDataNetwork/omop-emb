# Embedding Storage Backends

`omop-emb` selects the storage backend via a `[vector_stores.*]` entry's `backend_type`, configured through [oa-configurator](https://AustralianCancerDataNetwork.github.io/oa-configurator/).

## Supported backends

| Backend | `backend_type` | Requires | Notes |
|---|---|---|---|
| **sqlite-vec** | `sqlitevec` | nothing extra | File or in-memory (`database_name = ":memory:"` on the underlying connection). |
| **pgvector** | `pgvector` | `omop-emb[pgvector]` + PostgreSQL | Scales to large corpora. HNSW indexing and `halfvec` storage. |

!!! note "FAISS is a sidecar, not a backend"
    FAISS (`omop-emb[faiss-cpu]`) is a read-acceleration layer that sits on top of sqlite-vec or pgvector. It is not a primary backend and has no `backend_type` of its own. See the [CLI reference](cli.md#faiss-sidecar) for how to export and use FAISS indices.

## Selecting a backend

```toml
[vector_stores.vector_store]
backend_type = "pgvector"    # or "sqlitevec"
database     = "emb_db"
```

See [Configuration Reference](configuration.md) for the full field list, both backends' setup.

## Index types

Each primary backend supports index types controlled by an `IndexConfig` object.

!!! important "Registration always uses FLAT"
    Models must always be registered with a `FlatIndexConfig`. After data has been ingested, call `rebuild_index` (or the `rebuild-index` CLI command) to switch to an HNSW index. Registering directly with `HNSWIndexConfig` raises a `ValueError`.

### FLAT

Sequential scan: no index structure is built. Every query compares the query vector against all stored embeddings. Always correct, requires no build step.

```python
from omop_emb.backends.index_config import FlatIndexConfig

FlatIndexConfig()  # no parameters
```

Use FLAT when the corpus is small (tens of thousands of concepts) or when exact results are required.

### HNSW

Hierarchical Navigable Small World graph. Approximate nearest-neighbour search with sub-linear query time. Supported by pgvector only; **not** supported by sqlite-vec.

| Parameter | Default | Effect |
|---|---|---|
| `num_neighbors` | `32` | Graph connectivity (`M`). Higher = better recall, larger index. |
| `ef_construction` | `64` | Build quality. Higher = better recall at build time, slower build. |
| `ef_search` | `16` | Query recall. Higher = better recall at query time, slower query. |

```python
from omop_emb.backends.index_config import HNSWIndexConfig
from omop_emb.config import MetricType

HNSWIndexConfig(
    metric_type=MetricType.COSINE,  # locked in at build time for HNSW
    num_neighbors=32,
    ef_construction=64,
    ef_search=16,
)
```

The HNSW workflow is always: **register (FLAT) → ingest → rebuild index**:

```python
# 1. Register with FLAT (always)
backend.register_model(model_name=..., provider_type=...,
                        index_config=FlatIndexConfig(), dimensions=768)

# 2. Ingest data
backend.upsert_embeddings(...)

# 3. Build HNSW index
backend.rebuild_index(model_name=...,
                       index_config=HNSWIndexConfig(metric_type=MetricType.COSINE))
```

Or via the CLI:

```bash
omop-emb embeddings add-embeddings
omop-emb maintenance rebuild-index --model-name embedding-model --index-type hnsw --metric-type cosine
```

!!! info "pgvector HNSW"
    HNSW is a SQL `CREATE INDEX USING hnsw` object built by `rebuild_index`. Without it, pgvector falls back to a sequential scan automatically. `ef_search` is applied per session at query time.

!!! warning "pgvector dimension limit"
    The pgvector `vector` column type supports **at most 2,000 dimensions**. Models with more than 2,000 dimensions automatically use the `halfvec` column type (up to 4,000 dimensions). Registering above 4,000 dimensions raises a `ValueError`.

## Metrics

Each backend supports a subset of distance metrics:

| Backend | FLAT | HNSW |
|---|---|---|
| sqlite-vec | L2, Cosine, L1 | none |
| pgvector | L2, Cosine, L1 | L2, Cosine, L1 |
| FAISS sidecar | L2, Cosine | L2, Cosine |

For FLAT models, the metric is supplied by the caller at query time. For HNSW models, the metric is locked in at `rebuild_index` time and must be supplied consistently at every query.
