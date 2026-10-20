# Embedding Storage Backends

`omop-emb` selects the storage backend via a `[vector_stores.*]` entry's `backend_type`, configured through [oa-configurator](https://AustralianCancerDataNetwork.github.io/oa-configurator/).

## Supported backends

| Backend | `backend_type` | Requires | Notes |
|---|---|---|---|
| **sqlite-vec** | `sqlitevec` | nothing extra | File or in-memory (`database_name = ":memory:"` on the underlying connection). |
| **pgvector** | `pgvector` | `omop-emb[pgvector]` + PostgreSQL | Scales to large corpora. HNSW indexing and `halfvec` storage. |

!!! note "FAISS is a sidecar, not a backend"
    FAISS (`omop-emb[faiss-cpu]`) is a read-acceleration layer that sits on top of sqlite-vec or pgvector. It is not a primary backend and has no `backend_type` of its own. See the [CLI reference](cli.md#build-faiss-cache) for how to export and use FAISS indices.

## Selecting a backend

```toml
[vector_stores.vector_store]
backend_type = "pgvector"    # or "sqlitevec"
database     = "emb_db"
```

See [Configuration Reference](configuration.md) for the full field list, both backends' setup, and the [Performance Guide](performance.md) for which backend to pick at which size and what to tune.

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
    HNSW is a SQL `CREATE INDEX USING hnsw` object built by `rebuild_index`. Without it, pgvector falls back to a sequential scan automatically. `ef_search` is read from the registry and applied to each query's transaction. A rebuild leaves exactly the index it describes: switching metric or reverting to FLAT drops the previous HNSW index. Afterwards `rebuild_index` runs `VACUUM (ANALYZE)` on the table by default, which refreshes the statistics the planner uses to choose between the index and an exact scan. On a large table this can take a while; pass `vacuum=False` (CLI: `--no-vacuum`) to skip it.

!!! info "pgvector HNSW and filtered queries"
    HNSW applies a concept filter after the index returns its candidates, so a selective filter can leave fewer than `k` rows. With pgvector 0.8.0 or newer, `omop-emb` sets `hnsw.iterative_scan` for each query, which keeps scanning the index until `k` rows pass the filter. The scan runs in `relaxed_order` and the rows are sorted by distance afterwards, because `strict_order` skips every row closer than the last one it returned and so loses neighbours found in later batches. Older pgvector versions have no iterative scan: `ef_search` is raised to at least `k` (at most 1000, the pgvector limit) and a filtered query logs a one-time warning, since its results can still be incomplete. Upgrade the extension to fix this. For very selective filters at a low `ef_search`, the planner may choose an exact scan over the index.

!!! warning "HNSW builds in containers"
    A parallel HNSW build can fail with `could not resize shared memory segment ... No space left on device` when the container's `/dev/shm` is small. Either enlarge `/dev/shm` (for example `shm_size` in Docker Compose) or set `max_parallel_maintenance_workers = 0` on the server, which builds serially and takes longer.

!!! warning "pgvector dimension limit"
    The pgvector `vector` column type supports **at most 2,000 dimensions**. Models with more than 2,000 dimensions automatically use the `halfvec` column type (up to 4,000 dimensions). Registering above 4,000 dimensions raises a `ValueError`.

## Metrics

Each backend supports a subset of distance metrics:

| Backend | FLAT | HNSW |
|---|---|---|
| sqlite-vec | L2, Cosine, L1 | none |
| pgvector | L2, Cosine, L1 | L2, Cosine, L1 |
| FAISS sidecar | L2, Cosine | L2, Cosine |

Stored vectors carry no metric. A model carries one only when its backend needs it, and queries must then use exactly that metric.

- **pgvector FLAT** computes the distance from the raw vectors, so queries choose any supported metric. A metric given for a FLAT model is ignored.
- **pgvector HNSW** is locked to the metric it was built with. Change it with `rebuild_index`.
- **sqlite-vec** declares the metric on the table, because its native `MATCH` search uses it. Every model is locked to one metric, cosine unless given `FlatIndexConfig(metric_type=...)`. Querying with another metric raises.

!!! important "Changing the metric of a sqlite-vec model"
    `rebuild_index` with another metric recreates the table from the stored vectors, so nothing is re-embedded:

    ```bash
    omop-emb maintenance rebuild-index --model <canonical-model-name> --metric-type l2
    ```

    Without `--metric-type` the metric is cosine. `MATCH` returns at most 4096 results per query, so a larger `k` raises.

    A table created by an earlier version has no declared metric and raises `LegacyVectorTableError` when it is first used. The same command without `--metric-type` recreates it with cosine.

    Deleted rows leave space that `VACUUM` alone does not return, because vec0 stores rows in chunks and does not refill the free slots of an earlier one. `maintenance prune-embeddings --apply --compact` recreates the table to return it.
