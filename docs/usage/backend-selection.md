# Embedding Storage Backends

`omop-emb` selects the storage backend via a `[vector_stores.*]` entry's `backend_type`, configured through [oa-configurator](https://AustralianCancerDataNetwork.github.io/oa-configurator/).

## Supported backends

| Backend | `backend_type` | Requires | Notes |
|---|---|---|---|
| **sqlite-vec** | `sqlitevec` | nothing extra | File or in-memory (`database_name = ":memory:"` on the underlying connection). |
| **pgvector** | `pgvector` | `omop-emb[pgvector]` + PostgreSQL | Scales to large corpora. HNSW indexing and `halfvec` storage. |
| **DuckDB VSS** | `duckdbvss` | `omop-emb[duckdbvss]` | File or in-memory, single process, no server. HNSW indexing via the `vss` extension (`INSTALL vss` needs network access once, then it's cached locally). |

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

Hierarchical Navigable Small World graph. Approximate nearest-neighbour search with sub-linear query time. Supported **only** by `pgvector` and DuckDB VSS.

| Parameter | Default | Effect |
|---|---|---|
| `num_neighbors` | `32` | Graph connectivity (`M`). Higher = better recall, larger index. |
| `ef_construction` | `200` | Build quality. Higher = better recall at build time, slower build. |
| `ef_search` | `200` | Query recall. Higher = better recall at query time, slower query. |
| `max_scan_tuples` | `None` | pgvector only. Rows a filtered query may read from the index. `None` keeps the server default of 20,000. |

```python
from omop_emb.backends.index_config import HNSWIndexConfig
from omop_emb.config import MetricType

HNSWIndexConfig(
    metric_type=MetricType.COSINE,  # locked in at build time for HNSW
    num_neighbors=32,
    ef_construction=200,
    ef_search=200,
)
```

The HNSW workflow is always: **register (FLAT) → ingest → rebuild index**:

```python
# 1. Register with FLAT (always)
backend.register_model(
    model_name=..., provider_type=..., index_config=FlatIndexConfig(), dimensions=768
)

# 2. Ingest data
backend.upsert_embeddings(...)

# 3. Build HNSW index
backend.rebuild_index(model_name=..., index_config=HNSWIndexConfig(metric_type=MetricType.COSINE))
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

    The iterative scan stops after `hnsw.max_scan_tuples` rows (Default: 20,000) so a filter that matches few of the nearest rows can still return fewer than `k` results, or none. Raise the cap per model with `omop-emb maintenance set-ef-search --model <NAME> --max-scan-tuples 100000`, or set `max_scan_tuples` on the `HNSWIndexConfig`. A higher cap makes those queries slower. Filtering by a single domain is the usual case: a rarer domain needs a larger cap.

!!! warning "DuckDB VSS HNSW and filtered queries"
    DuckDB's built-in `vss` extension applies a concept filter after the index returns its candidates, the same way pgvector does, but with no `iterative_scan`-equivalent at all: raising `ef_search` has no effect on a filtered query, confirmed empirically up to `ef_search=20000` on a 300k-row table (always short, never returning `k` rows). `omop-emb` works around this by materializing the filtered rows into a CTE before computing distance, which stops DuckDB's optimizer from using the index and forces an exact scan over just the filtered subset instead. This is always correct, but always pays the exact-scan cost for a filtered query; there is no faster, index-accelerated path today. See the [Performance Guide](performance.md) for measured latency. A community fork, [`hnsw_acorn`](https://duckdb.org/community_extensions/extensions/hnsw_acorn), adds predicate-aware index traversal; not adopted here, since it's a single-maintainer fork distributed outside DuckDB's own extension.

!!! info "Changing HNSW settings on an existing index"
    `ef_search` and `max_scan_tuples` are read at query time, so `omop-emb maintenance set-ef-search` changes them without a rebuild. `ef_construction` and `num_neighbors` shape the graph, so changing them needs `rebuild-index`. Stores indexed before the defaults changed to 200 keep the values stored in their registry until they are rebuilt or updated. The FAISS cache applies the `ef_search` of the config passed at search time, not the registry value.

!!! info "DuckDB space after a rebuild"
    A DuckDB file does not shrink when an index is dropped and rebuilt, so rebuilding with different settings can leave unused space in the file until it is compacted.

!!! warning "HNSW builds in containers"
    A parallel HNSW build can fail with `could not resize shared memory segment ... No space left on device` when the container's `/dev/shm` is small. Either enlarge `/dev/shm` (for example `shm_size` in Docker Compose) or set `max_parallel_maintenance_workers = 0` on the server, which builds serially and takes longer.

## Precision

Two separate options control half precision. Both are pgvector only. sqlite-vec and DuckDB VSS store float32 and reject `float16`, and DuckDB's HNSW index accepts only float32.

| Option | Set | Changed by | Effect |
|---|---|---|---|
| Storage precision (`--storage-precision`, `storage_precision`) | At registration | Re-registering with another value is a conflict | `float16` stores the column as `halfvec`. Halves the table and the index, at some loss of accuracy in the stored vectors. |
| Index precision (`--index-precision`, `index_precision`, on any index config) | When the HNSW index is built | `rebuild-index` | `float16` over a float32 column indexes a `halfvec` cast of it. Halves the index and keeps the full vectors in the table. |

`None` is the default for the index precision and means the index follows the stored column. FLAT has no index, so it rejects an explicit value.

Defaults follow the dimensions. Up to 2,000 dimensions the column is float32. Above that the column defaults to `float16`, because a float32 `vector` index supports at most 2,000 dimensions and `halfvec` supports 4,000. Registering above 4,000 dimensions raises an error.

A wide model can keep float32 storage by choosing `--storage-precision float32` at registration and building its HNSW index with `--index-precision float16`. Without that, an HNSW index on a float32 column above 2,000 dimensions is rejected with a message naming the option.

Models registered before the precision was recorded have none in the registry, so the backend derives it from the dimensions as before. The first writer to open an older registry adds the column. Readers work on an older registry in the meantime. Bundles record the stored precision. An import uses it when the target backend supports it and otherwise registers the model with the backend's default. An import that rebuilds the index also checks the bundle's index config before loading anything. It drops the index precision or `max_scan_tuples` when the target does not support them, with a warning, and raises for anything else it cannot build.

Queries on a half-precision index over a float32 column rank by the half-precision distance, so the returned similarities are those of the cast vectors. Stores registered before the option existed keep the precision they were created with. Recall at half precision has not been measured yet.

```bash
omop-emb embeddings add-embeddings --storage-precision float16
omop-emb maintenance rebuild-index --model <NAME> --index-type hnsw --index-precision float16
```

## Backend capabilities

Each backend declares what it supports as `backend.capabilities`. Registration, `rebuild-index` and `set-ef-search` check it before changing anything and raise an error that names the backend and the option.

| Capability | sqlite-vec | pgvector | DuckDB VSS |
|---|---|---|---|
| Index types | FLAT | FLAT, HNSW | FLAT, HNSW |
| Metrics | L2, cosine, L1 | L2, cosine, L1 | L2, cosine |
| `max_scan_tuples` | no | yes | no |
| Storage precision | float32 | float32, float16 | float32 |
| Index precision (HNSW) | none | float32, float16 | float32 |
| Largest model | no limit | 4,000 dimensions | no limit |
| Largest HNSW index | none | 2,000 (float32), 4,000 (float16) | no limit |

## Metrics

Each backend supports a subset of distance metrics:

| Backend | FLAT | HNSW |
|---|---|---|
| sqlite-vec | L2, Cosine, L1 | none |
| pgvector | L2, Cosine, L1 | L2, Cosine, L1 |
| DuckDB VSS | L2, Cosine | L2, Cosine |
| FAISS sidecar | L2, Cosine | L2, Cosine |

Stored vectors carry no metric. A model carries one only when its backend needs it, and queries must then use exactly that metric.

- **pgvector FLAT** computes the distance from the raw vectors, so queries choose any supported metric. A metric given for a FLAT model is ignored.
- **pgvector HNSW** is locked to the metric it was built with. Change it with `rebuild_index`.
- **sqlite-vec** declares the metric on the table, because its native `MATCH` search uses it. Every model is locked to one metric, cosine unless given `FlatIndexConfig(metric_type=...)`. Querying with another metric raises.
- **DuckDB VSS FLAT** & **DuckDB VSS HNSW**  same as `pgevctor`. 

!!! important "Changing the metric of a sqlite-vec model"
    `rebuild_index` with another metric recreates the table from the stored vectors, so nothing is re-embedded:

    ```bash
    omop-emb maintenance rebuild-index --model <canonical-model-name> --metric-type l2
    ```

    Without `--metric-type` the metric is cosine. `MATCH` returns at most 4096 results per query, so a larger `k` raises.

    A table created by an earlier version has no declared metric and raises `LegacyVectorTableError` when it is first used. The same command without `--metric-type` recreates it with cosine.

    Deleted rows leave space that `VACUUM` alone does not return, because vec0 stores rows in chunks and does not refill the free slots of an earlier one. `maintenance prune-embeddings --apply --compact` recreates the table to return it.
