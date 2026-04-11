# Embedding Generation CLI

This tool generates vector embeddings for OMOP CDM concepts and stores them in the configured embedding backend.

The CLI targets concepts that do not yet have embeddings for the selected model
and backend, processes them in batches, and stores the results in either the
`pgvector` or `faiss` backend.

!!! note "Supported Models"

    The CLI works with the underlying `omop-llm` client. If your endpoint is
    OpenAI-compatible and the client cannot infer embedding dimensions, pass
    `--embedding-dim` or set `OMOP_EMB_EMBEDDING_DIM`.
---

## Prerequisites

- **Installation**: install the PostgreSQL backend dependencies:

  ```bash
  pip install "omop-emb[pgvector]"
  pip install "omop-emb[faiss]"
  ```

- **Database**: Postgres implementation of OMOP CDM. See [`omop-graph` documentation](reference-missing) for information how to setup.
- **Existing schema**: the CLI expects an existing OMOP database. It only
  creates its own embedding registry/storage metadata and does not need to
  bootstrap the full OMOP schema.
- **Environment**: `OMOP_DATABASE_URL` must be exported or existing in the .env file  (e.g., `postgresql://user:pass@localhost:5432/omop`).
- **Schema resolution**: OMOP vocabulary tables are resolved through the
  PostgreSQL connection `search_path`. If your OMOP tables live in a schema such
  as `staging_vocabulary`, ensure the connection resolves `concept` to that
  schema.
- **Connectivity**: Access to an embedding endpoint that accepts batched text
  input and returns numeric embedding vectors. OpenAI-compatible APIs and
  custom shim services are supported via `--api-base` plus `--embedding-path`.

---

## `add-embeddings`

### Usage
```bash
omop-emb add-embeddings --api-base <URL> [OPTIONS]
```
where `[OPTIONS]` are optional arguments that can be specified as described below.


### Command Options

### Command Options

| Option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`--api-base`** | | `String` | **Required** | Base URL for the embedding API service, e.g. `http://localhost:8000/v1`. |
| **`--embedding-path`** | | `String` | `/embeddings` or `OMOP_EMB_EMBEDDING_PATH` | Relative path for the embedding endpoint, e.g. `/embeddings` or `/embed`. |
| **`--api-key`** | | `String` | `OMOP_EMB_API_KEY` or none | Optional API key for the embedding API provider. |
| **`--index-type`** | | `IndexType` | `FLAT` | The storage index for the embeddings for retrieval. FAISS supports `FLAT` and `HNSW`. |
| **`--batch-size`** | `-b` | `Integer` | `100` | Number of concepts to process in each chunk. |
| **`--model`** | `-m` | `String` | `OMOP_EMB_MODEL` or `text-embedding-3-small` | Name of the embedding model to use for generating vectors. |
| **`--embedding-dim`** | | `Integer` | `OMOP_EMB_EMBEDDING_DIM` or auto-detect | Explicit embedding dimension override for models whose dimensions cannot be inferred automatically. |
| **`--overwrite-model-registration`** | | `Boolean` | `False` | Force a clean rebuild for this model name by deleting backend-owned storage and SQL registration before re-registering it. |
| **`--backend`** | | `Literal['pgvector', 'faiss']` | `OMOP_EMB_BACKEND` | Embedding backend to use. Requires the respective backend extra to be installed. |
| **`--faiss-base-dir`** | | `String` | `None` | Optional base directory for FAISS backend storage. |
| **`--hnsw-num-neighbors`** | | `Integer` | `32` | FAISS HNSW `M` parameter when `--index-type hnsw`. |
| **`--hnsw-ef-search`** | | `Integer` | `64` | FAISS HNSW `efSearch` parameter when `--index-type hnsw`. |
| **`--hnsw-ef-construction`** | | `Integer` | `200` | FAISS HNSW `efConstruction` parameter when `--index-type hnsw`. |
| **`--standard-only`** | | `Boolean` | `False` | If set, only generate embeddings for OMOP standard concepts (`standard_concept = 'S'`). |
| **`--vocabulary`** | | `List[String]` | `None` | Filter to embed concepts only from specific OMOP vocabularies. |
| **`--num-embeddings`** | `-n` | `Integer` | `None` | Limit the number of concepts processed (useful for testing). |

### Environment Variables

- `OMOP_DATABASE_URL`: database connection string for the OMOP store.
  The connection should resolve the OMOP `concept` table via PostgreSQL
  `search_path` if your vocabulary tables are not in the default schema.
- `OMOP_EMB_METADATA_SCHEMA`: schema for `omop-emb`'s own `model_registry` and
  backend-specific metadata tables. Defaults to `public`.
- `OMOP_EMB_API_KEY`: optional API key if your embedding service requires it.
- `OMOP_EMB_EMBEDDING_PATH`: embedding endpoint path if `--embedding-path` is omitted.
- `OMOP_EMB_MODEL`: default model name if `--model` is omitted.
- `OMOP_EMB_EMBEDDING_DIM`: explicit embedding dimension override if `--embedding-dim` is omitted.
- `OMOP_EMB_BACKEND`: default embedding backend if `--backend` is omitted.
- `OMOP_EMB_FAISS_INDEX_DIR`: optional FAISS storage base dir used when the backend is `faiss`.

### Notes

- `--api-base` should be the base service URL such as `http://localhost:8000/v1`,
  not the full embeddings endpoint. Use `--embedding-path` for endpoint shapes
  such as `/embeddings` or `/embed`.
- `--api-key` is optional. If omitted, no bearer token is sent.
- `--num-embeddings` is only an upper bound. The actual selected count may be
  lower, including zero.
- `OMOP_EMB_METADATA_SCHEMA` is separate from the OMOP vocabulary schema. It
  controls where `omop-emb` stores `model_registry` and backend-owned tables,
  while `concept` is still resolved through the database connection's
  `search_path`.
- If an existing model name is already registered with different dimensions or
  other incompatible configuration, use `--overwrite-model-registration` to
  delete the old backend storage and re-register it.
- For FAISS, `--overwrite-model-registration` also deletes the model's on-disk
  directory so stale HDF5/index files cannot survive into the rebuild.
- If stale FAISS artifacts exist on disk without a matching SQL registration,
  the CLI now fails early and tells you to rerun with
  `--overwrite-model-registration`.
- For larger FAISS stores, prefer `--index-type hnsw` over `flat`.

### FAISS Storage Layout

When `--backend faiss` is used, `add-embeddings` writes the raw vectors and
concept IDs into an HDF5 file first. That HDF5 file is the durable source of
truth for the stored embeddings.

FAISS search indexes are separate files derived from the HDF5 data and they are
metric-specific. For example, a model directory may contain:

- `embeddings.h5`: raw vectors and concept IDs
- `index_flat/flat_cosine_index.faiss`: cosine search index
- `index_flat/flat_l2_index.faiss`: L2 search index

This means:

- creating embeddings does not by itself guarantee that every FAISS metric
  index already exists;
- `search` defaults to `--metric-type cosine`;
- if the required FAISS index file for the chosen metric is missing, it may be
  built lazily from `embeddings.h5` on first search, which can take a long time
  for large stores;
- for large datasets, prefer running `rebuild-index` explicitly for the metric
  or metrics you plan to query.
- when `--index-type hnsw` is used, the chosen HNSW settings are stored in the
  model registry metadata and reused for later rebuilds and searches.

## `search`

### Usage
```bash
omop-emb search "type 2 diabetes" --api-base <URL> [OPTIONS]
```

This command embeds the query text and searches the stored embeddings for the
selected model and backend.

### Common options

- `--model`: registered embedding model name to query.
- `--backend`: `faiss` or `pgvector`.
- `--metric-type`: nearest-neighbor metric, default `cosine`.
- `--k`: number of results to return.
- `--vocabulary`: optional vocabulary filter on the candidate concepts.
- `--standard-only`: restrict search to standard OMOP concepts.
- `--embedding-path`: relative embedding endpoint path such as `/embeddings` or `/embed`.
- `--api-key`: optional bearer token for the embedding service.

The output is tab-separated with `rank`, `concept_id`, `similarity`, and
`concept_name`.

## `rebuild-index`

### Usage
```bash
omop-emb rebuild-index --model <MODEL> --backend faiss [OPTIONS]
```

This command rebuilds FAISS index files from the stored HDF5 vectors for an
already registered model. It is useful after recovering from inconsistent local
index files or when you want to materialize multiple metrics ahead of time.

### Common options

- `--model`: registered embedding model name.
- `--backend`: currently only `faiss` supports explicit rebuild.
- `--faiss-base-dir`: base directory containing the model's FAISS storage.
- `--metric-type`: repeat to rebuild multiple metrics. If omitted, all metrics
  supported by the model's index type are rebuilt.
- `--batch-size`: streaming batch size used while rebuilding from HDF5.

Examples:

```bash
omop-emb rebuild-index \
  --model my-embedding-model \
  --backend faiss \
  --faiss-base-dir ./data \
  --metric-type cosine
```

If you plan to search with multiple metrics, rebuild each required metric or
omit `--metric-type` to build all metrics supported by the model's index type.

## `switch-index-type`

### Usage
```bash
omop-emb switch-index-type --model <MODEL> --backend faiss [OPTIONS]
```

This command updates the registered FAISS `index_type` for an existing model,
stores any HNSW configuration in the model metadata, and optionally rebuilds
the derived FAISS index files from `embeddings.h5`.

Common options:

- `--index-type`: target FAISS index type such as `hnsw` or `flat`.
- `--metric-type`: metric(s) to rebuild after switching. If omitted, all
  metrics supported by the target index type are rebuilt.
- `--hnsw-num-neighbors`: HNSW `M` parameter for `--index-type hnsw`.
- `--hnsw-ef-search`: HNSW `efSearch` parameter for `--index-type hnsw`.
- `--hnsw-ef-construction`: HNSW `efConstruction` parameter for `--index-type hnsw`.
- `--rebuild/--no-rebuild`: rebuild FAISS files immediately after switching.

Example:

```bash
omop-emb switch-index-type \
  --model my-embedding-model \
  --backend faiss \
  --index-type hnsw \
  --hnsw-num-neighbors 48 \
  --hnsw-ef-search 96 \
  --hnsw-ef-construction 240
```
