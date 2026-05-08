# CLI Reference

`omop-emb` provides a CLI for concept ingestion, similarity search, index
management, and diagnostics. All commands load a `.env` file from the working
directory automatically.

## Prerequisites

- **Backend installed**: `pip install omop-emb` (sqlite-vec) or
  `pip install "omop-emb[pgvector]"`.
- **Backend configured**: set `OMOP_EMB_BACKEND` and the matching connection
  variables (see [Installation](installation.md)).
- **Embedding API**: an OpenAI-compatible embeddings endpoint (e.g. Ollama or
  OpenAI). Required for ingestion and search commands.
- **OMOP CDM** (`OMOP_CDM_DB_URL`): required only for concept ingestion
  (`add-embeddings`, `add-embeddings-with-index`). Not required for search,
  `list-models`, `rebuild-index`, `delete-model`, or diagnostics.

---

## Environment Variables

### Backend

| Variable | Default | Description |
|---|---|---|
| `OMOP_EMB_BACKEND` | `sqlitevec` | Backend selector: `sqlitevec`, `pgvector`. |

### sqlite-vec

| Variable | Description |
|---|---|
| `OMOP_EMB_SQLITE_PATH` | Path to the sqlite-vec database file (or `:memory:`). |

### pgvector (individual components)

| Variable | Default | Description |
|---|---|---|
| `OMOP_EMB_DB_HOST` | — | PostgreSQL host. |
| `OMOP_EMB_DB_PORT` | `5432` | PostgreSQL port. |
| `OMOP_EMB_DB_USER` | — | PostgreSQL user. |
| `OMOP_EMB_DB_PASSWORD` | — | PostgreSQL password. |
| `OMOP_EMB_DB_NAME` | — | PostgreSQL database name. |
| `OMOP_EMB_DB_DRIVER` | `postgresql+psycopg` | SQLAlchemy driver string. |
| `OMOP_EMB_DB_URL` | — | Full connection URL. Overrides individual components. |

### Ingestion (CDM access)

| Variable | Description |
|---|---|
| `OMOP_CDM_DB_URL` | SQLAlchemy URL for the OMOP CDM database. |

---

## Ingestion commands

### `add-embeddings`

Bulk-generate and store embeddings for OMOP concepts that do not yet have
embeddings. Models are registered with a FLAT index; use `rebuild-index`
afterwards to build an HNSW index.

```bash
omop-emb add-embeddings --api-base <URL> --api-key <KEY> [OPTIONS]
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--api-base` | | **required** | Base URL of the embedding API. |
| `--api-key` | | **required** | API key for the embedding API. |
| `--model` | `-m` | `text-embedding-3-small` | Embedding model name. |
| `--batch-size` | `-b` | `100` | Concepts per API batch. |
| `--standard-only` | | `False` | Embed only standard concepts (`standard_concept = 'S'`). |
| `--vocabulary` | | `None` | Restrict to specific OMOP vocabularies (repeatable). |
| `--domain` | | `None` | Restrict to specific OMOP domains (repeatable). |
| `--num-embeddings` | `-n` | `None` | Cap on total concepts processed (useful for testing). |
| `--cdm-batch-size` | | `50000` | Batch size for fetching concept metadata from the CDM. |
| `--verbose` | `-v` | | Increase log verbosity (pass twice for DEBUG). |

---

### `add-embeddings-with-index`

Ingest embeddings and immediately build an index in one step. Equivalent to
running `add-embeddings` followed by `create-index`.

```bash
omop-emb add-embeddings-with-index --api-base <URL> --api-key <KEY> [OPTIONS]
```

Accepts all options from `add-embeddings`, plus:

| Option | Default | Description |
|---|---|---|
| `--index-type` | `flat` | Index to build after ingestion (`flat` or `hnsw`). |
| `--metric-type` | `cosine` | Distance metric. Required and locked in for `hnsw`. |
| `--index-hnsw-num-neighbors` | `None` | HNSW graph connectivity (`M`). |
| `--index-hnsw-ef-search` | `None` | HNSW query recall parameter. |
| `--index-ef-construction` | `None` | HNSW build quality parameter. |

---

### `create-index`

Build or rebuild the index for a model that already has embeddings stored.
`--api-base` and `--api-key` are used only to resolve the canonical model name.

```bash
omop-emb create-index --api-base <URL> --api-key <KEY> --model <NAME> [OPTIONS]
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--api-base` | | **required** | Base URL of the embedding API. |
| `--api-key` | | **required** | API key. |
| `--model` | `-m` | `text-embedding-3-small` | Embedding model name. |
| `--index-type` | | `flat` | `flat` or `hnsw`. |
| `--metric-type` | | `cosine` | Distance metric. Required and locked in for `hnsw`. |
| `--index-hnsw-num-neighbors` | | `None` | HNSW `M` parameter. |
| `--index-hnsw-ef-search` | | `None` | HNSW query recall parameter. |
| `--index-ef-construction` | | `None` | HNSW build quality parameter. |
| `--verbose` | `-v` | | Increase log verbosity. |

---

## Search

### `search`

Query stored embeddings for nearest OMOP concepts. Outputs tab-separated rows:
`query_id`, `query_text`, `rank`, `concept_id`, `similarity`, `concept_name`.

If `OMOP_CDM_DB_URL` is set, results are enriched with concept names from the
CDM. Without it, the `concept_name` column is left empty.

```bash
omop-emb search --api-base <URL> --api-key <KEY> --query "hypertension" [OPTIONS]
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--api-base` | | **required** | Base URL of the embedding API. |
| `--api-key` | | **required** | API key. |
| `--query` | | `None` | Query text (repeatable). At least one of `--query` or `--queries-file` is required. |
| `--queries-file` | | `None` | Path to a `.txt` file with one query per line. |
| `--model` | `-m` | `text-embedding-3-small` | Embedding model name. |
| `--metric-type` | | `cosine` | Distance metric for search. |
| `--k` | | `10` | Number of nearest concepts to return per query. |
| `--batch-size` | `-b` | `100` | Batch size for embedding generation. |
| `--standard-only` | | `False` | Return only standard OMOP concepts. |
| `--vocabulary` | | `None` | Filter results to specific vocabularies (repeatable). |
| `--domain` | | `None` | Filter results to specific domains (repeatable). |
| `--faiss-cache-dir` | | `None` | Use a FAISS sidecar index for search instead of the primary backend. Requires `omop-emb[faiss-cpu]`. |
| `--verbose` | `-v` | | Increase log verbosity. |

---

## Model management

### `list-models`

List all registered embedding models in the configured backend.

```bash
omop-emb list-models [OPTIONS]
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--model` | `-m` | `None` | Filter by model name. |
| `--provider-type` | | `None` | Filter by provider (`ollama`, `openai`). |
| `--verbose` | `-v` | | Increase log verbosity. |

---

### `rebuild-index`

Build or rebuild the storage index for an already-registered model. Use this to
switch between FLAT and HNSW without re-ingesting.  The canonical model name is
passed directly via `--model`; supply `--provider-type` to canonicalize a raw
name if needed.

```bash
omop-emb rebuild-index --model <CANONICAL_NAME> [OPTIONS]
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--model` | `-m` | **required** | Canonical model name. |
| `--provider-type` | | `None` | Provider used to canonicalize the model name when needed. |
| `--index-type` | | `flat` | `flat` or `hnsw`. |
| `--metric-type` | | `cosine` | Distance metric (required and locked in for `hnsw`). |
| `--index-hnsw-num-neighbors` | | `None` | HNSW `M` parameter. |
| `--index-hnsw-ef-search` | | `None` | HNSW query recall parameter. |
| `--index-ef-construction` | | `None` | HNSW build quality parameter. |
| `--verbose` | `-v` | | Increase log verbosity. |

---

### `delete-model`

Permanently delete a registered model and all its stored embeddings. This
operation is irreversible.

```bash
omop-emb delete-model --model <NAME> [OPTIONS]
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--model` | `-m` | **required** | Canonical model name. |
| `--provider-type` | | `None` | Provider used to canonicalize the model name when needed. |
| `--yes` | `-y` | `False` | Skip confirmation prompt. |
| `--verbose` | `-v` | | Increase log verbosity. |

---

## Diagnostics

### `health-check`

Verify backend connectivity and list registered models.

```bash
omop-emb health-check
```

---

## FAISS sidecar

Requires `pip install "omop-emb[faiss-cpu]"`.

FAISS indices are exported from the primary backend (sqlite-vec or pgvector)
and stored on disk as `.faiss` / `.json` / `metadata.npz` files under a model
sub-directory of `--cache-dir`.  The primary backend remains the source of
truth; FAISS is a read-acceleration layer only.

### `export-faiss-cache`

Export all embeddings from the primary backend into a FAISS index on disk.

```bash
omop-emb export-faiss-cache --model <NAME> --cache-dir <DIR> [OPTIONS]
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--model` | `-m` | **required** | Canonical model name. |
| `--cache-dir` | | **required** | Root directory for FAISS index files. |
| `--metric-type` | | `cosine` | Distance metric for the FAISS index (`cosine` or `l2`). |
| `--index-type` | | `flat` | FAISS index type: `flat` (exact) or `hnsw` (approximate). |
| `--hnsw-m` | | `32` | HNSW number of neighbours. Only used when `--index-type=hnsw`. |
| `--provider-type` | | `None` | Provider used to canonicalize the model name when needed. |
| `--batch-size` | `-b` | `100000` | Embeddings fetched per backend round-trip. |
| `--verbose` | `-v` | | Increase log verbosity. |

---

### `check-faiss-cache`

Check whether the FAISS index on disk is fresh relative to the primary backend.
Exits with code `0` if fresh, `1` if stale or missing.

```bash
omop-emb check-faiss-cache --model <NAME> --cache-dir <DIR> [OPTIONS]
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--model` | `-m` | **required** | Canonical model name. |
| `--cache-dir` | | **required** | Root cache directory. |
| `--metric-type` | | `cosine` | Metric of the index to check. |
| `--index-type` | | `flat` | Index type to check (`flat` or `hnsw`). |
| `--provider-type` | | `None` | Provider used to canonicalize the model name when needed. |
| `--verbose` | `-v` | | Increase log verbosity. |

---

### `import-faiss-cache`

Import embeddings from an on-disk FAISS index back into the primary backend.
Reconstructs raw vectors from the `.faiss` file (exact reconstruction requires
`flat` or `hnsw` index types; IVF/PQ indices are lossy and unsupported).

```bash
omop-emb import-faiss-cache --model <NAME> --cache-dir <DIR> --provider-type <TYPE> [OPTIONS]
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--model` | `-m` | **required** | Canonical model name. |
| `--cache-dir` | | **required** | Root cache directory containing the FAISS index files. |
| `--provider-type` | | **required** | Embedding provider. Used to register the model if not already present. |
| `--metric-type` | | `cosine` | Metric of the index to import from. |
| `--index-type` | | `flat` | Index type to import from (`flat` or `hnsw`). |
| `--batch-size` | `-b` | `10000` | Vectors upserted per backend call. |
| `--force` | | `False` | Overwrite existing embeddings without prompting. |
| `--verbose` | `-v` | | Increase log verbosity. |
