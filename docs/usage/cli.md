# CLI Reference

`omop-emb` provides a CLI for concept ingestion, index management, and
diagnostics. All commands load a `.env` file from the working directory
automatically.

## Prerequisites

- **Backend installed**: `pip install omop-emb` (sqlite-vec) or
  `pip install "omop-emb[pgvector]"`.
- **Backend configured**: set `OMOP_EMB_BACKEND` and the matching connection
  variables (see [Installation](installation.md)).
- **Embedding API**: an OpenAI-compatible embeddings endpoint (e.g. Ollama or
  OpenAI). Ingestion commands require `--api-base` and `--api-key`.
- **OMOP CDM** (`OMOP_CDM_DB_URL`): required only for the ingestion commands
  (`add-embeddings`, `add-embeddings-with-index`). Not required for
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
| `--verbose` | `-v` | | Increase log verbosity (pass twice for DEBUG). |

---

### `add-embeddings-with-index`

Ingest embeddings and immediately build an HNSW index in one step. Equivalent
to running `add-embeddings` followed by `rebuild-index`.

```bash
omop-emb add-embeddings-with-index --api-base <URL> --api-key <KEY> [OPTIONS]
```

Accepts all options from `add-embeddings`, plus:

| Option | Default | Description |
|---|---|---|
| `--index-type` | `flat` | Index to build after ingestion (`flat` or `hnsw`). |
| `--metric-type` | `cosine` | Distance metric. Required and locked in for `hnsw`. |
| `--num-neighbors` | `16` | HNSW graph connectivity (`M`). |
| `--ef-search` | `16` | HNSW query recall parameter. |
| `--ef-construction` | `64` | HNSW build quality parameter. |

---

### `create-index`

Build or rebuild the index for a model that already has embeddings stored.

```bash
omop-emb create-index --api-base <URL> --api-key <KEY> --model <NAME> [OPTIONS]
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--api-base` | | **required** | Base URL of the embedding API (used to resolve canonical model name). |
| `--api-key` | | **required** | API key. |
| `--model` | `-m` | **required** | Embedding model name. |
| `--index-type` | | `flat` | `flat` or `hnsw`. |
| `--metric-type` | | `cosine` | Distance metric. Required and locked in for `hnsw`. |
| `--num-neighbors` | | `16` | HNSW `M` parameter. |
| `--ef-search` | | `16` | HNSW query recall parameter. |
| `--ef-construction` | | `64` | HNSW build quality parameter. |

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
switch between FLAT and HNSW without re-ingesting.

```bash
omop-emb rebuild-index --model <NAME> [OPTIONS]
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--model` | `-m` | **required** | Canonical model name. |
| `--provider-type` | | `openai` | Provider the model was registered with. |
| `--index-type` | | `flat` | `flat` or `hnsw`. |
| `--metric-type` | | `cosine` | Distance metric (required and locked in for `hnsw`). |
| `--num-neighbors` | | `16` | HNSW `M` parameter. |
| `--ef-search` | | `16` | HNSW query recall parameter. |
| `--ef-construction` | | `64` | HNSW build quality parameter. |
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
| `--provider-type` | | `openai` | Provider the model was registered with. |
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

### `export-faiss-cache`

Export a FAISS sidecar cache from the embedding store for a registered model.

```bash
omop-emb export-faiss-cache --model <NAME> [OPTIONS]
```

### `check-faiss-cache`

Check whether the FAISS sidecar cache is stale relative to the primary backend.

```bash
omop-emb check-faiss-cache --model <NAME> [OPTIONS]
```
