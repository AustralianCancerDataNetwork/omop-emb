# Embedding Generation CLI

This tool generates vector embeddings for OMOP CDM concepts and stores them in the configured embedding backend.

At present, the production CLI path is PostgreSQL-oriented and stores embeddings in Postgres/pgvector-backed model tables. It specifically targets concepts that do not yet have embeddings and processes them in batches.

!!! note "Supported Models"

    Currently supported are only Ollama models
---

## Prerequisites

- **Installation**: install the backend dependencies you plan to use:

  ```bash
  pip install "omop-emb[pgvector]"
  # or
  pip install "omop-emb[faiss]"
  ```

- **Database**: PostgreSQL implementation of OMOP CDM. See [`omop-graph` documentation](https://AustralianCancerDataNetwork.github.io/omop-graph) for information how to setup.
- **Environment**: `OMOP_DATABASE_URL` must be exported or present in `.env`  (e.g., `postgresql://user:pass@localhost:5432/omop`).
- **Backend config**: set `OMOP_EMB_BACKEND` (`pgvector` or `faiss`) and optionally `OMOP_EMB_BASE_STORAGE_DIR`.
- **Connectivity**: Access to an OpenAI-compatible embeddings endpoint. *Currently only Ollama supported*.

!!! note "Backend Scope"

    `omop-emb` now defines a backend abstraction layer for both PostgreSQL and FAISS-style storage.
    The current `add-embeddings` CLI still targets the PostgreSQL backend path.

---

## `add-embeddings`

### Usage
```bash
omop-emb add-embeddings --api-base <URL> --api-key <KEY> [OPTIONS]
```
where `[OPTIONS]` are optional arguments that can be specified as described below.


### Command Options

| Option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`--api-base`** | | `String` | **Required** | Base URL for the embedding API service. |
| **`--api-key`** | | `String` | **Required** | API key for the embedding API provider. |
| **`--index-type`** | | `IndexType` | `FLAT` | The storage index for the embeddings for retrieval. Currently supported: `FLAT`. |
| **`--batch-size`** | `-b` | `Integer` | `100` | Number of concepts to process in each chunk. |
| **`--model`** | `-m` | `String` | `text-embedding-3-small` | Name of the embedding model to use for generating vectors. |
| **`--backend`** | | `Literal['pgvector', 'faiss']` | `None` | Embedding backend to use (can be replaced by `OMOP_EMB_BACKEND`). Requires the corresponding optional dependency. |
| **`--storage-base-dir`** | | `String` | `None` | Optional base directory for backend storage and local metadata registry (`metadata.db`). |
| **`--standard-only`** | | `Boolean` | `False` | If set, only generate embeddings for OMOP standard concepts (`standard_concept = 'S'`). |
| **`--vocabulary`** | | `List[String]` | `None` | Filter to embed concepts only from specific OMOP vocabularies. |
| **`--num-embeddings`** | `-n` | `Integer` | `None` | Limit the number of concepts processed (useful for testing). |

## Environment Variables

- `OMOP_DATABASE_URL`: OMOP CDM database connection string.
- `OMOP_EMB_BACKEND`: backend selector used when `--backend` is not provided.
- `OMOP_EMB_BASE_STORAGE_DIR`: local storage root for metadata and file-based artifacts. If unset, `omop-emb` defaults to `~/.omop_emb`.

Paths that include `~` are expanded automatically.

---

## `migrate-legacy-pgvector-registry`

Migrate legacy pgvector registry rows from a source database table into the local metadata registry (`metadata.db`).

This command is intended for compatibility with older setups that kept registry metadata in the database instead of the local metadata store.

### Usage
```bash
omop-emb migrate-legacy-pgvector-registry [OPTIONS]
```

### Options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`--storage-base-dir`** | `String` | `None` | Optional path to local metadata registry location. If unset, falls back to `OMOP_EMB_BASE_STORAGE_DIR`, otherwise defaults to `~/.omop_emb`. |
| **`--source-database-url`** | `String` | `OMOP_DATABASE_URL` | Source database URL containing the legacy registry table. |
| **`--legacy-table`** | `String` | `model_registry` | Name of the legacy registry table in the source database. |
| **`--dry-run`** | `Boolean` | `False` | Show what would be migrated without writing changes. |
| **`--drop-legacy-registry`** | `Boolean` | `False` | Drop the legacy table after successful migration. |

### Recommended Migration Flow

1. Validate what will migrate:

```bash
omop-emb migrate-legacy-pgvector-registry --dry-run
```

2. Run the migration:

```bash
omop-emb migrate-legacy-pgvector-registry
```

3. Optionally remove legacy table after verification:

```bash
omop-emb migrate-legacy-pgvector-registry --drop-legacy-registry
```

### Field Mapping

The migration command supports these legacy field names when reading rows:

- model name: `model_name`
- dimensions: `dimensions`
- index type: `index_type` (fallback: `index_method`)
- storage identifier: `storage_identifier` (fallback: `table_name`)
- metadata: `details` (fallback: `metadata`)
