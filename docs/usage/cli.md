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

### Command Options

| Option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`--api-base`** | | `String` | **Required** | Base URL for the embedding API service. |
| **`--api-key`** | | `String` | **Required** | API key for the embedding API provider. |
| **`--index-type`** | | `IndexType` | `FLAT` | The storage index for the embeddings for retrieval. Currently supported: `FLAT`. |
| **`--batch-size`** | `-b` | `Integer` | `100` | Number of concepts to process in each chunk. |
| **`--model`** | `-m` | `String` | `text-embedding-3-small` | Name of the embedding model to use for generating vectors. |
| **`--backend`** | | `Literal['pgvector', 'faiss']` | `None` | Embedding backend to use (can be replaced by `OMOP_EMB_BACKEND`). Requires the corresponding optional dependency. |
| **`--faiss-base-dir`** | | `String` | `None` | Optional base directory for FAISS backend storage. |
| **`--standard-only`** | | `Boolean` | `False` | If set, only generate embeddings for OMOP standard concepts (`standard_concept = 'S'`). |
| **`--vocabulary`** | | `List[String]` | `None` | Filter to embed concepts only from specific OMOP vocabularies. |
| **`--num-embeddings`** | `-n` | `Integer` | `None` | Limit the number of concepts processed (useful for testing). |

## Environment Variables

- `OMOP_DATABASE_URL`: OMOP CDM database connection string.
- `OMOP_EMB_BACKEND`: backend selector used when `--backend` is not provided.
- `OMOP_EMB_BASE_STORAGE_DIR`: local storage root for metadata and file-based artifacts. If unset, `omop-emb` defaults to `./.omop_emb` in the current working directory.

Paths that include `~` are expanded automatically.

---

## `export-pgvector`

Export pgvector embedding tables to CSV files plus a manifest so they can be restored later.

### Usage
```bash
omop-emb export-pgvector --output-dir <SNAPSHOT_DIR> [OPTIONS]
```

### Options

| Option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`--output-dir`** | `-o` | `String` | **Required** | Directory where snapshot files are written. |
| **`--storage-base-dir`** | | `String` | `None` | Optional path to local metadata registry (`metadata.db`). If unset, falls back to `OMOP_EMB_BASE_STORAGE_DIR`, otherwise defaults to `./.omop_emb` in the current working directory. Paths with `~` are expanded. |
| **`--model`** | `-m` | `List[String]` | `None` | Optional model-name filter. Repeat to export specific models only. |
| **`--index-type`** | | `IndexType` | `None` | Optional index-type filter. |

### Output

The command writes:

- `manifest.json`: snapshot metadata and table mapping
- One CSV per embedding table named `<storage_identifier>.csv`

---

## `import-pgvector`

Restore pgvector embedding tables from files previously created by `export-pgvector`.

### Usage
```bash
omop-emb import-pgvector --input-dir <SNAPSHOT_DIR> [OPTIONS]
```

### Options

| Option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`--input-dir`** | `-i` | `String` | **Required** | Directory containing `manifest.json` and CSV files. |
| **`--storage-base-dir`** | | `String` | `None` | Optional path to local metadata registry (`metadata.db`). If unset, falls back to `OMOP_EMB_BASE_STORAGE_DIR`, otherwise defaults to `./.omop_emb` in the current working directory. Paths with `~` are expanded. |
| **`--replace`** | | `Boolean` | `False` | If set, truncate destination embedding tables before import. |
| **`--batch-size`** | `-b` | `Integer` | `5000` | Number of rows inserted per SQL batch. |

### Notes

- Import re-registers pgvector models into local metadata before loading rows.
- Import uses upsert semantics (`ON CONFLICT (concept_id) DO UPDATE`) unless `--replace` is set.
