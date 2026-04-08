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

- **Database**: Postgres implementation of OMOP CDM. See [`omop-graph` documentation](reference-missing) for information how to setup.
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
- `OMOP_EMB_BASE_STORAGE_DIR`: local storage root for metadata and file-based artifacts.
