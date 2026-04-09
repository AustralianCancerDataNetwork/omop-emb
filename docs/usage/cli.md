# Embedding Generation CLI

This tool generates vector embeddings for OMOP CDM concepts and stores them in the configured embedding backend.

At present, the production CLI path is PostgreSQL-oriented and stores embeddings in Postgres/pgvector-backed model tables. It specifically targets concepts that do not yet have embeddings and processes them in batches.

!!! note "Supported Models"

    The CLI works with the underlying `omop-llm` client. If your endpoint is
    OpenAI-compatible and the client cannot infer embedding dimensions, pass
    `--embedding-dim` or set `OMOP_EMB_EMBEDDING_DIM`.
---

## Prerequisites

- **Installation**: install the PostgreSQL backend dependencies:

  ```bash
  pip install "omop-emb[postgres]"
  ```

- **Database**: Postgres implementation of OMOP CDM. See [`omop-graph` documentation](reference-missing) for information how to setup.
- **Environment**: `OMOP_DATABASE_URL` must be exported or existing in the .env file  (e.g., `postgresql://user:pass@localhost:5432/omop`).
- **Connectivity**: Access to an embeddings endpoint compatible with the
  configured `omop-llm` client.

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
| **`--model`** | `-m` | `String` | `OMOP_EMB_MODEL` or `text-embedding-3-small` | Name of the embedding model to use for generating vectors. |
| **`--embedding-dim`** | | `Integer` | `OMOP_EMB_EMBEDDING_DIM` or auto-detect | Explicit embedding dimension override for models whose dimensions cannot be inferred automatically. |
| **`--backend`** | | `Literal['pgvector', 'faiss']` | `None` | Embedding backend to use (can be replaced by `OMOP_EMB_BACKEND` env var). Requires the respective backend installed using `pip install omop-emb[pgvector or faiss]` |
| **`--faiss-base-dir`** | | `String` | `None` | Optional base directory for FAISS backend storage. |
| **`--standard-only`** | | `Boolean` | `False` | If set, only generate embeddings for OMOP standard concepts (`standard_concept = 'S'`). |
| **`--vocabulary`** | | `List[String]` | `None` | Filter to embed concepts only from specific OMOP vocabularies. |
| **`--num-embeddings`** | `-n` | `Integer` | `None` | Limit the number of concepts processed (useful for testing). |

### Environment Variables

- `OMOP_DATABASE_URL`: database connection string for the OMOP store.
- `OMOP_EMB_MODEL`: default model name if `--model` is omitted.
- `OMOP_EMB_EMBEDDING_DIM`: explicit embedding dimension override if `--embedding-dim` is omitted.
- `OMOP_EMB_BACKEND`: default embedding backend if `--backend` is omitted.
