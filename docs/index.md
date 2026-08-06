# OMOP Embeddings

`omop-emb` generates and retrieves vector embeddings for OMOP CDM concepts. It works standalone out of the box (sqlite-vec, no external database required) and optionally scales to PostgreSQL via the pgvector extension.

The package supports:

- dynamic embedding model registration: multiple models per backend, tracked in the embedding database
- embedding and lookup for OMOP concepts across configurable storage backends
- <ins>**Two storage backends**</ins>:
    - **`sqlite-vec`** (default): zero-config, file-based or in-memory; no external service required
    - **`pgvector`**: PostgreSQL with the pgvector extension (FLAT sequential scan or HNSW SQL index)
- **FAISS** sidecar on top of `sqlite-vec` backend for approximate nearest-neighbour search
- CLI scripts to ingest OMOP CDM concepts and manage registered models

## Installation

Install the backend you want to use:

```bash
pip install omop-emb                       # sqlite-vec only (default backend)
pip install "omop-emb[pgvector]"           # adds PostgreSQL/pgvector support
pip install "omop-emb[faiss-cpu]"          # adds FAISS sidecar support
pip install "omop-emb[pgvector,faiss-cpu]" # everything
```

## Configuration

`omop-emb` is configured entirely through [oa-configurator](https://AustralianCancerDataNetwork.github.io/oa-configurator/) (`~/.config/omop/config.toml`); there are no `OMOP_EMB_*` environment variables. `OmopEmbConfig` (`[tools.omop_emb]`) has three fields:

| Field | References | Description |
|---|---|---|
| `cdm_db` | a `[databases.*]` entry, `kind = "cdm"` | The OMOP CDM database, for concept enrichment during ingestion/search |
| `embedding_model_name` | a `[models.*]` entry | Which model generates embeddings |
| `vector_store_name` | a `[vector_stores.*]` entry | Which storage backend (sqlite-vec or pgvector) holds them |

See [Getting Started: Configuration](getting-started/configuration.md) for the full setup walkthrough, and [Configuration reference](usage/configuration.md) for every field.

Document/query embedding prefixes for asymmetric models (nomic-embed-text, E5, BGE, ...) live on the `[models.*]` entry itself (`document_prefix`/`query_prefix`), not on `omop-emb`'s own config; see [Asymmetric Embeddings](usage/asymmetric-embeddings.md).

## Documentation overview

- [Getting Started: Configuration](getting-started/configuration.md)
- [Installation](usage/installation.md)
- [Embedding storage backends](usage/backend-selection.md)
- [CLI Reference](usage/cli.md)
- [Asymmetric Embeddings](usage/asymmetric-embeddings.md)
- [Interface guide](usage/interface-guide.md)
