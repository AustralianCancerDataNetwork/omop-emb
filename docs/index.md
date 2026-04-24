# OMOP Embeddings

`omop-emb` is an optional package to super-charge [`omop-graph`](https://AustralianCancerDataNetwork.github.io/omop-graph) and provide additional graph reasoning tools for information retrieval and RAG-based knowledge extraction.

The package currently supports:

- dynamic embedding model registration
  - model metadata is stored locally in SQLite (`metadata.db`)
  - multiple embedding models can be tracked per backend and index type
- embedding and lookup for OMOP concepts
- supports various storage backends
  - [pgvector](https://github.com/pgvector/pgvector): storage in the original OMOP database
  - [FAISS](https://github.com/facebookresearch/faiss): on-disk vector storage and index files
- Extension to [`omop-alchemy`](https://AustralianCancerDataNetwork.github.io/OMOP_Alchemy/) to support new tables
- CLI scripts to add embeddings to an already existing OMOP CDM

## Installation

Install the backend you actually want to use:

```bash
pip install "omop-emb[pgvector]"
pip install "omop-emb[faiss]"
pip install "omop-emb[all]"
```

A plain `pip install omop-emb` installs only the shared core package.

At runtime, backend choice should also be explicit. The intended direction is:

- install-time choice via extras
- runtime choice via config such as `OMOP_EMB_BACKEND=pgvector` or `OMOP_EMB_BACKEND=faiss` or passing it as an argument to the respective interface (e.g. see [CLI reference](usage/cli.md))

## Environment Variables { data-toc-label="Environment Variables" }

| Variable | Description | Details |
|---|---|---|
| `OMOP_EMB_BACKEND` | Backend to use: `pgvector` or `faiss` | [Embedding Storage](usage/backend-selection.md) |
| `OMOP_EMB_BASE_STORAGE_DIR` | Base directory for `metadata.db` and FAISS artifacts | [Installation](usage/installation.md) |
| `OMOP_DATABASE_URL` | SQLAlchemy URL for the OMOP CDM database | [Installation](usage/installation.md) |
| `OMOP_EMB_DOCUMENT_EMBEDDING_PREFIX` | Task prefix prepended to concept texts at index time | [Asymmetric Embeddings](usage/asymmetric-embeddings.md) |
| `OMOP_EMB_QUERY_EMBEDDING_PREFIX` | Task prefix prepended to search queries at query time | [Asymmetric Embeddings](usage/asymmetric-embeddings.md) |

The prefix variables are optional and default to `""`. They are only needed for asymmetric embedding models (e.g. nomic-embed-text, E5, BGE) that require different task prefixes for indexing versus searching.


!!! info "Important caveats"
    - `omop-emb` depends on OMOP CDM database access for concept metadata and filtering.
    - Current operational and test coverage is PostgreSQL-focused. Extension planned in the future.


## Documentation overview
- [Installation](usage/installation.md)
- [**EmbeddingInterface Guide**](usage/interface-guide.md) — The primary API for embedding operations with model name validation
- [Embedding storage backends](usage/backend-selection.md)
- [CLI Reference](usage/cli.md)
