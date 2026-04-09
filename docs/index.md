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

Recommended runtime environment variables:

- **`OMOP_EMB_BACKEND`**: `pgvector` or `faiss`
- **`OMOP_EMB_BASE_STORAGE_DIR`**: base directory for local metadata and FAISS artifacts; defaults to `omop_emb/.omop_emb` the root direcotry of hte package.
- **`OMOP_DATABASE_URL`**: OMOP CDM database URL


!!! info "Important caveats"
    - `omop-emb` depends on OMOP CDM database access for concept metadata and filtering.
    - Current operational and test coverage is PostgreSQL-focused. Extension planned in the future.


## Documentation overview
- [Installation](usage/installation.md)
- [Embedding storage backends](usage/backend-selection.md)
- [CLI Reference](usage/cli.md)
