# OMOP Embeddings

`omop-emb` is an optional package to super-charge [`omop-graph`](https://AustralianCancerDataNetwork.github.io/omop-graph) and provide additional graph reasoning tools for information retrieval and RAG-based knowledge extraction.

The package currently supports:

- dynamic embedding model registration
  - multiple embedding models can be stored in the respective backend
- embedding and lookup for OMOP concepts
- supports various backends with a PostgreSQL linker
  - [pgvector](https://github.com/pgvector/pgvector): storage in the original OMOP database
  - [FAISS](https://github.com/facebookresearch/faiss): efficient storage on disk for low-RAM applications 
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


!!! info "Important caveats"

  - `omop-emb` depends on an OMOP PostgreSQL database for storage of embeddings (pgvector) or to keep track of already embedded concepts.


## Documentation overview
- [Installation](usage/installation.md)
- [Backend Selection](usage/backend-selection.md)
- [CLI Reference](usage/cli.md)
