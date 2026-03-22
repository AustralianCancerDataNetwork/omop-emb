# OMOP Embeddings

`omop-emb` is an optional package to super-charge `omop-graph` and provide additional graph reasoning tools for information retrieval and RAG-based knowledge extraction.

The package currently supports:

- dynamic embedding model registration
- embedding population and lookup for OMOP concepts
- a PostgreSQL/pgvector backend
- an initial FAISS backend structure
- [`omop-alchemy`](https://AustralianCancerDataNetwork.github.io/OMOP_Alchemy/) wrapper of new tables
- CLI scripts to add embeddings to an already existing OMOP CDM

## Installation

Install the backend you actually want to use:

```bash
pip install "omop-emb[postgres]"
pip install "omop-emb[faiss]"
pip install "omop-emb[all]"
```

A plain `pip install omop-emb` installs only the shared core package.

At runtime, backend choice should also be explicit. The intended direction is:

- install-time choice via extras
- runtime choice via config such as `OMOP_EMB_BACKEND=postgres` or `OMOP_EMB_BACKEND=faiss`

Important caveats:

- PostgreSQL-specific embedding dependencies are optional, but a relational
  database backend is still required for OMOP access and model registration.
- Even when using the FAISS backend for embedding retrieval, database-backed
  OMOP metadata remains required.
- Database backends other than PostgreSQL have not yet been tested.

## Documentation overview
- [Installation](usage/installation.md)
- [Backend Selection](usage/backend-selection.md)
- [CLI Reference](usage/cli.md)
