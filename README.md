# omop-emb
Embedding layer for OMOP CDM.

`omop-emb` now separates model metadata from embedding storage:

- model metadata is stored locally in SQLite (`metadata.db`)
- embedding vectors are stored by the selected backend (`pgvector` or `faiss`)
- OMOP concept metadata remains in the OMOP CDM database

## Installation

`omop-emb` now exposes backend-specific optional dependencies so installation
can match the embedding backend you actually intend to use.

```bash
pip install "omop-emb[pgvector]"
pip install "omop-emb[faiss]"
pip install "omop-emb[all]"
```

Notes:

- `pgvector` installs the PostgreSQL/pgvector dependencies.
- `faiss` installs the FAISS-based backend dependencies. This currently only includes CPU support
- `all` installs both backend stacks for development or mixed environments.
- A plain `pip install omop-emb` installs the shared core package only.
- PostgreSQL-specific embedding dependencies are optional, but `omop-emb`
  still requires OMOP CDM database access.
- Non-PostgreSQL database backends have not yet been tested.

## Runtime Configuration

Common environment variables:

- `OMOP_EMB_BACKEND`: backend name (`pgvector` or `faiss`) used by the backend factory.
- `OMOP_EMB_BASE_STORAGE_DIR`: local base directory for `omop-emb` artifacts, including local metadata (`metadata.db`) and FAISS files. If unset, `omop-emb` defaults to `./.omop_emb` in the current working directory.
- `OMOP_DATABASE_URL`: SQLAlchemy URL for the OMOP CDM database.
- `OMOP_EMB_DOCUMENT_EMBEDDING_PREFIX`: task prefix prepended to concept texts at index time. Required for asymmetric models (e.g. `search_document: ` for nomic-embed-text, `passage: ` for E5).
- `OMOP_EMB_QUERY_EMBEDDING_PREFIX`: task prefix prepended to search queries at query time. Required for asymmetric models (e.g. `search_query: ` for nomic-embed-text, `query: ` for E5).

The prefix variables default to `""` and are safe to omit for symmetric models. See [Asymmetric Embeddings](https://AustralianCancerDataNetwork.github.io/omop-emb/usage/asymmetric-embeddings/) for details.

Extended documentation can be found [here](https://AustralianCancerDataNetwork.github.io/omop-emb).

# Project Roadmap

- [x] Interface for PostgreSQL storage of vectors
- [x] Interface for FAISS storage of embeddings
- [x] Extensive unit testing
    - [x] Backend testing
    - [x] Corruption and restoration of DB testing
- [ ] Support importing and exporting of calculated embeddings
- [ ] Support non-Flat indices for each backend
- [ ] `faiss` GPU support
- [ ] [`pgvectorscale`](https://github.com/timescale/pgvectorscale) support
- [ ] Vector-quantisation for more efficient storage
