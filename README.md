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
