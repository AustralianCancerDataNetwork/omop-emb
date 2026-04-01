# omop-emb
Embedding layer for OMOP CDM.

## Installation

`omop-emb` now exposes backend-specific optional dependencies so installation
can match the embedding backend you actually intend to use.

```bash
pip install "omop-emb[postgres]"
pip install "omop-emb[faiss]"
pip install "omop-emb[all]"
```

Notes:

- `postgres` installs the PostgreSQL/pgvector dependencies.
- `faiss` installs the FAISS-based backend dependencies. This currently only includes CPU support
- `all` installs both backend stacks for development or mixed environments.
- A plain `pip install omop-emb` installs the shared core package only.
- PostgreSQL-specific embedding dependencies are now optional, but `omop-emb`
  still requires some database backend for OMOP access and model registration.
- Non-PostgreSQL database backends have not yet been tested.

Extended documentation can be found [here](https://AustralianCancerDataNetwork.github.io/omop-emb).

# Project Roadmap

- [x] Interface for postgres storage of vectors
- [x] Interface for FAISS storage of embeddings
- [ ] Extensive unit testing
    - [ ] Backend testing
    - [ ] Corruption and restoration of DB testing
- [ ] Support non-Flat indices for each backend
- [ ] `faiss` GPU support
- [ ] [`pgvectorscale`](https://github.com/timescale/pgvectorscale) support
- [ ] Vector-quantisation for more efficient storage
