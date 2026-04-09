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

## Quick Start

`omop-emb` reads OMOP concepts through SQLAlchemy and writes embedding metadata
through PostgreSQL even when you use the `faiss` backend for vector storage.

Example:

```bash
omop-emb add-embeddings \
  --api-base http://localhost:8000/v1/embeddings \
  --api-key <key> \
  --backend faiss \
  --faiss-base-dir ./data \
  --model my-embedding-model \
  --embedding-dim 1024 \
  --vocabulary SNOMED \
  --num-embeddings 500
```

Important:

- `OMOP_DATABASE_URL` must point to the PostgreSQL database that exposes your
  OMOP vocabulary tables.
- The code queries the OMOP `concept` table by ORM table name, not by a
  hard-coded schema-qualified path such as `vocabulary.concept`.
- PostgreSQL resolves that table through the connection `search_path`. If your
  OMOP vocabulary tables live in a non-default schema such as
  `staging_vocabulary`, your `search_path` must include that schema or the
  connection must otherwise resolve `concept` to the intended table.
- If you pass `--num-embeddings`, that is only a limit. The actual number of
  selected concepts may still be zero if the query resolves to the wrong table,
  the vocabulary filter matches nothing, or the model is already registered as
  embedded in the SQL registry.

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
