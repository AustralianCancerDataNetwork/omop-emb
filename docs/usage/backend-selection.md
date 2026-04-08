# Backend Selection

`omop-emb` now has a backend abstraction layer so embedding storage and
retrieval can be selected explicitly instead of being inferred implicitly from
whatever happens to be installed.

## Supported backend names

The current backend factory recognizes:

- `pgvector`: The [pgvector](https://github.com/pgvector/pgvector) extension to a standard postgres database to store embeddings directly in the database.
- `faiss`: The [FAISS](https://github.com/facebookresearch/faiss) storage solution for on-disk storage.

There is no implicit default backend name. You must pass one explicitly or set
`OMOP_EMB_BACKEND`.

## Runtime selection

The intended pattern is:

1. choose the backend at install time with package extras
2. choose the backend again at runtime explicitly

Examples:

```bash
export OMOP_EMB_BACKEND=pgvector
export OMOP_EMB_BACKEND=faiss
export OMOP_EMB_BASE_STORAGE_DIR=$HOME/.omop_emb
```

You can also pass the backend name directly in Python.

## Python factory

The backend factory lives in `omop_emb.backends`:

```python
from omop_emb.backends import get_embedding_backend

backend = get_embedding_backend("pgvector")
backend = get_embedding_backend("faiss")
```

The factory currently exposes:

- `get_embedding_backend(...)`
- `normalize_backend_name(...)`

## Why explicit selection is necessary

Explicit backend selection improves clarity in a multi-backend world:

- users can see which backend they intended to use
- missing optional dependencies fail clearly
- the system avoids silent fallback between incompatible storage implementations

This is especially important when embeddings affect retrieval behavior, because
silent fallback can make users think semantic retrieval is active when it is
not.

## Dependency errors

If a backend is requested but its optional dependencies are missing, the
factory raises an explicit backend dependency error rather than falling back to
another backend.

This is the intended behavior.

Examples of the error classes exposed by the backend layer:

- `EmbeddingBackendDependencyError`
- `UnknownEmbeddingBackendError`
- `EmbeddingBackendConfigurationError`

## Current scope

At the moment:

- the backend abstraction and backend factory exist
- PostgreSQL and FAISS backend classes exist
- the production CLI path still targets the PostgreSQL embedding workflow
- PostgreSQL-specific embedding dependencies are optional, but OMOP database
  access is still required for concept metadata
- model registration metadata is stored locally in SQLite (`metadata.db`) under
  `OMOP_EMB_BASE_STORAGE_DIR`
- database backends other than PostgreSQL have not yet been tested

So this page documents the selection model and Python interface shape now, even
before every runtime path has been migrated to delegate through the backend
factory.
