# Backend Selection for embeddings

`omop-emb` now has a backend abstraction layer so embedding storage and
retrieval can be selected explicitly instead of being inferred implicitly from
whatever happens to be installed.

## Supported backend names

The current backend factory recognizes:

- `pgvector`: The [pgvector](https://github.com/pgvector/pgvector) extension to a standard PostgreSQL database to store embeddings directly in the database.
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
export OMOP_EMB_BASE_STORAGE_DIR=$PWD/.omop_emb
```

You can also pass the backend name directly in Python.

Storage directory behavior:

- If `OMOP_EMB_BASE_STORAGE_DIR` is unset and no explicit path is passed, `omop-emb` defaults to `~/.omop_emb`.
- If a path includes `~`, it is expanded (for example `~/.omop_emb`).

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

## Index Types

Each backend supports two index types controlled by an `IndexConfig` object passed at model registration.

```python
from omop_emb.backends import FlatIndexConfig, HNSWIndexConfig
```

### FLAT

Sequential scan — no index structure is built. Every query compares the vector against all stored embeddings. Correct by construction and requires no warm-up step.

```python
FlatIndexConfig()  # no parameters
```

Use FLAT when the corpus is small (tens of thousands of concepts), or when exact results are required.

### HNSW

Hierarchical Navigable Small World graph. Approximate nearest-neighbour search with sub-linear query time. Configurable via three parameters:

| Parameter | Default | Effect |
|---|---|---|
| `num_neighbors` | `32` | Graph connectivity (`M`). Higher = better recall, larger index. |
| `ef_construction` | `64` | Build quality. Higher = better recall at index time, slower build. |
| `ef_search` | `16` | Query recall. Higher = better recall at query time, slower query. |

```python
HNSWIndexConfig(
    num_neighbors=32,
    ef_construction=64,
    ef_search=16,
)
```

Use HNSW when the corpus is large and query latency matters.

!!! info "Backend differences"
    - **FAISS**: the HNSW index is stored as a `.faiss` file alongside the HDF5 embedding data. It is built from the stored data the first time it is needed, or via `backend.initialise_indexes(...)`.
    - **pgvector**: the HNSW index is a SQL `CREATE INDEX USING hnsw` object. Call `backend.initialise_indexes(...)` after inserting data to create it; without this, pgvector falls back to a sequential scan automatically. `ef_search` is applied per session via `SET hnsw.ef_search = <value>` at query time.

!!! warning "pgvector dimension limit"
    The pgvector `vector` column type supports **at most 2,000 dimensions**. Registering a model with more than 2,000 dimensions raises `ValueError` immediately. Use `halfvec` (up to 4,000 dims) or `bit` (up to 64,000 dims) column types for larger embeddings — these are planned for a future release.

---

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
