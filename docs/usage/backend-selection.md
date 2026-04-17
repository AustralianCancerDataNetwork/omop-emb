# Backend Selection

`omop-emb` uses an explicit backend abstraction for embedding storage and retrieval.

## Supported backends

- `pgvector`
- `faiss`

There is no implicit default. Pass the backend explicitly or set `OMOP_EMB_BACKEND`.

## Runtime selection

```bash
export OMOP_EMB_BACKEND=pgvector
export OMOP_EMB_BACKEND=faiss
export OMOP_EMB_BASE_STORAGE_DIR=$PWD/.omop_emb
```

## Python factory

```python
from omop_emb.backends import get_embedding_backend

backend = get_embedding_backend("pgvector")
backend = get_embedding_backend("faiss")
```

## Current scope

- the CLI supports both `pgvector` and `faiss`
- OMOP database access is still required for concept metadata and filtering
- model registration metadata is stored locally in SQLite under `OMOP_EMB_BASE_STORAGE_DIR`
- database backends other than PostgreSQL are not yet tested
