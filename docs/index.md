# OMOP Embeddings

`omop-emb` extends the OMOP tooling stack with embedding storage, retrieval, and CLI workflows for concept search.

The package supports:

- local model registration metadata in SQLite (`metadata.db`)
- explicit backend selection via `pgvector` or `faiss`
- concept embedding generation and nearest-neighbor lookup
- CLI workflows for embedding, search, export/import, and FAISS index maintenance

## Installation

```bash
pip install "omop-emb[pgvector]"
pip install "omop-emb[faiss]"
pip install "omop-emb[all]"
```

Runtime environment typically includes:

- `OMOP_EMB_BACKEND`
- `OMOP_EMB_BASE_STORAGE_DIR`
- `OMOP_DATABASE_URL`

## Documentation overview

- [Installation](usage/installation.md)
- [**EmbeddingInterface Guide**](usage/interface-guide.md) — The primary API for embedding operations with model name validation
- [Embedding storage backends](usage/backend-selection.md)
- [CLI reference](usage/cli.md)
