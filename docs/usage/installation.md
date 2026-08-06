# Installation

`omop-emb` supports backend-specific optional dependencies so you can install only what you need.

## sqlite-vec (default, no extras required)

```bash
pip install omop-emb
```

The default backend is sqlite-vec: a file-based or in-memory vector store that requires no external database server. This is ready to use immediately after install.

## pgvector backend

```bash
pip install "omop-emb[pgvector]"
```

Adds `psycopg` and the `pgvector` SQLAlchemy type adapter. Requires a running PostgreSQL instance with the pgvector extension installed (e.g. [`pgvector/pgvector`](https://hub.docker.com/r/pgvector/pgvector) Docker image).

## FAISS sidecar

```bash
pip install "omop-emb[faiss-cpu]"
```

Adds `faiss-cpu`. FAISS is a read-acceleration sidecar that layers on top of any primary backend; it does not replace sqlite-vec or pgvector. The primary backend remains the source of truth; FAISS indices are exported from it and used for faster in-memory approximate-nearest-neighbour search.

## Everything

```bash
pip install "omop-emb[pgvector,faiss-cpu]"
```

Installs all optional dependencies. Recommended for development and mixed environments.

---

## Configuring the backend

The backend (sqlite-vec or pgvector) is selected by a `[vector_stores.*]` entry's `backend_type`, configured via [oa-configurator](https://AustralianCancerDataNetwork.github.io/oa-configurator/), not environment variables:

```bash
omop-config connections add emb --dialect postgresql+psycopg --host localhost --database-name omop_emb
omop-config databases add emb_db --kind generic --connection emb
omop-config vector-stores add vector_store --backend-type pgvector --database emb_db
omop-config configure omop_emb --vector-store-name vector_store
```

See [Configuration Reference](configuration.md) for the full field list (both backends, plus `faiss_cache_dir`), and [Getting Started: Configuration](../getting-started/configuration.md) for the end-to-end walkthrough.
