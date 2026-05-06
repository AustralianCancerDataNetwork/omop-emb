# Installation

`omop-emb` supports backend-specific optional dependencies so you can install
only what you need.

## sqlite-vec (default, no extras required)

```bash
pip install omop-emb
```

The default backend is sqlite-vec — a file-based or in-memory vector store that
requires no external database server. This is ready to use immediately after
install.

## pgvector backend

```bash
pip install "omop-emb[pgvector]"
```

Adds `psycopg` and the `pgvector` SQLAlchemy type adapter. Requires a running
PostgreSQL instance with the pgvector extension installed (e.g.
[`pgvector/pgvector`](https://hub.docker.com/r/pgvector/pgvector) Docker image).

## FAISS sidecar

```bash
pip install "omop-emb[faiss]"
```

Adds `faiss-cpu` and `h5py`. FAISS is a read-acceleration sidecar that layers
on top of any primary backend — it does not replace sqlite-vec or pgvector.

## Everything

```bash
pip install "omop-emb[pgvector,faiss]"
```

Installs all optional dependencies. Recommended for development and mixed
environments.

---

## Configuring the backend at runtime

Set `OMOP_EMB_BACKEND` to select the backend (default: `sqlitevec`):

```bash
export OMOP_EMB_BACKEND=sqlitevec   # default — no external service needed
export OMOP_EMB_BACKEND=pgvector    # requires PostgreSQL + pgvector
```

### sqlite-vec connection

Point to a database file (or use `:memory:` for a transient in-memory store):

```bash
export OMOP_EMB_SQLITE_PATH=/data/omop_emb.db
# or, for a transient in-memory database:
export OMOP_EMB_SQLITE_PATH=:memory:
```

### pgvector connection

Supply individual connection components (recommended, matches `.env` and
Docker Compose patterns):

```bash
export OMOP_EMB_DB_HOST=localhost
export OMOP_EMB_DB_PORT=5432
export OMOP_EMB_DB_USER=omop_emb
export OMOP_EMB_DB_PASSWORD=omop_emb
export OMOP_EMB_DB_NAME=omop_emb
```

Or supply a full SQLAlchemy URL (overrides all individual components):

```bash
export OMOP_EMB_DB_URL=postgresql+psycopg://omop_emb:omop_emb@localhost:5432/omop_emb
```

The default driver string is `postgresql+psycopg` (psycopg3). Override via
`OMOP_EMB_DB_CONN` if you need a different driver (e.g. `postgresql+psycopg2`).

### Docker Compose

A reference `docker-compose.yaml` for the pgvector service ships with the
package. Create a `.env` file alongside it:

```bash
OMOP_EMB_DB_USER=omop_emb
OMOP_EMB_DB_PASSWORD=omop_emb
OMOP_EMB_DB_NAME=omop_emb
OMOP_EMB_DB_HOST=omop-emb-db
OMOP_EMB_DB_PORT=5432
```

Then start the service:

```bash
docker compose up -d omop-emb-db
```
