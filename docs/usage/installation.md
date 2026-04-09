# Installation

`omop-emb` supports backend-specific optional dependencies so users can install
only what they need.

## Core only

```bash
pip install omop-emb
```

This installs the shared core package only. It does not guarantee that any
particular embedding backend is available.

It also does not remove the need for a database backend. `omop-emb` still
requires database-backed OMOP access and model registration.

## PostgreSQL backend

```bash
pip install "omop-emb[pgvector]"
```

Use this when you want the current pgvector/PostgreSQL-backed embedding store
and CLI flow.

## FAISS backend

```bash
pip install "omop-emb[faiss]"
```

Use this when you want the FAISS backend dependencies available.

Even in this case, a database backend is still required for OMOP concept
metadata and model registration.

## Everything

```bash
pip install "omop-emb[all]"
```

This is the most convenient choice for development, testing, and mixed
environments where you want both backend stacks available.

## Recommended runtime pattern

For clarity, backend selection should be explicit at runtime as well as
install-time.

Examples:

```bash
export OMOP_EMB_BACKEND=pgvector
export OMOP_EMB_BACKEND=faiss
```

That avoids silent fallback between backend implementations.

## Current database support caveat

PostgreSQL-specific embedding dependencies are now optional, but the broader
system has not yet been tested against non-PostgreSQL database backends.

So the current position is:

- PostgreSQL embedding infrastructure is optional
- a database backend is still always required
- database backends other than PostgreSQL should currently be treated as
  unverified
