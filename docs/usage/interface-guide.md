# Interface Guide

`omop-emb` exposes two complementary Python interfaces:

- **`EmbeddingWriterInterface`** — write + read. Wraps an `EmbeddingClient` for embedding generation, model registration, and upsert.
- **`EmbeddingReaderInterface`** — read-only. No `EmbeddingClient` needed; nearest-neighbour queries and registry lookups only.

Both interfaces accept a **pre-constructed** `EmbeddingBackend` (sqlite-vec or pgvector) and validate model names via the configured provider.

---

## Constructing a backend

Resolve the active backend from environment variables using `resolve_backend`:

```python
from omop_emb.backends import resolve_backend

backend = resolve_backend()  # reads OMOP_EMB_BACKEND + connection variables
```

Or construct one directly:

```python
from omop_emb.backends.sqlitevec import SQLiteVecEmbeddingBackend
from omop_emb.backends.pgvector import PGVectorEmbeddingBackend

# sqlite-vec
backend = SQLiteVecEmbeddingBackend.from_path(db_path="/data/omop_emb.db")

# pgvector
backend = PGVectorEmbeddingBackend.from_db_url(db_url="postgresql+psycopg://user:pass@host:5432/db")
```

---

## EmbeddingWriterInterface

### Creating the interface

```python
from omop_emb import EmbeddingWriterInterface, EmbeddingClient
from omop_emb.config import MetricType, ProviderType

embedding_client = EmbeddingClient(
    model="nomic-embed-text:v1.5",
    api_base="http://localhost:11434/v1",
    provider_type=ProviderType.OLLAMA,
)

writer = EmbeddingWriterInterface(
    backend=backend,
    metric_type=MetricType.COSINE,
    embedding_client=embedding_client,
    omop_cdm_engine=cdm_engine,  # optional; used to enrich search results
)
```

### Register and initialise

```python
from omop_emb.backends.index_config import FlatIndexConfig

# Always register with FLAT first
writer.register_model()                              # uses FlatIndexConfig() by default
writer.register_model(index_config=FlatIndexConfig())  # explicit equivalent
```

`register_model` is idempotent — calling it when the model is already registered
is safe and returns the existing record.

### Generate and store embeddings

```python
# Fetch candidate concepts from the CDM, then pass the returned rows back as
# concept_meta so filter columns can be stored alongside the embeddings.
missing = writer.get_concepts_without_embedding(
    omop_cdm_engine=cdm_engine,
)

writer.embed_and_upsert_concepts(
    concept_ids=tuple(missing.keys()),
    concept_texts=tuple(row.concept_name for row in missing.values()),
    concept_meta=missing,
)
```

!!! info "Asymmetric embedding models"
    `embed_and_upsert_concepts` always applies the **document** role, and
    `get_nearest_concepts_from_query_texts` always applies the **query** role.
    When calling `embed_texts` directly you must pass `embedding_role` explicitly.
    See [Asymmetric Embeddings](asymmetric-embeddings.md) for task prefix configuration.

### Build an HNSW index

After all embeddings are ingested, optionally upgrade to an approximate index:

```python
from omop_emb.backends.index_config import HNSWIndexConfig
from omop_emb.config import MetricType

writer.rebuild_index(
    index_config=HNSWIndexConfig(
        metric_type=MetricType.COSINE,
        num_neighbors=16,
        ef_construction=64,
        ef_search=16,
    )
)
```

This is equivalent to running `omop-emb maintenance rebuild-index --index-type hnsw` from the CLI.

---

## EmbeddingReaderInterface

Use this when you only need to query stored embeddings — no embedding generation,
no `EmbeddingClient` required.

```python
from omop_emb import EmbeddingReaderInterface
from omop_emb.config import MetricType, ProviderType

reader = EmbeddingReaderInterface(
    model="nomic-embed-text:v1.5",
    backend=backend,
    metric_type=MetricType.COSINE,
    provider_name_or_type=ProviderType.OLLAMA,
    omop_cdm_engine=cdm_engine,   # optional; enriches results with concept_name
)
```

### Query nearest concepts

```python
import numpy as np
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter

query_vec = np.array([[...]], dtype=np.float32)   # shape (Q, D)

results = reader.get_nearest_concepts(
    query_embedding=query_vec,
    k=10,
    concept_filter=EmbeddingConceptFilter(
        require_standard=True,
        domains=("Condition", "Drug"),
        require_active=True,
    ),
)
# results: tuple[tuple[NearestConceptMatch, ...], ...] — one inner tuple per query row
```

### Query by text

```python
from omop_emb import EmbeddingClient
from omop_emb.config import ProviderType

embedding_client = EmbeddingClient(
    model="nomic-embed-text:v1.5",
    api_base="http://localhost:11434/v1",
    provider_type=ProviderType.OLLAMA,
)

results = reader.get_nearest_concepts_from_query_texts(
    query_texts=("high blood pressure", "type 2 diabetes"),
    embedding_client=embedding_client,
    k=5,
)
```

### FAISS fast path

Supply `faiss_cache_dir` to route searches through a pre-built FAISS index
instead of the primary backend SQL path.  The cache must have been built
first with `omop-emb maintenance build-faiss-cache` (builds the FAISS index
directly from the backend).  Requires `omop-emb[faiss-cpu]`.

```python
reader = EmbeddingReaderInterface(
    model="nomic-embed-text:v1.5",
    backend=backend,
    metric_type=MetricType.COSINE,
    provider_name_or_type=ProviderType.OLLAMA,
    faiss_cache_dir="/data/faiss_cache",
)
# Searches automatically use FAISS when the cache is fresh; SQL path otherwise.
```

The environment variable `OMOP_EMB_FAISS_CACHE_DIR` is checked as a fallback
when `faiss_cache_dir` is not passed directly.

---

## EmbeddingConceptFilter

`EmbeddingConceptFilter` is an in-database pre-filter applied during KNN search.
All filtering happens before the nearest-neighbour step — only matching concepts
are candidates.

```python
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter

concept_filter = EmbeddingConceptFilter(
    domains=("Condition", "Observation"),   # restrict to specific OMOP domains
    vocabularies=("SNOMED", "ICD10CM"),     # restrict to specific vocabularies
    concept_ids=(313217, 4329847),          # restrict to specific concept IDs
    require_standard=True,                  # standard_concept = 'S' or 'C'
    require_active=True,                    # invalid_reason NOT IN ('D', 'U')
    limit=20,                               # cap on results returned
)
```

All fields are optional and combinable. `require_standard` and `require_active`
are stored as columns in the embedding table and are resolved entirely inside the
primary backend — no CDM round-trip at query time.

---

## EmbeddingClient and providers

!!! note
    Currently, only `OllamaProvider` is supported.

`EmbeddingClient` wraps any OpenAI-compatible endpoint. It canonicalises the
model name at construction time and exposes `canonical_model_name` as the stable
identifier used in the registry.

```python
from omop_emb import EmbeddingClient
from omop_emb.config import ProviderType

# Ollama — provider specified explicitly (works with any hostname or IP)
client = EmbeddingClient(
    model="nomic-embed-text:v1.5",
    api_base="http://host.docker.internal:11434/v1",
    provider_type=ProviderType.OLLAMA,
)

print(client.canonical_model_name)  # "nomic-embed-text:v1.5"
print(client.embedding_dim)         # auto-discovered via Ollama /api/show
```

---

## Model name validation

### Valid names

**Ollama:**

- ✅ `nomic-embed-text:v1.5`
- ✅ `llama3:8b`
- Any name with an explicit, immutable tag

### Invalid names (raise `ValueError`)

**Ollama:**

- ❌ `llama3` — "must include an explicit tag"
- ❌ `llama3:latest` — "uses the mutable ':latest' tag"

!!! info
    **Why the strictness?** In long-term healthcare data storage, `:latest` is a
    moving target. Running `ollama pull llama3` silently changes which model
    version `:latest` points to, breaking consistency between stored embeddings
    and new query embeddings.

---

## Utility functions

```python
from omop_emb import list_registered_models
from omop_emb.config import ProviderType

models = list_registered_models(
    backend=backend,
    provider_type=ProviderType.OLLAMA,  # optional filter
)
for m in models:
    print(m.model_name, m.provider_type, m.dimensions, m.index_type)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                Your Application Code                │
└──────────────┬──────────────────────────────────────┘
               │
        ┌──────┴──────────────────┐
        │                         │
        ▼                         ▼
┌───────────────────┐   ┌──────────────────────┐
│ EmbeddingWriter   │   │  EmbeddingReader      │
│ Interface         │   │  Interface            │
│ (write + read)    │   │  (read only)          │
└───────┬───────────┘   └──────────┬───────────┘
        │                          │
        ▼                          │
┌───────────────────┐              │
│  EmbeddingClient  │              │
│  + Provider       │              │
└───────────────────┘              │
        │                          │
        └──────────┬───────────────┘
                   │
          ┌────────┴────────┐
          │    Backend      │
          │ sqlite-vec      │
          │ pgvector        │
          └────────┬────────┘
                   │ (optional fast path)
          ┌────────┴────────┐
          │  FAISS sidecar  │
          │  (read-only)    │
          └─────────────────┘
```

`EmbeddingWriterInterface` inherits from `EmbeddingReaderInterface` — all reader
methods are available on the writer too.

---

## Best practices

1. **Use the interfaces**, not backends directly — they enforce canonical naming.
2. **`EmbeddingWriterInterface` for write flows**, `EmbeddingReaderInterface` for query-only services.
3. **Use `embedding_client.canonical_model_name`** when constructing a matching reader — it is guaranteed to be canonical.
4. **Always register with `FlatIndexConfig`** first. Run `rebuild_index` or `omop-emb maintenance rebuild-index` after ingestion to build HNSW.
5. **CDM enrichment is optional** — omit `omop_cdm_engine` when `concept_name` is not needed to avoid the CDM round-trip.
6. **FAISS is a read-acceleration sidecar, never the source of truth** — build it directly from the backend with `omop-emb maintenance build-faiss-cache` and supply `faiss_cache_dir` to `EmbeddingReaderInterface` for faster approximate search.
