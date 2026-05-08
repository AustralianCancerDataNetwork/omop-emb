# Interface Guide

`omop-emb` exposes two complementary Python interfaces:

- **`EmbeddingWriterInterface`** — write + read. Wraps an `EmbeddingClient` for embedding generation, model registration, and upsert.
- **`EmbeddingReaderInterface`** — read-only. No `EmbeddingClient` needed; nearest-neighbour queries and registry lookups only.

Both interfaces accept a **pre-constructed** `EmbeddingBackend` (sqlite-vec or pgvector) and validate model names via the configured provider.

---

## Constructing a backend

Resolve the active backend from environment variables using `resolve_backend`:

```python
from omop_emb.cli.utils import resolve_backend

backend = resolve_backend()  # reads OMOP_EMB_BACKEND + connection variables
```

Or construct one directly:

```python
from omop_emb.backends.sqlitevec import SQLiteVecBackend
from omop_emb.backends.pgvector import PGVectorEmbeddingBackend

# sqlite-vec
backend = SQLiteVecBackend(db_path="/data/omop_emb.db")

# pgvector
backend = PGVectorEmbeddingBackend(db_url="postgresql+psycopg://user:pass@host:5432/db")
```

---

## EmbeddingWriterInterface

### Creating the interface

```python
from omop_emb import EmbeddingWriterInterface, EmbeddingClient
from omop_emb.config import MetricType

embedding_client = EmbeddingClient(
    model="nomic-embed-text:v1.5",
    api_base="http://localhost:11434/v1",
)

writer = EmbeddingWriterInterface(
    backend=backend,
    metric_type=MetricType.COSINE,
    embedding_client=embedding_client,
    omop_cdm_engine=cdm_engine,  # optional; required for embed_and_upsert_concepts
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
# Generate embeddings from CDM concepts and upsert in one step.
# omop_cdm_engine is used to fetch domain_id, vocabulary_id, standard_concept,
# and invalid_reason from the CDM and store them as filter metadata.
writer.embed_and_upsert_concepts(
    omop_cdm_engine=cdm_engine,
    concept_ids=(1, 2, 3),
    concept_texts=("Hypertension", "Diabetes mellitus", "Aspirin"),
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

This is equivalent to running `omop-emb rebuild-index --index-type hnsw` from the CLI.

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

embedding_client = EmbeddingClient(
    model="nomic-embed-text:v1.5",
    api_base="http://localhost:11434/v1",
)

results = reader.get_nearest_concepts_from_query_texts(
    query_texts=("high blood pressure", "type 2 diabetes"),
    embedding_client=embedding_client,
    k=5,
)
```

### FAISS fast path

Supply `faiss_cache_dir` to route searches through a pre-exported FAISS index
instead of the primary backend SQL path.  The cache must have been exported
first with `omop-emb export-faiss-cache`.  Requires `omop-emb[faiss-cpu]`.

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

`EmbeddingClient` wraps any OpenAI-compatible endpoint. It canonicalises the
model name at construction time and exposes `canonical_model_name` as the stable
identifier used in the registry.

```python
from omop_emb import EmbeddingClient, OllamaProvider, OpenAIProvider

# Ollama — provider inferred from URL
client = EmbeddingClient(
    model="nomic-embed-text:v1.5",
    api_base="http://localhost:11434/v1",
)

# OpenAI — provider inferred from URL + API key
client = EmbeddingClient(
    model="text-embedding-3-small",
    api_base="https://api.openai.com/v1",
    api_key="sk-...",
)

# Explicit provider (custom or future backends)
client = EmbeddingClient(
    model="nomic-embed-text:v1.5",
    api_base="http://my-custom-host/v1",
    provider=OllamaProvider(),
)

print(client.canonical_model_name)  # "nomic-embed-text:v1.5"
print(client.embedding_dim)         # auto-discovered via Ollama /api/show
```

Provider inference rules (evaluated in order):

| Condition | Provider |
|-----------|----------|
| `"ollama"` in `api_base` | `OllamaProvider` |
| `localhost` or `127.0.0.1` in `api_base` **and** `api_key == "ollama"` | `OllamaProvider` |
| everything else | `OpenAIProvider` |

Pass `provider=` explicitly to override inference for any custom backend.

---

## Model name validation

### Valid names

**Ollama:**

- ✅ `nomic-embed-text:v1.5`
- ✅ `llama3:8b`
- Any name with an explicit, immutable tag

**OpenAI-compatible:**

- ✅ `text-embedding-3-small`
- ✅ `text-embedding-3-large`

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
4. **Always register with `FlatIndexConfig`** first. Run `rebuild_index` or `omop-emb rebuild-index` after ingestion to build HNSW.
5. **CDM enrichment is optional** — omit `omop_cdm_engine` when `concept_name` is not needed to avoid the CDM round-trip.
6. **FAISS is a read-acceleration sidecar** — export with `omop-emb export-faiss-cache` and supply `faiss_cache_dir` to `EmbeddingReaderInterface` for faster approximate search.
