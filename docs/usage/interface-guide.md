# EmbeddingInterface Guide

`omop-emb` exposes two complementary interfaces:

- **`EmbeddingWriterInterface`** — write + read. Requires an `EmbeddingClient` (embedding generation, model registration, upsert).
- **`EmbeddingReaderInterface`** — read-only. No `EmbeddingClient` needed (nearest-neighbor queries, registry lookups).

Both interfaces validate model names via the provider on construction.

---

## EmbeddingWriterInterface

### Creating the interface

```python
from omop_emb import EmbeddingWriterInterface, EmbeddingClient

embedding_client = EmbeddingClient(
    model="nomic-embed-text:v1.5",   # validated by OllamaProvider
    api_base="http://localhost:11434/v1",
)

interface = EmbeddingWriterInterface(
    embedding_client=embedding_client,
    backend_name_or_type="faiss",    # or "pgvector", or BackendType.FAISS
)
```

`backend_name_or_type` falls back to the `OMOP_EMB_BACKEND` environment variable when omitted.

### Register and initialise

```python
from omop_emb.backends import FlatIndexConfig, HNSWIndexConfig

# FLAT index — sequential scan, no warm-up needed
interface.register_model(engine=db_engine, index_config=FlatIndexConfig())
interface.initialise_store(db_engine)

# HNSW index — approximate nearest-neighbour, configurable
interface.register_model(
    engine=db_engine,
    index_config=HNSWIndexConfig(
        num_neighbors=32,
        ef_construction=64,
        ef_search=16,
    ),
)
interface.initialise_store(db_engine)
```

`register_model` persists the model and its index configuration in the local registry. `initialise_store` loads all registered models into memory.

!!! info "pgvector HNSW: explicit index creation"
    For the pgvector backend with HNSW, the SQL index must be created separately after embeddings are inserted:

    ```python
    from omop_emb import MetricType, IndexType

    interface._backend.initialise_indexes(
        model_name=interface.canonical_model_name,
        provider_type=interface.provider_type,
        index_type=IndexType.HNSW,
        metric_types=[MetricType.L2],
    )
    ```

    Without this call, pgvector falls back to a sequential scan. This step is idempotent — calling it when the index already exists is safe. See [Index Types](backend-selection.md#index-types) for details.

### Generate and store embeddings

```python
# Generate embeddings for texts and upsert in one step
interface.embed_and_upsert_concepts(
    session=session,
    index_type=IndexType.FLAT,
    concept_ids=(1, 2, 3),
    concept_texts=("Hypertension", "Diabetes", "Aspirin"),
)

# Or generate and upsert separately
from omop_emb.embeddings import EmbeddingRole

embeddings = interface.embed_texts(
    ["Hypertension", "Diabetes"],
    embedding_role=EmbeddingRole.DOCUMENT,
)
interface.upsert_concept_embeddings(
    session=session,
    index_type=IndexType.FLAT,
    concept_ids=(1, 2),
    embeddings=embeddings,
)
```

!!! info "Asymmetric embedding models"
    `embed_and_upsert_concepts` always applies the **document** role, and `get_nearest_concepts_from_query_texts` always applies the **query** role. When calling `embed_texts` directly you must pass `embedding_role` explicitly. See [Asymmetric Embeddings](asymmetric-embeddings.md) for how to configure task prefixes.

### Query nearest concepts

```python
from omop_emb import MetricType

# Query by pre-computed embedding
results = interface.get_nearest_concepts(
    session=session,
    index_type=IndexType.FLAT,
    query_embedding=query_vec,   # shape (q, D)
    metric_type=MetricType.COSINE,
)

# Query by text (embeds automatically)
results = interface.get_nearest_concepts_by_texts(
    session=session,
    index_type=IndexType.FLAT,
    query_texts=("high blood pressure",),
    metric_type=MetricType.COSINE,
)
# results: tuple of {concept_id: similarity_score} dicts, one per query
```

---

## EmbeddingReaderInterface

Use this when you only need to query stored embeddings — no embedding generation, no `EmbeddingClient` required.

```python
from omop_emb import EmbeddingReaderInterface, ProviderType, IndexType, MetricType

reader = EmbeddingReaderInterface(
    canonical_model_name="nomic-embed-text:v1.5",
    provider_name_or_type=ProviderType.OLLAMA,   # or "ollama"
    backend_name_or_type="faiss",
)
reader.initialise_store(db_engine)

results = reader.get_nearest_concepts(
    session=session,
    index_type=IndexType.FLAT,
    query_embedding=query_vec,
    metric_type=MetricType.COSINE,
)
```

The constructor validates `canonical_model_name` against the provider rules. For Ollama, an untagged name or `:latest` will raise `ValueError` on construction.

---

## EmbeddingClient and providers

`EmbeddingClient` wraps any OpenAI-compatible endpoint. It canonicalises the model name at construction time and exposes `canonical_model_name` as the stable identifier used in the registry.

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
    **Why the strictness?** In long-term healthcare data storage, `:latest` is a moving target. Running `ollama pull llama3` silently changes which model version `:latest` points to, breaking consistency between stored embeddings and new query embeddings.

---

## Utility functions

```python
from omop_emb import list_registered_models

models = list_registered_models(
    backend_name_or_type="faiss",
    provider_type=ProviderType.OLLAMA,
)
for m in models:
    print(m.model_name, m.provider_type, m.dimensions)
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
                   ▼
          ┌─────────────────┐
          │    Backend      │
          │ (pgvector/faiss)│
          └─────────────────┘
```

`EmbeddingWriterInterface` inherits from `EmbeddingReaderInterface` — all reader methods are available on the writer too.

---

## Best practices

1. **Use the interfaces**, not backends directly — they enforce canonical naming.
2. **`EmbeddingWriterInterface` for write flows**, `EmbeddingReaderInterface` for query-only services.
3. **Use `embedding_client.canonical_model_name`** when constructing a matching reader — it is guaranteed to be canonical.
4. **Provide `storage_base_dir`** explicitly in production to control where `metadata.db` and FAISS files land.
