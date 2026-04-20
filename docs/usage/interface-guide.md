# EmbeddingInterface Guide

The `EmbeddingInterface` is the primary API for managing concept embeddings in `omop-emb`. It provides a backend-neutral interface for registering models, generating embeddings, and querying stored vectors.

## Overview

`EmbeddingInterface` is designed to be **the** interface for all embedding operations. It enforces **model name canonicalization** and **validation** to ensure data consistency:

- Model names must be **canonical** (validated by embedding providers)
- Ollama models require explicit, immutable tags (e.g., `llama3:8b`, not `llama3` or `llama3:latest`)
- OpenAI-compatible models use names verbatim (e.g., `text-embedding-3-small`)

## Creating an Interface

### With an EmbeddingClient (automatic validation)

```python
from omop_emb import EmbeddingInterface, EmbeddingClient

# Create an embedding client — this canonicalizes the model name automatically
embedding_client = EmbeddingClient(
    model="llama3:8b",  # Will be validated by OllamaProvider
    api_base="http://localhost:11434/v1",
)

# Create interface with the client — validation happens automatically
interface = EmbeddingInterface.from_backend_name(
    embedding_client=embedding_client,
    backend_name="faiss",  # or "pgvector"
)

# The interface now has access to the provider for validation
canonical_model = embedding_client.model  # e.g., "llama3:8b"
```

### Without an EmbeddingClient (for query-only scenarios)

```python
from omop_emb import EmbeddingInterface, OllamaProvider

# For queries without embedding generation, pass provider explicitly
provider = OllamaProvider()

interface = EmbeddingInterface.from_backend_name(
    backend_name="faiss",
    provider=provider,  # Enable validation
)

# Now all operations validate model names
nearest = interface.get_nearest_concepts(
    session=session,
    canonical_model_name="llama3:8b",  # Will be validated
    index_type=IndexType.FLAT,
    query_embedding=query_vec,
    metric_type=MetricType.COSINE,
)
```

### Without Validation (trusting the caller)

```python
# If neither embedding_client nor provider is passed,
# validation is skipped (trust the caller):
interface = EmbeddingInterface(backend=my_backend)

# Validation will not happen; you must ensure model names are canonical
```

## Model Name Validation

When an `EmbeddingInterface` has a provider (from `embedding_client` or explicit `provider` parameter), all methods that accept a `canonical_model_name` will validate it.

### Valid Names

**Ollama:**
- ✅ `llama3:8b`
- ✅ `nomic-embed-text:v1.5`
- ✅ Any name with explicit, immutable tag

**OpenAI-compatible:**
- ✅ `text-embedding-3-small`
- ✅ `text-embedding-3-large`

### Invalid Names (will raise ValueError)

**Ollama:**
- ❌ `llama3` → "must include an explicit tag"
- ❌ `llama3:latest` → "uses the mutable ':latest' tag"

**Why the strictness?**

In healthcare contexts where embeddings are stored long-term:
- Untagged names like `llama3` default to `:latest`, which is mutable
- Running `ollama pull llama3` could silently change which model version `:latest` points to
- This breaks consistency: stored embeddings from model v1 don't match new queries from model v2

## Usage Examples

### Register and Embed Concepts

```python
from omop_emb import EmbeddingInterface, EmbeddingClient, IndexType

embedding_client = EmbeddingClient(
    model="llama3:8b",
    api_base="http://localhost:11434/v1",
)

interface = EmbeddingInterface.from_backend_name(
    embedding_client=embedding_client,
    backend_name="pgvector",
)

# Register model — validates name automatically
interface.setup_and_register_model(
    engine=db_engine,
    canonical_model_name=embedding_client.model,  # "llama3:8b"
    dimensions=embedding_client.embedding_dim,
    index_type=IndexType.FLAT,
)

# Generate and store embeddings
interface.embed_and_upsert_concepts(
    session=session,
    canonical_model_name=embedding_client.model,  # Validated
    index_type=IndexType.FLAT,
    concept_ids=(1, 2, 3),
    concept_texts=("Hypertension", "Diabetes", "Aspirin"),
)
```

### Query Nearest Concepts

```python
# Query with existing embeddings (no generation needed)
nearest = interface.get_nearest_concepts(
    session=session,
    canonical_model_name="llama3:8b",  # Validated against provider
    index_type=IndexType.FLAT,
    query_embedding=query_vec,
    metric_type=MetricType.COSINE,
)

# Or query with text (convenience wrapper)
nearest = interface.get_nearest_concepts_by_texts(
    session=session,
    canonical_model_name="llama3:8b",  # Validated
    index_type=IndexType.FLAT,
    query_texts=("high blood pressure", "low glucose"),
    metric_type=MetricType.COSINE,
)
```

### Check Model Registration

```python
# Check if a model is already registered
if interface.is_model_registered(
    canonical_model_name="llama3:8b",  # Validated
    index_type=IndexType.FLAT,
):
    print("Model is registered")
```

## Error Messages

When validation fails, you get a clear, actionable error:

```python
# ❌ This fails with a clear message
interface.get_nearest_concepts(
    session=session,
    canonical_model_name="llama3",  # Untagged!
    index_type=IndexType.FLAT,
    query_embedding=query_vec,
    metric_type=MetricType.COSINE,
)
# ValueError: Invalid canonical_model_name 'llama3': Ollama model name
# 'llama3' must include an explicit tag. Use a specific version 
# (e.g. 'llama3:8b') instead of relying on the mutable ':latest' pointer.
```

## Architecture

```
┌─────────────────────────────────────────────┐
│         Your Application Code               │
└─────────────┬───────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────┐
│      EmbeddingInterface (API boundary)      │
│  - Validates model names                    │
│  - Routes to backend                        │
│  - Enforces canonical naming                │
└────────┬────────────────────┬───────────────┘
         │                    │
         ↓                    ↓
    ┌─────────┐        ┌─────────────┐
    │Embedding│        │  Backend    │
    │ Client  │        │(pgvector/   │
    │+Provider│        │ faiss)      │
    └─────────┘        └─────────────┘
```

The interface is the **validation and enforcement layer**. All embedding operations should go through it.

## Best Practices

1. **Always use `EmbeddingInterface`** — Don't interact with backends directly
2. **Provide a provider** — Either via `embedding_client` or explicit `provider` parameter
3. **Use `embedding_client.model`** — This is guaranteed to be canonical
4. **Validate early** — If validation fails, you'll know immediately with a clear error
5. **Trust the validation** — Once a name passes validation, it's canonical and safe to store

## Migration Guide

If you were previously calling backends directly:

```python
# ❌ Old approach (no validation)
backend.register_model(model_name=raw_input, ...)

# ✅ New approach (validated)
interface = EmbeddingInterface(backend=backend, provider=provider)
interface.register_model(canonical_model_name=raw_input, ...)
# Will validate or raise ValueError
```
