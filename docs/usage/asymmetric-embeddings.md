# Asymmetric Embeddings { data-toc-label="Asymmetric Embeddings" }

## What are asymmetric embedding models?

Most general-purpose models (e.g. `text-embedding-3-small`) produce vectors in a symmetric space: the same transformation is applied whether you are indexing a document or submitting a search query.

**Asymmetric models**, such as [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), [E5](https://huggingface.co/intfloat/e5-large-v2), and [BGE](https://huggingface.co/BAAI/bge-large-en-v1.5), are trained with *task-specific prefixes* prepended to the input. The model's training objective explicitly separates the representation space for documents being indexed from the space for queries being searched. Sending text without the correct prefix does not raise an error, but similarity scores degrade substantially and silently.

## Why it matters for OMOP concept search

In `omop-emb` there are two distinct embedding roles:

| Role | Example texts | Purpose |
|------|--------------|---------|
| **Document** | `"Hypertension"`, `"Type 2 diabetes mellitus"` | Concepts stored in the vector index |
| **Query** | `"high blood pressure"`, `"T2DM"` | Free-text search terms at query time |

For a symmetric model these are interchangeable. For an asymmetric model the document prefix must be applied to every concept at index time, and the query prefix to every search term at query time. Mixing the two reduces retrieval quality without any visible error.

## Configuration

`omop-emb` reads two environment variables:

| Variable | Role | Example value |
|---|---|---|
| `OMOP_EMB_DOCUMENT_EMBEDDING_PREFIX` | Prepended to all **document** texts before indexing | `search_document: ` |
| `OMOP_EMB_QUERY_EMBEDDING_PREFIX` | Prepended to all **query** texts before searching | `search_query: ` |

Both default to `""`. `EmbeddingWriterInterface` reads them once, at construction, and caches the result for the lifetime of the instance. `EmbeddingReaderInterface` has no equivalent cache, as it has no notion of a model (by design) at all outside of `get_nearest_concepts_from_query_texts`, which reads them fresh on every call unless you pass `prefixes` explicitly. Either way, when a prefix is empty, `omop-emb` logs a warning explaining what the variable is for.

!!! warning "A caller-supplied `prefixes` mapping must cover both roles"
    `generate_embeddings()`/`embed_texts()` accept an optional `prefixes` override. If you pass one, it must include **both** `EmbeddingRole.DOCUMENT` and `EmbeddingRole.QUERY`. A partial mapping raises `ValueError` rather than silently treating the missing role as unprefixed. Omit `prefixes` entirely to load both roles from config instead of overriding either.

!!! tip "Prefix examples by model family"
    | Model | Document prefix | Query prefix |
    |---|---|---|
    | `nomic-embed-text:v1.5` | `search_document: ` | `search_query: ` |
    | `e5-large-v2` | `passage: ` | `query: ` |
    | `bge-large-en-v1.5` | *(none)* | `Represent this sentence for searching relevant passages: ` |

    Always check the model card: task prefixes are model-specific and can change between versions.

## Role assignment in the API

The two high-level methods handle roles automatically. You only need to think about roles when calling `embed_texts` directly.

### Indexing concepts: `DOCUMENT` is automatic

`embed_and_upsert_concepts` always uses `EmbeddingRole.DOCUMENT`:

```python
interface.embed_and_upsert_concepts(
    index_type=IndexType.FLAT,
    concept_ids=(1, 2, 3),
    concept_texts=("Hypertension", "Diabetes", "Aspirin"),
)
```

### Querying: `QUERY` is automatic

`get_nearest_concepts_from_query_texts` always uses `EmbeddingRole.QUERY`:

```python
results = interface.get_nearest_concepts_from_query_texts(
    index_type=IndexType.FLAT,
    query_texts=("high blood pressure",),
    metric_type=MetricType.COSINE,
)
```

### Direct embedding generation: caller chooses the role

When you call `embed_texts` directly you must pass the role explicitly:

```python
from omop_emb import EmbeddingRole

# Indexing: use DOCUMENT
doc_embeddings = interface.embed_texts(
    ["Hypertension", "Diabetes"],
    embedding_role=EmbeddingRole.DOCUMENT,
)

# Searching: use QUERY
query_embeddings = interface.embed_texts(
    ["high blood pressure"],
    embedding_role=EmbeddingRole.QUERY,
)
```

The same applies to `EmbeddingReaderInterface.generate_embeddings()`, the lower-level entry point used when you hold an `omop_llm.ModelBackend` directly without a full writer interface (e.g. on-the-fly query embedding):

```python
from omop_emb import EmbeddingReaderInterface

vecs = EmbeddingReaderInterface.generate_embeddings(
    model_backend, texts, embedding_role=EmbeddingRole.DOCUMENT
)
```

### Inspecting configured prefixes

```python
print(EmbeddingReaderInterface.load_embedding_prefixes())
# {<EmbeddingRole.DOCUMENT: 'document'>: 'search_document: ',
#  <EmbeddingRole.QUERY: 'query'>: 'search_query: '}
```
