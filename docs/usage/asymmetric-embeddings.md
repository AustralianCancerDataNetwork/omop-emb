# Asymmetric Embeddings { data-toc-label="Asymmetric Embeddings" }

Asymmetric embedding models (nomic-embed-text, the E5 family, BGE, and others) need a different text prefix depending on whether the text is being indexed or searched. `EmbeddingRole`, prefix application, and prefix configuration are all owned by `omop_llm`/`oa-configurator`, not `omop-emb` — see:

- [omop-llm: Asymmetric Embeddings](https://AustralianCancerDataNetwork.github.io/omop-llm/usage/asymmetric-embeddings/) for what asymmetric models are, why prefixing matters, and how `EmbeddingRole`/prefix application work.
- [oa-configurator: `[models.<name>]`](https://AustralianCancerDataNetwork.github.io/OA_Configurator/config-reference/#modelsname) for the `document_prefix`/`query_prefix` config schema and `omop-config models add`/`list`.

`omop-emb` re-exports `EmbeddingRole` for convenience (`from omop_emb import EmbeddingRole` is equivalent to `from omop_llm import EmbeddingRole`) and only forwards `role=` through to `omop_llm.ModelBackend.embed_texts`, which applies the prefix internally — `omop-emb` never sees the raw prefix strings itself. `omop-emb`'s own config (`OmopEmbConfig.embedding_model_name`) only names *which* `[models.*]` entry to use.

## Configuring the model `omop-emb` uses

Two steps, via `oa-configurator`'s CLI (see its [Quickstart](https://AustralianCancerDataNetwork.github.io/OA_Configurator/quickstart/#3-configure-an-llmembedding-model-optional) for the full walkthrough):

```bash
omop-config providers add local-ollama --provider ollama --base-url http://localhost:11434
omop-config models add nomic-embed \
    --provider local-ollama \
    --model nomic-embed-text:v1.5 \
    --embedding-dim 768 \
    --document-prefix "search_document: " \
    --query-prefix "search_query: "
```

Then point `omop-emb` at it, either via `omop-config configure omop_emb` (prompts for `embedding_model_name` among its other settings) or by hand-editing `[tools.omop_emb.extra]`:

```toml
[tools.omop_emb.extra]
embedding_model_name = "nomic-embed"
```

`EmbeddingWriterInterface`/`cli_embeddings.py` resolve this name via `oa_configurator.Resolver.resolve_model(...)` at construction time — nothing about the provider, connection, dimension, or prefixes needs to be repeated in `omop-emb`'s own config.

## The two roles, in OMOP terms

| Role | Example texts | Purpose |
|------|--------------|---------|
| **Document** | `"Hypertension"`, `"Type 2 diabetes mellitus"` | Concepts stored in the vector index |
| **Query** | `"high blood pressure"`, `"T2DM"` | Free-text search terms at query time |

For a symmetric model these are interchangeable. For an asymmetric model, mixing the two reduces retrieval quality without any visible error — the document prefix must be applied to every concept at index time, and the query prefix to every search term at query time.

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
    role=EmbeddingRole.DOCUMENT,
)

# Searching: use QUERY
query_embeddings = interface.embed_texts(
    ["high blood pressure"],
    role=EmbeddingRole.QUERY,
)
```

The same applies to `EmbeddingReaderInterface.generate_embeddings()`, the lower-level entry point used when you hold an `omop_llm.ModelBackend` directly without a full writer interface (e.g. on-the-fly query embedding):

```python
from omop_emb import EmbeddingReaderInterface

vecs = EmbeddingReaderInterface.generate_embeddings(
    model_backend, texts, role=EmbeddingRole.DOCUMENT
)
```
