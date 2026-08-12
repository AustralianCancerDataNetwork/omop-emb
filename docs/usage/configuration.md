# Configuration Reference

All configuration is done via [oa-configurator](https://AustralianCancerDataNetwork.github.io/oa-configurator/) (`~/.config/omop/config.toml`). There are no `OMOP_EMB_*` environment variables and no `.env` file read by the Python package itself.

---

## `[tools.omop_emb]`

Written by `omop-config configure omop_emb`.

| Field | References | Required | Description |
|---|---|---|---|
| `cdm_db` | a `[databases.*]` entry, `kind = "cdm"` | yes | The OMOP CDM database. Needed for the concept ingestion CLI commands (`add-embeddings`, `add-embeddings-with-index`) and for search-result enrichment; not needed for `list-models`, `rebuild-index`, `delete-model`, or library usage against already-computed embeddings. |
| `embedding_model_name` | a `[models.*]` entry | yes | Which model generates embeddings. Defaults to `"embedding-model"`. |
| `vector_store_name` | a `[vector_stores.*]` entry | yes | Which storage backend holds them. Defaults to `"vector_store"`. |

```toml
[tools.omop_emb]
cdm_db               = "cdm_db"
embedding_model_name = "embedding-model"
vector_store_name    = "vector_store"
```

---

## `[vector_stores.<name>]`

Backend selection: `backend_type` (`sqlitevec` or `pgvector`), and `database` naming a `[databases.*]` entry of kind `"generic"`. A sqlite-vec store is expressed the same way as pgvector: `database` points at an entry whose own `[connections.*]` entry has `dialect = "sqlite"`, `database_name = <path or ":memory:">`. There is no separate sqlite-path field.

```toml
# sqlite-vec
[connections.emb]
dialect       = "sqlite"
database_name = "/data/omop_emb.db"

[databases.emb_db]
kind       = "generic"
connection = "emb"

[vector_stores.vector_store]
backend_type = "sqlitevec"
database     = "emb_db"
```

```toml
# pgvector
[connections.emb]
dialect       = "postgresql+psycopg"
host          = "localhost"
port          = 5432
user          = "omop_emb"
password      = "omop_emb"
database_name = "omop_emb"

[databases.emb_db]
kind       = "generic"
connection = "emb"

[vector_stores.vector_store]
backend_type = "pgvector"
database     = "emb_db"
```

`faiss_cache_dir` (optional): directory for FAISS index files, for the read-only search path only. Read by `EmbeddingReaderInterface` when `--faiss-cache-dir` isn't passed explicitly to `embeddings search`.

```toml
[vector_stores.vector_store]
backend_type    = "pgvector"
database        = "emb_db"
faiss_cache_dir = "/data/faiss_cache"
```

See [oa-configurator's Config Reference](https://AustralianCancerDataNetwork.github.io/oa-configurator/config-reference/#vector_storesname) for the full field list, and [Backend selection](backend-selection.md) for choosing between the two backends.

---

## `[models.<name>]`

The embedding model itself, and asymmetric-model prefixes, live on the `[models.*]` entry `embedding_model_name` points at, not on `omop-emb`'s own config:

| Field | Description |
|---|---|
| `provider` | Name of a `[providers.*]` entry this model is served through |
| `model` | Model name or identifier |
| `embedding_dim` | Dimensionality override. Usually unset; auto-discovered via the provider or a live probe. |
| `document_prefix` | Task prefix prepended to concept texts at index time (asymmetric models only) |
| `query_prefix` | Task prefix prepended to search queries at query time (asymmetric models only) |

```toml
[providers.local-ollama]
provider = "ollama"
base_url = "http://localhost:11434"

[models.embedding-model]
provider        = "local-ollama"
model           = "nomic-embed-text:v1.5"
document_prefix = "search_document: "
query_prefix    = "search_query: "
```

| Model | Document prefix | Query prefix |
|---|---|---|
| `nomic-embed-text` | `search_document: ` | `search_query: ` |
| E5 family | `passage: ` | `query: ` |
| BGE family | `Represent this sentence for searching relevant passages: ` | `query: ` |

Symmetric models (e.g. `text-embedding-3-small`) do not need prefixes; leave both unset. See [Asymmetric Embeddings](asymmetric-embeddings.md) for details.

---

## Setup

```bash
omop-config init
omop-config configure omop_alchemy   # CDM database
omop-config configure omop_emb       # walks through models/vector_stores interactively
```

See [Getting Started: Configuration](../getting-started/configuration.md) for the full walkthrough.
