# omop-emb

Vector embedding layer for OMOP CDM concepts.

`omop-emb` generates, stores, and retrieves embeddings for OMOP concepts. It works out of the box with **sqlite-vec** (no external database required) and scales to **PostgreSQL/pgvector** for larger deployments. The database is the source of truth; FAISS is an optional read-acceleration sidecar, not a primary store.

## Installation

```bash
pip install omop-emb                         # sqlite-vec backend (default, no extras needed)
pip install "omop-emb[pgvector]"             # adds PostgreSQL/pgvector support
pip install "omop-emb[faiss-cpu]"            # adds FAISS sidecar support
pip install "omop-emb[pgvector,faiss-cpu]"   # everything
```

## Configuration

`omop-emb` is configured entirely through [oa-configurator](https://github.com/AustralianCancerDataNetwork/oa-configurator) (`~/.config/omop/config.toml`); there are no `OMOP_EMB_*` environment variables. Set up a CDM database, an embedding model, and a vector store once:

```bash
omop-config init
omop-config connections add cdm --dialect postgresql+psycopg --host localhost --database-name omop_cdm
omop-config databases add cdm_db --kind cdm --connection cdm

omop-config providers add local-ollama --provider ollama --base-url http://localhost:11434
omop-config models add embedding-model --provider local-ollama --model nomic-embed-text:v1.5

omop-config databases add emb_db --kind generic --connection cdm
omop-config vector-stores add vector_store --backend-type pgvector --database emb_db

omop-config configure omop_emb   # points OmopEmbConfig at the entries above, prompts for anything unset
```

`omop-config configure omop_emb` writes `[tools.omop_emb]` with `cdm_db`, `embedding_model_name`, and `vector_store_name` (each defaulting to the entry names above, if you use the same names).

## Quick start

```bash
omop-emb embeddings add-embeddings --model-name embedding-model
omop-emb embeddings search --model-name embedding-model \
    --query "hypertension" --query "type 2 diabetes" \
    --standard-only --domain Condition --k 5
```

`--model-name` defaults to the configured `embedding_model_name`, so it can be omitted once configured. See the [CLI reference](https://AustralianCancerDataNetwork.github.io/omop-emb/usage/cli/) for the full command list.

**pgvector with HNSW index:**

```bash
omop-emb embeddings add-embeddings
omop-emb maintenance rebuild-index --model-name embedding-model --index-type hnsw --metric-type cosine
```

## Documentation

Full documentation: <https://AustralianCancerDataNetwork.github.io/omop-emb>

- [Installation & backend setup](https://AustralianCancerDataNetwork.github.io/omop-emb/usage/installation/)
- [Configuration reference](https://AustralianCancerDataNetwork.github.io/omop-emb/usage/configuration/)
- [Backend selection & index types](https://AustralianCancerDataNetwork.github.io/omop-emb/usage/backend-selection/)
- [CLI reference](https://AustralianCancerDataNetwork.github.io/omop-emb/usage/cli/)
- [Interface guide](https://AustralianCancerDataNetwork.github.io/omop-emb/usage/interface-guide/)

## Roadmap

- [x] sqlite-vec backend (default, zero-config)
- [x] pgvector backend (PostgreSQL)
- [x] HNSW index support for pgvector
- [x] FAISS sidecar (approximate nearest-neighbour read acceleration)
- [x] Embedding bundle export / import CLI (`maintenance export`, `maintenance import`, `maintenance build-faiss-cache`)
- [x] In-DB concept filtering (domain, vocabulary, standard status, active status)
- [x] Transparent FAISS fast path in `EmbeddingReaderInterface`
- [x] Extensive backend and registry testing
- [ ] FAISS GPU support
- [ ] [`pgvectorscale`](https://github.com/timescale/pgvectorscale) support
- [ ] Vector quantisation for more efficient storage
