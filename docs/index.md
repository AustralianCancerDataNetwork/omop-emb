# OMOP Embeddings

`omop-emb` is an optional package to super-charge `omop-graph` and provide additional graph reasoning tools for information retrieval and RAG-based knowledge extraction.

The package itself is in its current stage a simple wrapper and supports the following:

- Dynamic embedding table creation and information retrieval based on the `model_name`
- Fast querying of embeddings using [`pgvectorscale`](https://github.com/timescale/pgvectorscale) for Postgres databases
- [`omop-alchemy`](https://AustralianCancerDataNetwork.github.io/OMOP_Alchemy/) wrapper of new tables
- CLI scripts to add embeddings to an already existing OMOP CDM


## Documentation overview
- [Installation](usage/installation.md)
- [CLI Reference](usage/cli.md)
