# Configuration

omop-emb reads all database, model, and vector-store settings from [oa-configurator](https://github.com/AustralianCancerDataNetwork/oa-configurator). No environment variables are needed for the Python package itself.

## Quick start

omop-emb requires the CDM database configured by omop-alchemy, an embedding model, and a vector store. Configure the CDM database first, then omop-emb:

```bash
omop-config init          # creates ~/.config/omop/config.toml if absent
omop-config configure omop_alchemy
omop-config configure omop_emb
```

`omop-config configure omop_emb` interactively resolves or creates whatever `[models.*]`/`[vector_stores.*]` entries it needs, recursing into `[providers.*]`/`[databases.*]`/`[connections.*]` as required.

## What gets configured

`OmopEmbConfig` (`[tools.omop_emb]`) has three fields:

- `cdm_db`: names a `[databases.*]` entry of kind `"cdm"`, shared by naming convention with omop-alchemy's own `cdm_db` field
- `embedding_model_name`: names a `[models.*]` entry
- `vector_store_name`: names a `[vector_stores.*]` entry (which itself names a `[databases.*]` entry of kind `"generic"` for the embedding table storage)

## Verify

```bash
omop-config verify
omop-emb diagnostics health-check
```

## Multiple instances

To configure a second vector store (e.g. for production), create it under its own name and point the field's own flag at it:

```bash
omop-config vector-stores add vector_store_prod --backend-type pgvector --database emb_db_prod
omop-config configure omop_emb --vector-store-name vector_store_prod
```

This creates `vector_store_prod` without touching the existing `vector_store`. To use a second CDM database instead, configure omop-alchemy the same way:

```bash
omop-config configure omop_alchemy --cdm-db cdm_db_prod
```

There is no "default" toggle to flip afterward; each deployment's `configure` call names the entry it wants directly. See the [oa-configurator integration guide](https://AustralianCancerDataNetwork.github.io/oa-configurator/integration/#multiple-environments) for the full multi-environment pattern.

## Further reading

- [oa-configurator integration guide](https://AustralianCancerDataNetwork.github.io/oa-configurator/integration/): full config reference, multi-package setups
- [Backend selection](../usage/backend-selection.md): choosing between pgvector and sqlite-vec
