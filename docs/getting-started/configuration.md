# Configuration

omop-emb reads all database connection and schema settings from
[oa-configurator](https://github.com/AustralianCancerDataNetwork/oa-configurator).
No environment variables are needed for the Python package itself.

## Quick start

omop-emb owns the embedding database and requires the CDM database configured by
omop-alchemy. Configure both:

```bash
omop-config init          # creates ~/.config/omop/config.toml if absent
omop-config configure omop_alchemy
omop-config configure omop_emb
```

## What gets configured

omop-emb owns the `emb_db` resource (pgvector embedding database). It requires the
`cdm_db` resource configured by omop-alchemy for concept ingestion.

Package-specific settings (backend, embedding prefixes) are stored under
`[tools.omop_emb]` in `config.toml`.

## Verify

```bash
omop-config verify
```

## Docker Compose

The included `docker-compose.yaml` spins up both a CDM PostgreSQL database and a
pgvector embedding database, plus a `python-emb` container. Default credentials work
out of the box:

```bash
docker compose up
```

The `python-emb` container runs `omop-config configure` for `omop_alchemy` and
`omop_emb` at startup. Your `~/.config/omop/config.toml` on the host is written on
safe to re-run on subsequent starts: connection flags always apply, and any values already stored in `config.toml` are preserved for fields not explicitly provided.

To also start Ollama (for local model inference), use the `standalone` profile:

```bash
docker compose --profile standalone up
```

### Overriding default values

The compose file uses built-in defaults for all database credentials. To use different
values, create a `.env` file in this directory with any of the following variables:

| Variable | Default | Description |
|---|---|---|
| `OMOP_CDM_DB_USER` | `omop` | CDM database username |
| `OMOP_CDM_DB_PASSWORD` | `omop` | CDM database password |
| `OMOP_CDM_DB_NAME` | `omop_cdm` | CDM database name |
| `OMOP_EMB_DB_USER` | `omop_emb` | Embedding database username |
| `OMOP_EMB_DB_PASSWORD` | `omop_emb` | Embedding database password |
| `OMOP_EMB_DB_NAME` | `omop_emb` | Embedding database name |

Copy the example and edit as needed:

```bash
cp .env.example .env
# edit .env
docker compose up
```

The `.env` file is only read by Docker Compose for variable substitution — it is not
loaded by omop-emb at runtime.

## Multiple instances

To configure a second embedding database (e.g. for production), use `--resource-name`:

```bash
omop-config configure omop_emb --resource-name emb_db_prod
```

This creates `emb_db_prod` without touching the existing `emb_db`. Because two
resources now exist, configure automatically prompts you to choose the default at
the end of the same run — no second invocation needed.

To use a second CDM database instead, configure omop-alchemy the same way:

```bash
omop-config configure omop_alchemy --resource-name cdm_db_prod
```

To change the default later, set `default_resource` directly in `config.toml`:

```toml
[tools.omop_emb]
default_resource = "emb_db_prod"
```

See the [oa-configurator integration guide](https://AustralianCancerDataNetwork.github.io/oa-configurator/integration/#multiple-environments) for the full multi-environment guide.

## Further reading

- [oa-configurator integration guide](https://AustralianCancerDataNetwork.github.io/oa-configurator/integration/) — full config reference, profiles, multi-package setups
- [Backend selection](../usage/backend-selection.md) — choosing between pgvector and sqlite-vec
