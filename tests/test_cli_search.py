from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

from oa_configurator import ConfigurationError

from omop_emb.cli import cli_embeddings


def test_search_skips_enrichment_when_cdm_is_unconfigured(monkeypatch):
    config = SimpleNamespace(vector_store_name="test-store")
    resolved_store = SimpleNamespace(faiss_cache_dir=None)
    search_args = {}
    monkeypatch.setattr(cli_embeddings, "_get_config", lambda: config)
    monkeypatch.setattr(cli_embeddings, "_resolve_model", lambda _name, _config: "model")
    monkeypatch.setattr(
        cli_embeddings.Resolver,
        "from_active_config",
        classmethod(
            lambda _cls: SimpleNamespace(
                resolve_vector_store=lambda _name: resolved_store,
            )
        ),
    )
    monkeypatch.setattr(
        cli_embeddings, "open_vector_store_reader", lambda _store: nullcontext("store")
    )

    def unconfigured_cdm():
        raise ConfigurationError("cdm_db is not configured")

    monkeypatch.setattr(cli_embeddings, "open_cdm_sessions", unconfigured_cdm)
    monkeypatch.setattr(
        cli_embeddings,
        "_search",
        lambda _queries, **kwargs: search_args.update(kwargs),
    )

    cli_embeddings.search(queries=["query"])

    assert search_args["cdm_sessions"] is None
