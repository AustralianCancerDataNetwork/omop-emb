# [1.1.0](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v1.0.2...v1.1.0) (2026-06-30)


### Features

* release 1.1.0 ([88753ba](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/88753baf7875c847f40dfb06f5338fddd1b7ca4b))

## [1.1.0](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v1.0.2...v1.1.0) (2026-06-30)


### Features

* **config:** `OmopEmbConfig` now subclasses `PackageConfigBase` (oa-configurator), replacing env-var-based setup with a typed TOML-backed config; `omop-config configure omop_emb` provisions all settings interactively or via named flags ([#23](https://github.com/AustralianCancerDataNetwork/omop-emb/issues/23)) ([c591f8a](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/c591f8a))
* **storage:** HDF5 embedding bundle (`storage/embedding_bundle.py`) replaces FAISS as the round-trip artefact; raw embeddings written via chunked streaming (no full-dataset accumulation); FAISS is now a derived, rebuildable accelerator ([#30](https://github.com/AustralianCancerDataNetwork/omop-emb/issues/30)) ([c591f8a](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/c591f8a))
* **cli:** `export-bundle`, `import-bundle`, and `build-faiss-cache` commands added to the maintenance CLI ([c591f8a](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/c591f8a))
* **faiss:** concept pre-filter moved from a frozen `metadata.npz` snapshot to a live backend query (`get_concept_ids_matching_filter`), with LRU caching, so filters cannot drift from current database state ([c591f8a](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/c591f8a))


### Bug Fixes

* **faiss:** `FAISSCache._load_index` re-read the index from disk on every search call; `HNSWIndexConfig.ef_search` was never applied at query time — fixed with an in-memory LRU index cache keyed on `(metric_type, index_config)` ([#24](https://github.com/AustralianCancerDataNetwork/omop-emb/issues/24)) ([eb2bb3e](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/eb2bb3e))
* **faiss:** `IndexFlatIP` (COSINE) returned raw inner product fed into a formula expecting cosine distance; `IndexFlatL2` returned squared distance where true Euclidean was expected — both now converted before scoring (`1 - IP` for COSINE, `sqrt()` for L2) ([#27](https://github.com/AustralianCancerDataNetwork/omop-emb/issues/27)) ([eb2bb3e](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/eb2bb3e))
* **faiss:** COSINE export silently discarded original vector magnitudes (FAISS stores L2-normalised vectors on disk); re-importing from a `.faiss` file now emits a clear data-loss warning ([#30](https://github.com/AustralianCancerDataNetwork/omop-emb/issues/30)) ([a38c601](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/a38c601))
* **registry:** embedding upserts never bumped `model_registry.updated_at`, so `FAISSCache.is_fresh()` could report a cache fresh indefinitely after new embeddings were added — fixed via `refresh_model_updated_at_timestamp()` called after every upsert path ([eb2bb3e](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/eb2bb3e))
* **registry:** `import_bundle()` with `force=True` replaced model content without bumping `updated_at` ([eb2bb3e](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/eb2bb3e))
* **registry:** timestamp precision switched from DB-side `func.now()` (whole-second on SQLite) to Python-side `datetime.now(timezone.utc)` to avoid staleness races within the same second ([eb2bb3e](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/eb2bb3e))
* **cli:** `search` command defaulted `--model` to a hardcoded string instead of resolving against `cfg.embedding_model`; omitting `--model` now correctly falls back to the configured model ([5f3cf4c](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/5f3cf4c))
* **cli:** missing `~/.config/omop/config.toml` on a fresh install raised a bare `FileNotFoundError`; now raises an actionable message pointing at `omop-config configure omop-emb` ([5f3cf4c](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/5f3cf4c))
* **backend:** sqlite-vec backend silently fell back to an ephemeral `:memory:` database when `sqlite_path` was not configured; now raises a `RuntimeError` with setup guidance (explicit `:memory:` still accepted when set intentionally) ([5f3cf4c](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/5f3cf4c))
* **backend:** `EmbeddingConceptFilter.limit` was applied in the pgvector KNN path but silently ignored in the sqlite-vec path ([5f3cf4c](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/5f3cf4c))
* **backends:** unified column definitions into a single shared spec; rebuilt sqlite-vec queries on SQLAlchemy Core objects, removing hardcoded SQL strings ([ed9d7fc](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/ed9d7fc))


### Performance Improvements

* **faiss:** FAISS index cache now uses LRU eviction (`max_cached_indices=2`) so long-lived instances cannot accumulate unbounded memory across metric/index combinations ([9713820](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/9713820))
* **faiss:** concept filter ID sets cached per `EmbeddingConceptFilter` with LRU eviction (`max_cached_filters=4`), avoiding repeated full-table scans on repeated searches with the same filter ([5f3cf4c](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/5f3cf4c))


## [1.0.2](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v1.0.1...v1.0.2) (2026-06-03)


### Bug Fixes

* Support orm-loader >0.4.0 ([#22](https://github.com/AustralianCancerDataNetwork/omop-emb/issues/22)) ([b9601dc](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/b9601dc8a6d5bd7d70bdb3e1b7455cadc0eb6cfc))

## [1.0.1](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v1.0.0...v1.0.1) (2026-05-25)


### Bug Fixes

* Upsert from faiss-cache to DB ([#18](https://github.com/AustralianCancerDataNetwork/omop-emb/issues/18)) ([691de69](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/691de6938442139c39bb6907e10c79c82e3fe525))

# [1.0.0](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.4.1...v1.0.0) (2026-05-12)


### Performance Improvements

* Refactoring of backend for local-first storage solution ([7ff45e7](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/7ff45e705fb43584a96b2ab35549f7a1052a2e7b))


### BREAKING CHANGES

* Introduction of 2 new backends, alteration of interfaces, removal of FAISS as a standalone backend.

- Support of `SQLiteVecBackend` and `PGVectorEmbeddingBackend` with common interfaces and methods for storage and retrieval
- Unified ORM for both backends and associated tables for easy interaction using SQLAlchemy
- Standardised embedding storage table and `ModelRegistry`
- Enforcement of a singular `IndexType` per model to streamline ingestion and retrieval. CLI interfaces to update `IndexType` are provided.
- Improved CLI maintenance modules
- Default `IndexType.FLAT` ingestion to accelerate bulk ingestion and interfaces for creating, modifying and deleting indices
- FAISS: demoted from ground truth to read-acceleration, per-index staleness tracking via .json sidecars, `import-faiss-cache` CLI for round-tripping, transparent fast path in `EmbeddingReaderInterface` via `faiss_cache_dir`
- temporary table JOINs replacing IN (...) clauses across both backends
- prevent the hard-capped limit for each dialect
- optional packages with `pgvector` backend and `faiss` export
- fail at import time with install hint (`omop-emb[pgvector]`, `omop-emb[faiss-cpu]`)

## [0.4.1](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.4.0...v0.4.1) (2026-04-24)


### Bug Fixes

* **ci:** remove conditional from uv installation ([02912c5](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/02912c524b6acd8e0e0575ef0ad5c31085a5f1f9))
* **ci:** setup uv before release and anchor version sed ([50f0fe3](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/50f0fe3de70feba09ed9cb846a478cfff33f283c))
* include lock file in CI ([a57ab1e](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/a57ab1eb1072a364b747b3cb16341033619e737a))

# [0.4.0](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.3.3...v0.4.0) (2026-04-20)


### Features

* EmbeddingClient and Providers in omop-emb, fix non-canonical model naming without tags ([#6](https://github.com/AustralianCancerDataNetwork/omop-emb/issues/6)) ([1aaf8e7](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/1aaf8e732c12eadbe79c288118d9ec8ec2e25575))

## [0.3.3](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.3.2...v0.3.3) (2026-04-15)


### Bug Fixes

* Limit the number of concepts retrieved through EmbeddingConceptFilter for unified interface ([#5](https://github.com/AustralianCancerDataNetwork/omop-emb/issues/5)) ([4bfa86b](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/4bfa86b4808b5f2f01297bee76474fc1aaf5ad9c))

## [0.3.2](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.3.1...v0.3.2) (2026-04-12)


### Bug Fixes

* correctly normalise all distances to [0,1] similarities ([5657aef](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/5657aefb41f59cdfb405c21fdd640b1ee601f74d))

## [0.3.1](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.3.0...v0.3.1) (2026-04-12)


### Bug Fixes

* cleanup residuals ([dc890f7](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/dc890f7b3fd6b838a49eeab739e579034e6cb3bd))

# [0.3.0](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.2.5...v0.3.0) (2026-04-12)


### Features

* Local metadata storage ([#2](https://github.com/AustralianCancerDataNetwork/omop-emb/issues/2)) ([1aa8da5](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/1aa8da55c4790abe7e2a4540fc3c8cdad2f88470))

## [0.2.5](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.2.4...v0.2.5) (2026-04-03)


### Bug Fixes

* commit pgvector extension ([5fd66b6](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/5fd66b6ba95907efd76d39ed7bebde34a9f08708))
* list existing models correctly, create vector extension only for pgvector ([47618df](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/47618dfaa1e7400485d07fede548a590a617bc3e))

## [0.2.4](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.2.3...v0.2.4) (2026-04-02)


### Bug Fixes

* trigger release for latest main changes ([a188e0c](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/a188e0cdf2c33d4869fb84daaa6dc2a2408e2206))

## [0.2.3](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.2.2...v0.2.3) (2026-04-01)


### Bug Fixes

* redirect omop-llm dependency to pypi, adaptations to CI/CD ([03315a2](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/03315a26d507e9eb5caa54ffdbaa8a3aff740836))

## [0.2.2](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.2.1...v0.2.2) (2026-04-01)


### Bug Fixes

* trigger initial 0.x release ([45a8308](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/45a830801ccc92cc5bf806a4109f3a3447a98519))

## [0.2.1](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.2.0...v0.2.1) (2026-04-01)


### Bug Fixes

* trigger PyPI publish after OIDC config ([2fb4b40](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/2fb4b40c4d9221ceac1ae1f3a25f7059380b53bd))

# [0.2.0](https://github.com/AustralianCancerDataNetwork/omop-emb/compare/v0.1.0...v0.2.0) (2026-04-01)


### Bug Fixes

* pull newest omop-llm ([c3fb805](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/c3fb8050d84804f48a44966e5d4271465485652d))
* remove dupblicat optional-dep ([c7a58c1](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/c7a58c16078615a42e5dc6787e0abb44449ab273))
* Remove duplicate "scripts" key after PR ([9d8369d](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/9d8369d9c54aecddb39efcefada186748eb6ff0f))


### Features

* diverse interface for embedding backends ([3d696dc](https://github.com/AustralianCancerDataNetwork/omop-emb/commit/3d696dc906ac7a94ee7464b82459b0a9b9db3ed2))
