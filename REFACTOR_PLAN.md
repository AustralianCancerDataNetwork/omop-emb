# omop-emb Registry Refactor Plan

## Design Decisions (settled)

### One row per model
The primary key is `(model_name, provider_type, backend_type)`. A model has exactly one
active index at any time. `index_type` and `metric_type` are non-key columns derived from
`index_config`.

### FLAT is the default / ingestion state
`FlatIndexConfig.metric_type = None` means "exact scan, any caller-supplied metric is
valid." `HNSWIndexConfig.metric_type` is always a concrete non-null value and locks the
model to one distance metric.

### `metric_type` comes from the caller at query time
Every data-op and query-op takes `metric_type` as an explicit parameter. The
`require_registered_model` decorator validates it against the registry:
- Registry `metric_type = None` (FLAT) → any caller-supplied metric accepted (further
  checked against `is_supported_index_metric_combination_for_backend`).
- Registry `metric_type = X` (HNSW) → caller must supply exactly `X`.

### `index_config` is out of data-op signatures
Data ops (`upsert_embeddings`, `get_nearest_concepts`, etc.) take only
`(model_name, provider_type, metric_type)` plus operation-specific params.
`index_config` only appears in `register_model` and `rebuild_index`.

### `metadata` column is separate from `index_config` column
Two JSON columns on the ORM:
- `index_config` — written only by `register_model` / `rebuild_index`; synced with
  `index_type` / `metric_type` columns via the `@validates` hook.
- `metadata` — free-form operational data (FAISS cache info, user-supplied extras).
  Never contains `index_config` data; that key is reserved.

### `storage_identifier` drops the metric suffix
`{backend}_{safe_model}`. One table per model, no metric encoding needed.

### Ingestion always uses FLAT; index building is a separate step
`register_model` enforces `FlatIndexConfig` only. `rebuild_index` transitions to HNSW
(or back to FLAT) after data has been ingested.

### `rebuild_index` is also `switch_index_type`
A single `backend.rebuild_index(model_name, provider_type, index_config)` call both
rebuilds the physical index and updates the registry row. Switching from FLAT to HNSW
and back is the same operation.

### Partial-write safety
Two operations span registry + DDL:
- `delete_model`: DDL (`DROP TABLE`) is executed **before** the registry delete so that
  a failure leaves the registry entry intact and the operation is re-runnable.
- `rebuild_index`: DDL (drop + create index) runs first, then the registry row is updated.
  If the registry update fails the physical index exists but the registry still says FLAT
  — the next search falls back to a seq scan; re-running `rebuild_index` recovers.
- `bulk_upsert_embeddings`: each batch is its own transaction; partial failure leaves
  partial ingestion, not corruption. Re-running is idempotent.

### CDM enrichment is concept_name only
The embedding table already stores `domain_id`, `vocabulary_id`, `is_standard` (and will
store `is_valid` once item 40 is done). The only CDM field not in the table is
`concept_name`. The `_enrich` path should fetch only that column.

---

## Completed (items 1–28)

All registry-core, backend, interface, CLI, FAISS touch-up, and test layers have been
refactored. Items below track the **remaining work**.

---

## TODO list

Work through these one at a time in order. Each item is one logical unit of change.
Wait for the user to commit after each item before starting the next.

---

### Phase A — Small targeted fixes

- [x] **29. `base_backend.py` + `pg_backend.py` — clarify `rebuild_index` docstring**
  The docstring for `rebuild_index` should explicitly state it also handles switching
  index type (FLAT → HNSW and back). Remove the misleading comment inside
  `pg_backend._rebuild_index_impl` that says "For FLAT use COSINE as a no-op rebuild"
  — FLAT has no index to rebuild; the FlatIndexManager ignores the metric argument.

- [x] **30. `embedding_utils.py` — move HAMMING guard from conversion to validation**
  `get_similarity_from_distance` raises `NotImplementedError` for HAMMING. This means
  DB rows are fetched and then the crash happens when building `NearestConceptMatch`.
  Move the guard to `get_distance` in `pg_sql.py` (raise `ValueError` before the query
  runs). Confirm the error is also raised in the sqlitevec equivalent if HAMMING is
  attempted there.

- [x] **31. CLI audit — `typer.echo` vs `logger`**
  Scan `cli_embeddings.py`, `cli_maintenance.py`, `cli_diagnostics.py`. Rule:
  `typer.echo` for always-visible user output (results, confirmation prompts, status
  lines the user should always see). `logger.info/warning` for operational steps that
  only appear with `-v`. Fix any misuse where an operational message uses `typer.echo`
  or a result line uses `logger`.

- [x] **32. `cli/utils.py` — use `BackendType()` constructor**
  Replace string comparisons (`backend_str == BackendType.SQLITEVEC.value`) with a
  `BackendType(backend_str)` constructor call wrapped in a try/except that raises a
  clear `RuntimeError` for unknown values. The final fallback `raise RuntimeError` then
  becomes unreachable dead code; remove it.

---

### Phase B — Bug fixes

- [ ] **33. `pg_backend.py` — fix FLAT metric passthrough in `_get_nearest_concepts_impl`**
  `_get_nearest_concepts_impl` currently ignores the caller's validated metric for FLAT
  models (`metric = model_record.metric_type or MetricType.COSINE`). If the caller
  requests L2, pgvector runs cosine operators — semantically wrong results.

  Fix: add `metric_type: MetricType` to the `_get_nearest_concepts_impl` abstract
  signature in `base_backend.py` and pass the caller's `metric_type` through from the
  public `get_nearest_concepts`. Remove the 8-line block comment at lines 263–271 once
  the bug is fixed. Update the sqlite-vec impl accordingly (it already reads the
  metric from the table spec, but confirm the new param is wired).

- [ ] **34. `interface.py` — split `_fetch_cdm_concept_metadata`**
  The function currently returns `dict[int, object]` (actually `dict[int, Row]`) and
  fetches 4 columns used inconsistently by its two callers:
  - `embed_and_upsert_concepts` needs `domain_id`, `vocabulary_id`, `standard_concept`.
  - `_enrich` needs only `concept_name`.
  Split into two focused private functions with correct return type annotations. Drop
  `invalid_reason` from the enrichment query (see item 36 for the filter approach).

---

### Phase C — Feature / schema work

- [ ] **35. `embedding_utils.py` + `pg_sql.py` — extend `EmbeddingConceptFilter.apply()`**
  Add an `apply_to_embedding_table(query: Select, table) -> Select` method alongside
  the existing CDM-targeted `apply()`. The new method maps filter fields to embedding
  table columns:
  - `concept_ids` → `table.concept_id.in_(...)`
  - `domains` → `table.domain_id.in_(...)`
  - `vocabularies` → `table.vocabulary_id.in_(...)`
  - `require_standard` → `table.is_standard == True`
  - `require_active` (see item 36) → `table.is_valid == True`
  Use this in `q_nearest_concept_ids` instead of the manual per-field expansion.

- [ ] **36. Narrow CDM enrichment + add `require_active` filter (two sub-items)**

  **36a. `interface.py` `_enrich` — concept_name only**
  Change `_enrich` to fetch only `concept_name` from CDM. Remove `is_active` from
  `NearestConceptMatch` enrichment (it's covered by the filter). Populate
  `NearestConceptMatch.is_standard` from the backend result directly (the embedding
  table already stores it) rather than from the CDM round-trip.
  `NearestConceptMatch.is_active` stays as an optional field for now but is not
  populated by the CDM path.

  **36b. Schema — add `is_valid` column to embedding table + `require_active` filter**
  Add a `is_valid: bool` column to the embedding table (`True` when CDM
  `invalid_reason NOT IN ('D', 'U')`). Populate it at upsert time via
  `ConceptEmbeddingRecord`. Add `require_active: bool = False` to
  `EmbeddingConceptFilter`. Wire it into `apply_to_embedding_table`. This is a schema
  migration — add `is_valid` with a `DEFAULT TRUE` to avoid breaking existing rows.
  Update `ConceptEmbeddingRecord`, `embed_and_upsert_concepts`, and the upsert SQL.

- [ ] **37. `base_backend.py` — reorder `delete_model` for safety**
  Move `_delete_storage_table` before `_registry.delete_model`. If the DDL step fails
  the registry entry remains intact and the call is re-runnable. If it succeeds, the
  physical table is gone before the registry entry is removed — the only failure mode
  leaves a registry entry pointing at a table that was already dropped, which is
  benign (next `_initialise_store` recreates an empty table).

- [ ] **38. Cache consolidation + fix `drop_pg_embedding_table`**
  Two related issues:
  - `_PGVECTOR_TABLE_CACHE` in `pg_sql.py` is module-level and persists across backend
    instances and test runs. `_table_cache` on `EmbeddingBackend` is instance-level and
    is the correct cache. Make `get_or_create_pg_embedding_table` not cache itself;
    let `_ensure_storage_table` in `base_backend` own all caching.
  - `drop_pg_embedding_table` issues a `DELETE FROM` (empties the table) instead of
    `DROP TABLE IF EXISTS` (removes it). Rename the function to match its true
    behaviour, or fix it to actually drop. Fix: issue `DROP TABLE IF EXISTS {tablename}`.

- [ ] **39. `cli/utils.py` + `cli_maintenance.py` — canonical model name clarity**
  Some internal method parameters named `model_name` silently require the canonical
  form (provider-normalised, e.g. `"nomic-embed-text:latest"`). Rename these
  parameters to `canonical_model_name` in the internal API so callers know which form
  is expected. Add a note to the `resolve_backend` / `EmbeddingReaderInterface`
  docstrings explaining the canonical vs raw distinction.

---

### Phase D — Docstrings pass

Every file that was modified in phases A–C (and the original refactor) must receive
Numpy-format docstrings:
- Classes: brief description + `Parameters` + `Notes` for non-obvious design.
- Methods / functions: description + `Parameters` + `Returns` + `Raises` + `Notes`.
- Enums: `Members` per value. Dataclasses: `Attributes`.

- [ ] **40. Docstrings — `embedding_utils.py`** (EmbeddingConceptFilter, NearestConceptMatch,
  get_similarity_from_distance, vector_column_type_for_dimensions)
- [ ] **41. Docstrings — `model_registry_orm.py`**
- [ ] **42. Docstrings — `index_config.py`**
- [ ] **43. Docstrings — `model_registry_types.py`**
- [ ] **44. Docstrings — `model_registry_manager.py`**
- [ ] **45. Docstrings — `base_backend.py`**
- [ ] **46. Docstrings — `sqlitevec_backend.py`**
- [ ] **47. Docstrings — `pg_backend.py`** + `pg_sql.py` + `pg_index_manager.py`
- [ ] **48. Docstrings — `interface.py`**
- [ ] **49. Docstrings — CLI files** (`cli_embeddings.py`, `cli_maintenance.py`,
  `cli_diagnostics.py`, `utils.py`)
- [ ] **50. Docstrings — `faiss_index_store.py`** (include the FLAT metric limitation note)

---

### Phase E — Optional dependency audit

- [ ] **51. Audit pgvector and FAISS import paths**
  Goal: every optional dependency has exactly one guarded entry point with a clear
  install-hint error. Trace all `import` sites:
  - `pgvector` is imported at module top of `pg_backend.py` inside a `try/except`.
    Direct construction of `PGVectorEmbeddingBackend` without going through
    `resolve_backend` gives a bare `ImportError` without the install hint. Unify the
    message.
  - `faiss` and `h5py` are checked lazily via `_require_faiss()`. Confirm no module
    that is imported at startup (e.g. `omop_emb/__init__.py`) re-exports a symbol from
    `faiss_index_store.py` without a guard.
  - Write a test that imports `omop_emb` with pgvector uninstalled (mock
    `importlib.import_module`) and confirms the error message includes the install hint.

---

### Phase F — Deferred / planning sessions

- [ ] **52. `metric_type` optional for FLAT + remove from non-KNN ops (deferred)**
  Merge of original points 1 and 10. Currently `metric_type` is required even on
  non-KNN operations (upsert, get_all_ids, has_any_embeddings) purely to satisfy the
  `require_registered_model` decorator — the impls don't use it.

  Design to settle:
  - Option A: lighter decorator for non-KNN ops that does only a 2-key model lookup
    (no metric validation).
  - Option B: make `metric_type` `Optional[MetricType]` everywhere and validate only
    when a value is supplied (HNSW: required; FLAT: optional, any value accepted).
  - Option C: remove `metric_type` from non-KNN ops entirely; validate only at KNN
    call site.

  Discuss once Phase C is complete and the call sites are cleaner.

- [ ] **53. FAISS plan (separate planning session)**
  FAISSCache has several open items: no CLI integration for `export`, staleness check
  is row-count only, `domains`/`vocabularies`/`require_standard` filters are ignored
  in FAISS search (documented footgun), `_create_empty_index` only handles Flat and
  HNSW configs. Plan a dedicated session when ready.

---

## Future (not on active list)

- **Bit-vector column support** (pgvector `bit` type for HAMMING and JACCARD metrics,
  sqlite-vec `int8`/bit column types). Both HAMMING (`<~>`) and JACCARD (`<%>`) in
  pgvector are bit-vector operators; our embedding table currently uses `vector`/`halfvec`
  (float32/float16). Supporting these metrics requires a new `VectorColumnType.BIT` path
  through the DDL and upsert layers. Needs research into pgvector bit-vector semantics
  and sqlite-vec quantization before planning.
