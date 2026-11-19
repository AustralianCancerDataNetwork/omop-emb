# Recall benchmark

Measures how many of the true nearest concepts a pgvector HNSW index returns, and how long it takes, for the global index and for per-domain partial indexes. Run it from the repository root as `python -m benchmarks.recall <command>`.

## Quick start

Needs a pgvector store with the model registered and an HNSW index, the OMOP CDM configured (to sample queries), and the embedding model reachable.

```bash
# 1. Pick the test queries and embed them. Writes recall_benchmark/plan.json and vectors.npy.
python -m benchmarks.recall queries

# 2. Compute the exact nearest concepts to score against. Slow, run once.
python -m benchmarks.recall truth

# 3. Measure the store as it is now, under a name.
python -m benchmarks.recall run --label before

# 4. Add the per-domain partial indexes. Prints a memory estimate and asks first.
python -m benchmarks.recall build --label partials

# 5. Measure again.
python -m benchmarks.recall run --label partials

# 6. Compare the runs. Writes recall_benchmark/comparison.md.
python -m benchmarks.recall compare
```

That is the whole benchmark. Step 2 is slow because the exact search scans every matching row. Plan on the order of an hour at 9.4M rows. It saves as it goes, so run it again after an interruption and it continues.

Steps 1, 2, 3 and 6 never write to the store. Step 4 adds indexes and records that setting in the registry, and leaves the global index alone. `drop` removes the partial indexes again.

## What each command does

| Command | What it does | Needed |
|---|---|---|
| `queries` | Chooses the test queries and embeds them. Also writes the scenarios, which are the domains, domain pairs and filter profiles that every later command uses | Yes |
| `truth` | Finds the true nearest concepts for every query and scenario by an exact scan of the same store. Every run is scored against these | Yes |
| `run --label X` | Measures recall, latency and the index the planner chose, for the indexes that exist right now | Yes |
| `build --label X` | Builds one partial index per domain, with `--valid-only` for valid concepts only | For the partial index runs |
| `compare` | Puts the labelled runs side by side | Yes |
| `drop` | Removes the partial indexes | No |
| `profile` | Prints rows per domain, valid and standard counts, index sizes and server settings. Add `--duplicates` to count rows sharing a vector | No |

## Comparing other index states

A run measures whichever indexes exist when it starts, and the label names that state. To compare more than the two states above, change the indexes, run again under a new label, then `compare`.

| Label | How to get there | Answers |
|---|---|---|
| `global-ef200` | `omop-emb maintenance rebuild-index --model <NAME> --index-type hnsw --index-ef-construction 200` | Whether a better-built global graph closes the gap |
| `global-ef200-half` | The same with `--index-precision float16` | Whether the half-size index loses recall. A full-precision build at 9.4M rows may not fit in memory, so this may be the only way to measure `ef_construction` 200 |
| `partials-valid` | `build --valid-only` | Whether valid-only partials are useful. Only queries that require active concepts can use them |

`rebuild-index` drops the global index and every partial index before it builds. Use a copy of the store, not one in use, and run `build` again afterwards to bring the partial indexes back.

## Using your own queries

`queries --queries file.json` takes a JSON list of `{"q": "text", "domain": "Drug"}` objects. Add `--vectors file.npy` with one precomputed vector per query to skip the model as well. The vectors must come from the model registered in the store. Either option skips the CDM sampling.

Other options of `queries` set the domains (`--domain`), pairs (`--pair`, for example `Drug+Condition`), filter profiles (`--profile`), queries per domain (`--per-domain`) and the seed. `run` takes `--setting` for the `ef_search` and scan cap values to measure, for example `--setting 200:100000`.

## What is measured

- **Scenarios** are a search without a domain, a search in each domain and each pair of domains, under three filter profiles: `any` (no flags), `active` and `active+standard`. The defaults are the domains Condition, Drug, Measurement, Observation and Procedure, and the pairs Measurement+Observation, Measurement+Procedure and Condition+Drug.
- **Settings** are `ef_search` 100, 200 and 400 at the default `max_scan_tuples`, and 200 with `max_scan_tuples` 100,000. Each setting runs one warm pass and three timed passes in random order, and pools the timings.
- **Recall** is counted at 10 and 25 by distance. A returned concept counts if its exact distance is no larger than the exact k-th distance, so concepts with identical vectors are interchangeable. The distance is recomputed from the stored vectors, so a half-precision index is scored fairly.
- **Also reported** are the share of queries below 0.8 recall@10, the share returning fewer than 25 results or none, and p50 and p95 latency of the production query path.
- **The index the planner used** is shown for a one-off plan and for a generic plan, which is what a driver reuses for a statement it runs repeatedly. This is the evidence for whether partial indexes are chosen at your size.
- **Split arms** run for pairs when partial indexes exist: one search per domain, merged by distance, sequentially and in parallel. They show whether building a merge for two-domain queries is worth it.

## Reading the comparison

- **Index states** lists the sizes of the global and partial indexes, the build time of the partials, `ef_construction`, the index precision and the server memory settings of each run.
- **Expected workload** weights single-domain searches 0.835 and searches without a domain 0.165, from the evaluation workload. Change them with `--weight-single` and `--weight-none` on `compare`.
- **Reference setting** lists the cells below 0.95 recall@10 at `ef_search` 200.
- **Index chosen by the planner** shows `global`, `partial:<domain>` or `seq scan` for each scenario.
- **Fan-out** applies one rule to the split arms: the combined filter is below 0.95 recall@10, the split reaches 0.95, and its p95 is at most twice the combined filter's. The thresholds are constants in `report.py`.

## Limits

- Timings come from a warm cache on one server and include its background load. Compare runs made on the same machine in the same session.
- Embedding time and concurrency are not measured.
- The `--model-name` option takes the name of a model entry in the omop-emb configuration. It defaults to the configured model.
