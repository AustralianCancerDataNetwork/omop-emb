# Performance Guide

Numbers are single-client medians for top-10 queries on 1024-dimension vectors (snowflake-arctic-embed-l-v2.0), measured in a 10-core container. They scale roughly with the number of vectors and dimensions, so read them as orders of magnitude.

## Which backend to use

| Stored vectors | Use | Query time |
|---|---|---|
| Up to about 50,000 | sqlite-vec | About 1 ms per 1,000 vectors: 10 ms at 10k, 43 ms at 50k. No server needed. |
| 50,000 to a few million, interactive | pgvector with an HNSW index | 2 to 5 ms at 300k |
| 50,000 to a few million, exact results or occasional batch jobs | pgvector FLAT | 0.4 s at 300k, 3.3 s at 1.8M |
| Any size, many queries at once | FAISS sidecar | 1.2 ms per query (HNSW) or 11 ms per query (flat) at 300k in batches of 100 |

Measured on a 300k-vector store:

| Backend and index | One query | Filtered query | 100 queries in one call |
|---|---|---|---|
| sqlite-vec | 342 ms | 138 ms | 345 ms per query |
| pgvector FLAT | 405 ms | 46 ms | 385 ms per query |
| pgvector HNSW, `ef_search` 16 | 2 to 5 ms | 4 ms | no gain over one at a time |
| FAISS flat | 30 ms | 8 ms | 10.8 ms per query |
| FAISS HNSW | 3.1 ms | 3.8 ms | 1.2 ms per query |

The filtered column uses a domain, standard and valid filter that keeps 6% of the concepts.

- sqlite-vec and pgvector run one scan per query vector, so a batch costs the sum of its queries. FAISS shares the scan across a batch, which makes a flat query about three times cheaper and an HNSW query about two and a half times cheaper than running it alone.
- HNSW is approximate. On a 300k sample, recall at 5 is 0.84 at the default `ef_search` of 16, and 0.98 at `ef_search` 200 once duplicate vectors are removed, which the valid-only default does.
- FAISS holds the whole index in memory, about 4.1 KB per vector for flat and 4.4 KB for HNSW (1.2 and 1.3 GB at 300k). The cache is a snapshot: rebuild it after the store changes (see `check-faiss-cache`).
- A cache build or export peaks at about 12 KB per vector of one batch, so the default 100,000-row batch needs about 1.5 GB. `--batch-size` lowers it.
- sqlite-vec files take about 4.2 KB per vector. pgvector tables take about 5.3 KB per vector, and an HNSW index adds 6 to 8 KB.
- Filtered HNSW queries on pgvector need extension 0.8.0 or newer, see [Embedding Storage](backend-selection.md#hnsw).

## Tuning pgvector

| Setting | What it does | Set it to |
|---|---|---|
| `shm_size` (container) | A parallel HNSW build asks for shared memory of `maintenance_work_mem` minus about 3 MiB. Docker's default is 64 MB, so larger builds fail with `could not resize shared memory segment`. Tables above roughly 100,000 rows build in parallel. | At least `maintenance_work_mem`, with margin. For 512MB use `1gb`. |
| `maintenance_work_mem` | The HNSW graph should fit in it, about 6 to 8 KB per vector at 1024 dimensions. A build that spills to disk is several times slower. | The expected index size if RAM allows, otherwise at least 512MB. Raise `shm_size` with it. |
| `max_parallel_maintenance_workers` | Workers for index builds, which need the shared memory above. | 2 to 4, or 0 to build serially without the shared-memory requirement. |
| `max_parallel_workers_per_gather` | Workers for a FLAT scan. On 1.8M vectors a scan took 5.8 s with none, 3.4 s with 2, and no less with 4 or 7. | Keep the default of 2. |
| `shared_buffers` and container memory | A FLAT scan reads the whole table (9.6 GB at 1.8M) and HNSW queries read the index. Whatever does not fit in memory is read from disk. | Enough for the index to fit. |
| `hnsw.ef_search` (per model, `--index-hnsw-ef-search`) | Recall against latency. At 300k a query took 2.5 to 5 ms at 16 and about 16 ms at 200. | The default of 16. Raise it to 64 to 200 only when recall matters. |

A Docker Compose service for a store above 100,000 vectors:

```yaml
omop-emb-db:
  image: pgvector/pgvector:pg18
  shm_size: 1gb
  command: >
    postgres
      -c shared_preload_libraries=vector
      -c maintenance_work_mem=512MB
      -c max_parallel_maintenance_workers=4
```

## Keeping the store small

- Only valid concepts are embedded by default. In one 1.8M-concept store 46% of the rows were invalid, and one retired name alone accounted for 26% of the rows with identical vectors, which also lowered HNSW recall.
- Remove invalid rows from an existing store with `maintenance prune-embeddings`, then run `rebuild-index` to reclaim the space.
- Scan time grows with the number of rows, so every pruned row speeds up FLAT queries and FAISS builds.
