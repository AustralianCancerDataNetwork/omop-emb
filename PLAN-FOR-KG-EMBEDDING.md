Goal**

Build graph-aware OMOP terminology embeddings that preserve strong lexical grounding while injecting semantic structure from:

- `concept_synonym`
- `concept_ancestor`
- `concept_relationship`

Use `omop-graph/` as a reusable graph/query layer where it helps, not as a dependency by default unless it materially simplifies the pipeline.

**Recommended architecture**

Start with a staged system, not a pure graph-embedding model.

1. Base encoder
   Fine-tune the same text embedding model family you already use for `omop-emb`.

2. Training signal
   Contrastive training over graph-derived positive and hard-negative pairs/triples.

3. Retrieval
   Continue using `omop-emb` with FAISS/HNSW for ANN search.

4. Optional reranking later
   Add graph-aware reranking after top-k retrieval if needed.

That is the lowest-risk path and should integrate cleanly with the current project.

**Core design**

Represent each concept with a textual serialization, not just `concept_name`.

Candidate serialized input:
```text
Name: <preferred concept_name>
Vocabulary: <vocabulary_id>
Domain: <domain_id>
Synonyms: <top N synonyms>
Parents: <top M direct parent names>
Relations: <selected typed relation summaries>
```

Keep this configurable so you can ablate components.

**Training data construction**

Build a graph-aware training dataset with four pair classes.

1. Synonym positives
   Source: `concept_synonym`
   Examples:
   - preferred name <-> synonym
   - synonym <-> synonym for same concept
   Weight: highest

2. Hierarchy positives
   Source: `concept_ancestor`
   Start with `min_levels_of_separation = 1`
   Examples:
   - parent <-> child
   - child <-> parent
   Weight: high but below synonym
   Optional later:
   - depth 2 positives with lower weight

3. Typed relation positives
   Source: `concept_relationship`
   Restrict to semantically meaningful predicates only
   Use `omop-graph` predicate typing to keep:
   - `ONTOLOGICAL`
   - maybe `MAPPING`
   Exclude at first:
   - `METADATA`
   - noisy structural predicates unless explicitly justified

4. Hard negatives
   Important for terminology search quality
   Construct from:
   - lexical near-matches that are graph-distant
   - same vocabulary/domain but not close in graph
   - same parent but wrong branch
   - current embedding top-k false positives

This hard-negative set is where a lot of the value will come from.

**Suggested loss**

Phase 1:
- Standard contrastive / multiple-negatives ranking loss

Phase 2:
- Weighted positives by edge type
- Example weighting:
  - synonym: 1.0
  - parent-child: 0.7
  - typed relation: 0.4 to 0.6
  - depth-2 ancestor: 0.3

Do not start with a complex custom graph loss unless needed.

**What to reuse from `omop-graph`**

Likely reusable immediately:
- synonym lookup/query patterns
  [omop-graph/src/omop_graph/graph/queries.py](/ai-agent/omop-emb/omop-graph/src/omop_graph/graph/queries.py:68)
- ancestor/descendant queries
  [omop-graph/src/omop_graph/graph/queries.py](/ai-agent/omop-emb/omop-graph/src/omop_graph/graph/queries.py:178)
- predicate classification
  [omop-graph/src/omop_graph/graph/edges.py](/ai-agent/omop-emb/omop-graph/src/omop_graph/graph/edges.py:35)
- `KnowledgeGraph` facade if you want a cleaner graph API
  [omop-graph/src/omop_graph/graph/kg.py](/ai-agent/omop-emb/omop-graph/src/omop_graph/graph/kg.py:53)

Do not depend on `omop-graph` blindly. First decide:
- copy/port selected query utilities into `omop-emb`
- or add an optional extra for graph-aware training

My bias:
- reuse query logic and predicate typing patterns
- keep training pipeline inside `omop-emb`
- avoid making runtime search depend on `omop-graph`

**Proposed repo additions**

Add a new offline pipeline area, separate from CLI search:

- `src/omop_emb/graph_training/`
- `src/omop_emb/graph_training/extract.py`
- `src/omop_emb/graph_training/serialize.py`
- `src/omop_emb/graph_training/pairs.py`
- `src/omop_emb/graph_training/hard_negatives.py`
- `src/omop_emb/graph_training/train.py`
- `src/omop_emb/graph_training/eval.py`

Add scripts/CLI commands later if useful:
- `omop-emb export-graph-training-data`
- `omop-emb train-graph-aware-model`
- `omop-emb eval-graph-aware-model`

I would start with export + eval before adding train CLI polish.

**Concrete implementation phases**

1. Data extraction layer
   Build reproducible exports from DB or local files:
   - concept core fields
   - synonyms
   - depth-1 ancestors/descendants
   - filtered typed relations

   Output parquet/jsonl shards keyed by `concept_id`.

2. Text serialization layer
   Implement configurable concept-to-text rendering:
   - preferred label only
   - label + synonyms
   - label + synonyms + parents
   - label + selected relations

   This lets you run ablations.

3. Pair generation layer
   Emit training examples with:
   - `anchor_text`
   - `positive_text`
   - `negative_text` or candidate pool
   - `pair_type`
   - `weight`
   - source metadata

4. Hard-negative mining
   Start simple:
   - BM25/token overlap negatives
   - current embedding false positives
   - graph-distant lexical confounders

5. Training loop
   Fine-tune with sentence-transformers style contrastive training or equivalent.
   Save model in a form usable by your embedding service.

6. Evaluation harness
   Measure:
   - exact concept grounding accuracy @k
   - synonym retrieval accuracy
   - parent/child neighborhood quality
   - graph-aware nearest-neighbor coherence
   - latency/ANN impact after indexing with HNSW

7. Integration back into `omop-emb`
   Once a model is good:
   - embed concepts with `omop-emb add-embeddings`
   - use FAISS HNSW for search
   - compare against current baseline

**Evaluation plan**

You need offline evaluation before rolling this into production.

Build at least 4 eval sets:

1. Exact synonym grounding
   Query = synonym
   Target = concept_id

2. Lexical ambiguity set
   Queries with confusing near-strings
   Measure top-k accuracy and failure modes

3. Hierarchy coherence
   For each concept, nearest neighbors should include:
   - same concept synonyms
   - parent/child
   - close descendants
   more often than graph-distant lexical lookalikes

4. Typed relation retrieval
   If relation-aware training is used, test whether meaningful typed neighbors rise without harming lexical grounding

Primary metrics:
- recall@1, @5, @10
- MRR
- neighborhood purity by ancestor depth/domain/vocabulary
- search latency with HNSW

**Important modeling constraints**

Do first:
- synonyms
- direct hierarchy
- selected typed relations

Do later:
- full ancestor closure
- deep graph message passing
- metadata predicates

Reason:
Too much ancestor signal too early will oversmooth the space and hurt lexical specificity.

**Concrete first experiment**

Baseline experiment matrix:

1. `label-only`
2. `label + synonyms`
3. `label + synonyms + direct parents`
4. `label + synonyms + direct parents + selected relations`

Train each with the same contrastive recipe.
Compare on grounding accuracy and nearest-neighbor coherence.

My expectation:
`label + synonyms + direct parents` is likely the first strong win.

**Potential future synergy with HNSW**

If graph-aware training works, HNSW benefits indirectly because:
- local neighborhoods in vector space become more semantically faithful
- HNSW’s routing graph becomes better aligned with ontology structure
- retrieval quality should improve at the same ANN settings

But do not try to inject OMOP graph edges into HNSW itself initially.

**Context for the next Codex session**

Use this exact starting brief:

> We are in `/ai-agent/omop-emb`. The project already supports FAISS HNSW and recently gained:
> - `switch-index-type`
> - HNSW metadata stored in `model_registry.details`
> - improved FAISS rebuild progress logging
>
> There is also a temporary linked repo at `/ai-agent/omop-emb/omop-graph/` with useful graph query utilities over OMOP vocabulary tables.
>
> We want to implement the first stage of graph-aware terminology embedding training for OMOP concepts. The data sources are:
> - `concept_synonym`
> - `concept_ancestor`
> - `concept_relationship`
>
> Desired approach:
> - keep retrieval model text-first, not pure graph-only
> - generate graph-derived contrastive training data
> - start with synonyms, direct parent/child relations, and selected semantic relationships
> - avoid noisy metadata relationships
> - create an offline export/eval pipeline before any full training integration
>
> Please:
> 1. inspect current repo structure and `omop-graph/`
> 2. propose exact module/file additions under `src/omop_emb/graph_training/`
> 3. implement phase 1 data extraction and pair generation
> 4. add tests for pair generation logic
> 5. document assumptions and output schema

**Suggested first deliverable for that session**

Implement only phase 1:

- graph-training data export
- text serialization
- pair generation for:
  - synonym positives
  - parent-child positives
  - selected typed relation positives
- hard-negative scaffolding
- tests
- docs

That is the right slice to build next.