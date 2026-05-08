"""Embedding-related CLI commands for omop-emb."""

import itertools
import logging
from typing import Annotated, Generator, List, Optional, Sequence, Union

import typer
from dotenv import load_dotenv
from tqdm import tqdm

from orm_loader.helpers import create_db

from .utils import configure_logging_level, resolve_backend, resolve_omop_cdm_engine
from omop_emb.backends.index_config import index_config_from_index_type
from omop_emb.config import IndexType, MetricType
from omop_emb.embeddings import EmbeddingClient
from omop_emb.interface import EmbeddingReaderInterface, EmbeddingWriterInterface
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter, NearestConceptMatch

logger = logging.getLogger(__name__)
app = typer.Typer(help="Commands related to embedding generation, storage, and management.")


def consolidate_queries(
    queries: Optional[Union[str, List[str]]],
    queries_file: Optional[str],
) -> Generator[str, None, None]:
    if queries is not None and queries_file is not None:
        raise ValueError("Cannot provide both --queries and --queries-file.")

    if queries_file is not None:
        with open(queries_file, "r") as f:
            for line in f:
                yield line.strip()
    elif queries is not None:
        if isinstance(queries, str):
            yield queries.strip()
        elif isinstance(queries, list):
            for query in queries:
                yield query.strip()
        else:
            raise ValueError("Invalid type for queries. Expected a string or a list of strings.")
    else:
        raise ValueError("No queries provided.")


def _render_search_results(
    *,
    query_id: int,
    query_text: str,
    matches: Sequence[NearestConceptMatch],
) -> list[str]:
    if not matches:
        return [f"{query_id}\t{query_text}\tNo matches found"]
    rendered_rows: list[str] = []
    for rank, match in enumerate(matches, start=1):
        concept_name = match.concept_name or ""
        rendered_rows.append(
            f"{query_id}\t{query_text}\t{rank}\t{match.concept_id}\t{match.similarity:.6f}\t{concept_name}"
        )
    return rendered_rows


@app.command()
def add_embeddings(
    api_base: Annotated[str, typer.Option(
        "--api-base",
        help="Base URL for the embedding API.",
        rich_help_panel="Embedding API Options",
    )],
    api_key: Annotated[str, typer.Option(
        "--api-key",
        help="API key for the embedding API.",
        rich_help_panel="Embedding API Options",
    )],
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size for generating and inserting embeddings.",
        rich_help_panel="Embedding API Options",
    )] = 100,
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Embedding model name (e.g. 'text-embedding-3-small').",
        rich_help_panel="Embedding API Options",
    )] = "text-embedding-3-small",
    standard_only: Annotated[bool, typer.Option(
        "--standard-only",
        help="Only embed OMOP standard concepts (standard_concept = 'S').",
        rich_help_panel="Concept Filters",
    )] = False,
    vocabularies: Annotated[Optional[list[str]], typer.Option(
        "--vocabulary",
        help="Embed only concepts from specific OMOP vocabularies. Repeat to add multiple.",
        rich_help_panel="Concept Filters",
    )] = None,
    domains: Annotated[Optional[list[str]], typer.Option(
        "--domain",
        help="Embed only concepts from specific OMOP domains. Repeat to add multiple.",
        rich_help_panel="Concept Filters",
    )] = None,
    num_embeddings: Annotated[Optional[int], typer.Option(
        "--num-embeddings", "-n",
        help="Limit the number of concepts to embed. Useful for testing.",
        rich_help_panel="Concept Filters",
    )] = None,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    """Bulk generate and store embeddings for OMOP concepts.

    Models are always registered with a FLAT (exact scan) index. Use
    ``create-index`` afterwards to upgrade to an HNSW approximate index.
    """
    configure_logging_level(verbosity)
    load_dotenv()

    backend = resolve_backend()
    omop_cdm_engine = resolve_omop_cdm_engine()
    create_db(omop_cdm_engine)

    embedding_client = EmbeddingClient(
        model=model,
        api_base=api_base,
        api_key=api_key,
        embedding_batch_size=batch_size,
    )
    # FLAT registration — metric_type=COSINE is used only for upsert validation;
    # FLAT accepts any backend-supported metric, so COSINE is always valid here.
    embedding_writer = EmbeddingWriterInterface(
        backend=backend,
        metric_type=MetricType.COSINE,
        embedding_client=embedding_client,
    )
    embedding_writer.register_model()

    concept_filter = EmbeddingConceptFilter(
        require_standard=standard_only,
        domains=tuple(domains) if domains else None,
        vocabularies=tuple(vocabularies) if vocabularies else None,
    )

    missing = embedding_writer.get_concepts_without_embedding(
        omop_cdm_engine=omop_cdm_engine,
        concept_filter=concept_filter,
    )
    if num_embeddings is not None:
        missing = dict(list(missing.items())[:num_embeddings])

    total_concepts = len(missing)
    typer.echo(f"Total concepts to process: {total_concepts:,}")

    from itertools import batched as _batched
    with tqdm(total=total_concepts, desc="Processing", unit="concept") as pbar:
        for batch in _batched(missing.items(), batch_size):
            batch_dict = dict(batch)
            embedding_writer.embed_and_upsert_concepts(
                omop_cdm_engine=omop_cdm_engine,
                concept_ids=tuple(batch_dict.keys()),
                concept_texts=tuple(batch_dict.values()),
                batch_size=batch_size,
            )
            pbar.update(len(batch_dict))

    logger.info("Completed embedding generation and storage.")


@app.command()
def create_index(
    api_base: Annotated[str, typer.Option(
        "--api-base",
        help="Base URL for the embedding API.",
        rich_help_panel="Embedding API Options",
    )],
    api_key: Annotated[str, typer.Option(
        "--api-key",
        help="API key for the embedding API.",
        rich_help_panel="Embedding API Options",
    )],
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Embedding model name to build the index for.",
        rich_help_panel="Embedding API Options",
    )] = "text-embedding-3-small",
    metric_type: Annotated[MetricType, typer.Option(
        "--metric-type",
        help="Distance metric. Required and locked in when --index-type is HNSW.",
        rich_help_panel="Index Options",
    )] = MetricType.COSINE,
    index_type: Annotated[IndexType, typer.Option(
        "--index-type",
        help="Index type to build (FLAT = exact scan, HNSW = approximate).",
        rich_help_panel="Index Options",
    )] = IndexType.FLAT,
    index_hnsw_num_neighbors: Annotated[Optional[int], typer.Option(
        "--index-hnsw-num-neighbors",
        help="HNSW: number of neighbors per graph node.",
        rich_help_panel="Index Options",
    )] = None,
    index_hnsw_ef_search: Annotated[Optional[int], typer.Option(
        "--index-hnsw-ef-search",
        help="HNSW: ef parameter controlling recall during search.",
        rich_help_panel="Index Options",
    )] = None,
    index_hnsw_ef_construction: Annotated[Optional[int], typer.Option(
        "--index-ef-construction",
        help="HNSW: ef parameter controlling graph quality during construction.",
        rich_help_panel="Index Options",
    )] = None,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    """Create or rebuild the index on an existing embedding table.

    Index creation is expensive and should only be done after all desired
    embeddings are generated. When --index-type is HNSW the --metric-type is
    locked in and all subsequent queries must use the same metric.
    """
    configure_logging_level(verbosity)
    load_dotenv()

    backend = resolve_backend()
    embedding_client = EmbeddingClient(
        model=model,
        api_base=api_base,
        api_key=api_key,
    )
    embedding_writer = EmbeddingWriterInterface(
        backend=backend,
        metric_type=metric_type,
        embedding_client=embedding_client,
    )

    index_config = index_config_from_index_type(
        index_type,
        num_neighbors=index_hnsw_num_neighbors,
        ef_search=index_hnsw_ef_search,
        ef_construction=index_hnsw_ef_construction,
        metric_type=metric_type if index_type == IndexType.HNSW else None,
    )
    embedding_writer.rebuild_index(index_config=index_config)
    metric_info = f" (metric={metric_type.value})" if index_type == IndexType.HNSW else ""
    typer.echo(f"Index ({index_type.value}) built for '{model}'{metric_info}.")


@app.command()
def add_embeddings_with_index(
    api_base: Annotated[str, typer.Option(
        "--api-base",
        help="Base URL for the embedding API.",
        rich_help_panel="Embedding API Options",
    )],
    api_key: Annotated[str, typer.Option(
        "--api-key",
        help="API key for the embedding API.",
        rich_help_panel="Embedding API Options",
    )],
    metric_type: Annotated[MetricType, typer.Option(
        "--metric-type",
        help="Distance metric for the index. Locked in when --index-type is HNSW.",
        rich_help_panel="Index Options",
    )] = MetricType.COSINE,
    index_type: Annotated[IndexType, typer.Option(
        "--index-type",
        help="Index type to build after ingestion (FLAT = no index, HNSW = approximate).",
        rich_help_panel="Index Options",
    )] = IndexType.FLAT,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size for generating and inserting embeddings.",
        rich_help_panel="Embedding API Options",
    )] = 100,
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Embedding model name (e.g. 'text-embedding-3-small').",
        rich_help_panel="Embedding API Options",
    )] = "text-embedding-3-small",
    standard_only: Annotated[bool, typer.Option(
        "--standard-only",
        help="Only embed OMOP standard concepts (standard_concept = 'S').",
        rich_help_panel="Concept Filters",
    )] = False,
    vocabularies: Annotated[Optional[list[str]], typer.Option(
        "--vocabulary",
        help="Embed only concepts from specific OMOP vocabularies. Repeat to add multiple.",
        rich_help_panel="Concept Filters",
    )] = None,
    domains: Annotated[Optional[list[str]], typer.Option(
        "--domain",
        help="Embed only concepts from specific OMOP domains. Repeat to add multiple.",
        rich_help_panel="Concept Filters",
    )] = None,
    num_embeddings: Annotated[Optional[int], typer.Option(
        "--num-embeddings", "-n",
        help="Limit the number of concepts to embed. Useful for testing.",
        rich_help_panel="Concept Filters",
    )] = None,
    index_hnsw_num_neighbors: Annotated[Optional[int], typer.Option(
        "--index-hnsw-num-neighbors",
        help="HNSW: number of neighbors per graph node.",
        rich_help_panel="Index Options",
    )] = None,
    index_hnsw_ef_search: Annotated[Optional[int], typer.Option(
        "--index-hnsw-ef-search",
        help="HNSW: ef parameter controlling recall during search.",
        rich_help_panel="Index Options",
    )] = None,
    index_hnsw_ef_construction: Annotated[Optional[int], typer.Option(
        "--index-ef-construction",
        help="HNSW: ef parameter controlling graph quality during construction.",
        rich_help_panel="Index Options",
    )] = None,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    """Generate embeddings then build an index. Combines ``add-embeddings`` and ``create-index``.

    Embeddings are always ingested with a FLAT (exact) index first, then the
    index is rebuilt to the requested type. Index creation is expensive. Only use a non-Flat index after all embeddings are stored.
    """
    add_embeddings(
        api_base=api_base,
        api_key=api_key,
        batch_size=batch_size,
        model=model,
        standard_only=standard_only,
        vocabularies=vocabularies,
        domains=domains,
        num_embeddings=num_embeddings,
        verbosity=verbosity,
    )

    create_index(
        api_base=api_base,
        api_key=api_key,
        model=model,
        metric_type=metric_type,
        index_type=index_type,
        index_hnsw_num_neighbors=index_hnsw_num_neighbors,
        index_hnsw_ef_search=index_hnsw_ef_search,
        index_hnsw_ef_construction=index_hnsw_ef_construction,
        verbosity=verbosity,
    )


@app.command()
def search(
    api_base: Annotated[str, typer.Option(
        "--api-base",
        help="Base URL for the embedding API.",
        rich_help_panel="Embedding API Options",
    )],
    api_key: Annotated[str, typer.Option(
        "--api-key",
        help="API key for the embedding API.",
        rich_help_panel="Embedding API Options",
    )],
    queries: Annotated[Optional[List[str]], typer.Option(
        "--query",
        help="Query text to search. Repeat to search multiple queries.",
    )] = None,
    queries_file: Annotated[Optional[str], typer.Option(
        "--queries-file",
        help="Path to a .txt file with one query per line.",
    )] = None,
    metric_type: Annotated[MetricType, typer.Option(
        "--metric-type",
        help="Distance metric for nearest-neighbor search. Must match the registered index metric for HNSW.",
        rich_help_panel="Search Options",
    )] = MetricType.COSINE,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size for query embedding generation.",
        rich_help_panel="Embedding API Options",
    )] = 100,
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Embedding model name.",
        rich_help_panel="Embedding API Options",
    )] = "text-embedding-3-small",
    k: Annotated[int, typer.Option(
        "--k",
        help="Number of nearest concepts to return.",
        rich_help_panel="Search Options",
    )] = 10,
    standard_only: Annotated[bool, typer.Option(
        "--standard-only",
        help="Only return standard OMOP concepts (standard_concept = 'S').",
        rich_help_panel="Concept Filters",
    )] = False,
    vocabularies: Annotated[Optional[list[str]], typer.Option(
        "--vocabulary",
        help="Filter results to specific OMOP vocabularies. Repeat to add multiple.",
        rich_help_panel="Concept Filters",
    )] = None,
    domains: Annotated[Optional[list[str]], typer.Option(
        "--domain",
        help="Filter results to specific OMOP domains. Repeat to add multiple.",
        rich_help_panel="Concept Filters",
    )] = None,
    faiss_cache_dir: Annotated[Optional[str], typer.Option(
        "--faiss-cache-dir",
        help="Directory to cache FAISS index files for on-disk search with FAISS instead of the regular backend.",
        rich_help_panel="Search Options",
    )] = None,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()

    queries_generator = consolidate_queries(queries=queries, queries_file=queries_file)
    backend = resolve_backend()

    # CDM enrichment is optional for search
    try:
        omop_cdm_engine = resolve_omop_cdm_engine()
    except RuntimeError:
        omop_cdm_engine = None
        logger.info("CDM engine not configured; concept names will not be enriched in results.")

    embedding_client = EmbeddingClient(
        model=model,
        api_base=api_base,
        api_key=api_key,
        embedding_batch_size=batch_size,
    )
    embedding_reader = EmbeddingReaderInterface(
        model=embedding_client.canonical_model_name,
        backend=backend,
        metric_type=metric_type,
        omop_cdm_engine=omop_cdm_engine,
        provider_name_or_type=embedding_client.provider.provider_type,
        faiss_cache_dir=faiss_cache_dir,
    )

    concept_filter = EmbeddingConceptFilter(
        require_standard=standard_only,
        domains=tuple(domains) if domains else None,
        vocabularies=tuple(vocabularies) if vocabularies else None,
        limit=k,
    )

    for batch_id, batched_queries in enumerate(itertools.batched(queries_generator, batch_size)):
        batched_matches = embedding_reader.get_nearest_concepts_from_query_texts(
            query_texts=batched_queries,
            embedding_client=embedding_client,
            concept_filter=concept_filter,
            k=k,
        )
        for query_id, (query_text, matches_per_query) in enumerate(
            zip(batched_queries, batched_matches)
        ):
            for row in _render_search_results(
                query_id=query_id + batch_id * batch_size,
                query_text=query_text,
                matches=matches_per_query,
            ):
                print(row)
