"""Embedding-related CLI commands for omop-emb."""

from dotenv import load_dotenv
import itertools
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm
from typing import Annotated, Optional, List, Union, Generator, Sequence
import typer
app = typer.Typer(help="Commands related to embedding generation, storage, and management.")

from orm_loader.helpers import create_db

from .utils import configure_logging_level, resolve_omop_cdm_engine, resolve_emb_engine
from omop_emb.storage.index_config import index_config_from_index_type
from omop_emb.config import IndexType, MetricType
from omop_emb.embeddings import EmbeddingClient
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter, NearestConceptMatch
from omop_emb.interface import EmbeddingWriterInterface, EmbeddingReaderInterface


def consolidate_queries(
    queries: Optional[Union[str, List[str]]],
    queries_file: Optional[str]
) -> Generator[str, None, None]:

    if queries is not None and queries_file is not None:
        raise ValueError("Cannot provide both --queries and --queries-file.")

    if queries_file is not None:
        with open(queries_file, 'r') as f:
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
        rendered_rows.append(
            f"{query_id}\t{query_text}\t{rank}\t{match.concept_id}\t{match.similarity:.6f}\t{match.concept_name}"
        )
    return rendered_rows


@app.command(help="Add embeddings for concepts to the embedding store.")
def add_embeddings(
    api_base: Annotated[str, typer.Option(
        "--api-base",
        help="Base URL for the API to use for generating embeddings.",
        rich_help_panel="Embedding API Options",
    )],
    api_key: Annotated[str, typer.Option(
        "--api-key",
        help="API key for the embedding API.",
        rich_help_panel="Embedding API Options",
    )],
    index_type: Annotated[IndexType, typer.Option(
        "--index-type",
        help="Index type for newly registered models.",
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
        help="HNSW: number of neighbors per graph node. Required when --index-type is HNSW.",
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
    configure_logging_level(verbosity)
    load_dotenv()

    emb_engine = resolve_emb_engine()
    omop_cdm_engine = resolve_omop_cdm_engine()
    create_db(omop_cdm_engine)

    embedding_client = EmbeddingClient(
        model=model,
        api_base=api_base,
        api_key=api_key,
        embedding_batch_size=batch_size,
    )
    embedding_writer = EmbeddingWriterInterface(
        emb_engine=emb_engine,
        omop_cdm_engine=omop_cdm_engine,
        embedding_client=embedding_client,
    )

    index_config = index_config_from_index_type(
        index_type,
        num_neighbors=index_hnsw_num_neighbors,
        ef_search=index_hnsw_ef_search,
        ef_construction=index_hnsw_ef_construction,
    )

    embedding_writer.register_model(index_config=index_config)

    concept_filter_count = EmbeddingConceptFilter(
        require_standard=standard_only,
        domains=tuple(domains) if domains else None,
        vocabularies=tuple(vocabularies) if vocabularies else None,
    )

    total_missing = embedding_writer.get_concepts_without_embedding_count(
        concept_filter=concept_filter_count,
        index_type=index_type,
    )
    total_concepts = min(total_missing, num_embeddings) if num_embeddings is not None else total_missing

    concept_filter_embeddings = EmbeddingConceptFilter(
        require_standard=standard_only,
        domains=tuple(domains) if domains else None,
        vocabularies=tuple(vocabularies) if vocabularies else None,
        limit=total_concepts,
    )

    concepts_iterator = embedding_writer.get_concepts_without_embedding_batched(
        concept_filter=concept_filter_embeddings,
        index_type=index_type,
        batch_size=batch_size,
    )

    logger.info(f"Total concepts to process: {total_concepts}")
    with tqdm(total=total_concepts, desc="Processing", unit="concept") as pbar:
        for batch_concepts in concepts_iterator:
            embedding_writer.embed_and_upsert_concepts(
                concept_ids=tuple(batch_concepts.keys()),
                concept_texts=tuple(batch_concepts.values()),
                batch_size=batch_size,
                index_type=index_type,
            )
            pbar.update(len(batch_concepts))

    logger.info("Completed embedding generation and storage.")


@app.command()
def search(
    api_base: Annotated[str, typer.Option(
        "--api-base",
        help="Base URL for the API to use for generating embeddings.",
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
    index_type: Annotated[IndexType, typer.Option(
        "--index-type",
        help="Index type to query against. Must match the index used when the model was registered.",
        rich_help_panel="Index Options",
    )] = IndexType.FLAT,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size for query embedding generation and batched search.",
        rich_help_panel="Embedding API Options",
    )] = 100,
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Embedding model name.",
        rich_help_panel="Embedding API Options",
    )] = "text-embedding-3-small",
    metric_type: Annotated[MetricType, typer.Option(
        "--metric-type",
        help="Similarity or distance metric for nearest-neighbor search.",
        rich_help_panel="Search Options",
    )] = MetricType.COSINE,
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
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)",
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()

    queries_generator = consolidate_queries(queries=queries, queries_file=queries_file)
    emb_engine = resolve_emb_engine()
    omop_cdm_engine = resolve_omop_cdm_engine()

    embedding_reader = EmbeddingReaderInterface(
        emb_engine=emb_engine,
        omop_cdm_engine=omop_cdm_engine,
        model=model,
        api_key=api_key,
        api_base=api_base,
    )

    embedding_client = EmbeddingClient(
        model=model,
        api_base=api_base,
        api_key=api_key,
        embedding_batch_size=batch_size,
    )

    concept_filter = EmbeddingConceptFilter(
        require_standard=standard_only,
        domains=tuple(domains) if domains else None,
        vocabularies=tuple(vocabularies) if vocabularies else None,
        limit=k,
    )

    for batch_id, batched_queries in enumerate(itertools.batched(
        queries_generator,
        batch_size,
    )):
        batched_matches = embedding_reader.get_nearest_concepts_from_query_texts(
            query_texts=batched_queries,
            metric_type=metric_type,
            concept_filter=concept_filter,
            embedding_client=embedding_client,
            index_type=index_type,
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
