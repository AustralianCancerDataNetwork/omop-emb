"""Embedding-related CLI commands for omop-emb."""

from dotenv import load_dotenv
import itertools
import logging
logger = logging.getLogger(__name__)
from sqlalchemy.orm import Session
from tqdm import tqdm
from typing import Annotated, Optional, List, Union, Generator, Sequence
import typer
app = typer.Typer(help="Commands related to embedding generation, storage, and management.")

from orm_loader.helpers import create_db

from .utils import (
    configure_logging_level,
    resolve_engine
)
from omop_emb.backends.index_config import index_config_from_index_type

from omop_emb.config import (
    IndexType,
    MetricType,
    BackendType,
    ENV_OMOP_EMB_BACKEND,
    ENV_BASE_STORAGE_DIR,
)
from omop_emb.embeddings import EmbeddingClient
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter, NearestConceptMatch
from omop_emb.interface import EmbeddingWriterInterface, EmbeddingReaderInterface

def consolidate_queries(
    queries: Optional[Union[str, List[str]]],
    queries_file: Optional[str]
) -> Generator[str, None, None]:
    
    if queries is not None and queries_file is not None:
        raise ValueError("Cannot provide both --queries and --queries-file. Please choose one method for providing queries.")

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
        help="Backend-specific index type for newly registered models and how it should be stored.",
        rich_help_panel="Index Options",
    )] = IndexType.FLAT,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size to use when generating and inserting embeddings.",
        rich_help_panel="Embedding API Options",
        )] = 100,
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Name of the embedding model to use for generating concept embeddings (e.g., 'text-embedding-3-small'). If not provided, embeddings will not be generated.",
        rich_help_panel="Embedding API Options",
    )] = "text-embedding-3-small",
    backend_type: Annotated[Optional[BackendType], typer.Option(
        "--backend",
        help=f"Embedding backend to use. Can be replaced by the `{ENV_OMOP_EMB_BACKEND}` environment variable.",
        rich_help_panel="Storage Options",
    )] = None,
    storage_base_dir: Annotated[Optional[str], typer.Option(
        "--storage-base-dir",
        help=f"Optional base directory for embedding backend storage. Can be set using `{ENV_BASE_STORAGE_DIR}` environment variable. Defaults to `~/.omop_emb`. Paths with `~` are expanded.",
        rich_help_panel="Storage Options"
    )] = None,
    standard_only: Annotated[bool, typer.Option(
        "--standard-only",
        help="If set, only generate embeddings for OMOP standard concepts (standard_concept = 'S').",
        rich_help_panel="Concept Filters"
    )] = False,
    vocabularies: Annotated[Optional[list[str]], typer.Option(
        "--vocabulary",
        help="Optional vocabulary filter. Repeat the option to embed concepts only from specific OMOP vocabularies.",
        rich_help_panel="Concept Filters"
    )] = None,
    domains: Annotated[Optional[list[str]], typer.Option(
        "--domain",
        help="Optional domain filter. Repeat the option to embed concepts only from specific OMOP domains.",
        rich_help_panel="Concept Filters",
    )] = None,
    num_embeddings: Annotated[Optional[int], typer.Option(
        "--num-embeddings", "-n",
        help="If set, limits the number of concepts for which embeddings are generated. Useful for testing and development to speed up the embedding generation step.",
        rich_help_panel="Concept Filters"
    )] = None,
    index_hnsw_num_neighbors: Annotated[Optional[int], typer.Option(
        "--index-hnsw-num-neighbors",
        help="HNSW: number of neighbors per graph node. Required when --index-type is HNSW.",
        rich_help_panel="Index Options",
    )] = None,
    index_hnsw_ef_search: Annotated[Optional[int], typer.Option(
        "--index-hnsw-ef-search",
        help="HNSW: ef parameter controlling recall during search. Required when --index-type is HNSW.",
        rich_help_panel="Index Options",
    )] = None,
    index_hnsw_ef_construction: Annotated[Optional[int], typer.Option(
        "--index-ef-construction",
        help="HNSW: ef parameter controlling graph quality during construction. Required when --index-type is HNSW.",
        rich_help_panel="Index Options",
    )] = None,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)"
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()

    engine = resolve_engine()

    embedding_client = EmbeddingClient(
        model=model,
        api_base=api_base,
        api_key=api_key,
        embedding_batch_size=batch_size
    )
    embedding_writer = EmbeddingWriterInterface(
        embedding_client=embedding_client,
        backend_name_or_type=backend_type,
        storage_base_dir=storage_base_dir,
    )

    concept_filter = EmbeddingConceptFilter(
        require_standard=standard_only,
        domains=tuple(domains) if domains else None,
        vocabularies=tuple(vocabularies) if vocabularies else None,
    )

    index_kwargs = {
        'num_neighbors': index_hnsw_num_neighbors,
        'ef_search': index_hnsw_ef_search,
        'ef_construction': index_hnsw_ef_construction,
    }

    index_config = index_config_from_index_type(
        index_type,
        **index_kwargs
    )

    # Ensure OMOP metadata tables exist, then initialize the embedding store.
    create_db(engine)
    embedding_writer.register_model(
        engine=engine,
        index_config=index_config,
    )

    with Session(engine) as reader, Session(engine) as writer:
        total_concepts_missing_concepts = embedding_writer.get_concepts_without_embedding_count(
            session=reader,
            concept_filter=concept_filter,
            index_type=index_type,
        )
        total_concepts = min(total_concepts_missing_concepts, num_embeddings) if num_embeddings is not None else total_concepts_missing_concepts

        concepts_without_embedding = embedding_writer.q_get_concepts_without_embedding(
            concept_filter=concept_filter,
            limit=total_concepts,
            index_type=index_type
        )

        logger.info(f"Total concepts to process: {total_concepts}")
        with tqdm(total=total_concepts, desc="Processing", unit="concept") as pbar:
            result = reader.execute(concepts_without_embedding)

            for row_chunk in result.partitions(batch_size):
                batch_concepts = {row.concept_id: row.concept_name for row in row_chunk}

                embedding_writer.embed_and_upsert_concepts(
                    session=writer,
                    concept_ids=tuple(batch_concepts.keys()),
                    concept_texts=tuple(batch_concepts.values()),
                    batch_size=batch_size,
                    index_type=index_type
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
        help="Backend-specific index type to query against. Must match the index type used when the model was registered and the embeddings were stored.",
        rich_help_panel="Index Options",
    )] = IndexType.FLAT,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size to use when generating query embeddings and executing batched search.",
        rich_help_panel="Embedding API Options",
        )] = 100,
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Name of the embedding model to use for generating concept embeddings (e.g., 'text-embedding-3-small'). If not provided, embeddings will not be generated.",
        rich_help_panel="Embedding API Options",
    )] = "text-embedding-3-small",
    backend_type: Annotated[Optional[BackendType], typer.Option(
        "--backend",
        help=f"Embedding backend to use. Can be replaced by the `{ENV_OMOP_EMB_BACKEND}` environment variable.",
        rich_help_panel="Storage Options",
    )] = None,
    storage_base_dir: Annotated[Optional[str], typer.Option(
        "--storage-base-dir",
        help=f"Optional base directory for embedding backend storage. Can be set using `{ENV_BASE_STORAGE_DIR}` environment variable. Defaults to `~/.omop_emb`. Paths with `~` are expanded.",
        rich_help_panel="Storage Options"
    )] = None,
    metric_type: Annotated[MetricType, typer.Option(
        "--metric-type",
        help="Similarity or distance metric to use for nearest-neighbor search.",
        rich_help_panel="Search Options"
    )] = MetricType.COSINE,
    k: Annotated[int, typer.Option(
        "--k",
        help="Number of nearest concepts to return.",
        rich_help_panel="Search Options"
    )] = 10,
    standard_only: Annotated[bool, typer.Option(
        "--standard-only",
        help="If set, only generate embeddings for OMOP standard concepts (standard_concept = 'S').",
        rich_help_panel="Concept Filters"
    )] = False,
    vocabularies: Annotated[Optional[list[str]], typer.Option(
        "--vocabulary",
        help="Optional vocabulary filter. Repeat the option to embed concepts only from specific OMOP vocabularies.",
        rich_help_panel="Concept Filters"
    )] = None,
    domains: Annotated[Optional[list[str]], typer.Option(
        "--domain",
        help="Optional domain filter. Repeat the option to embed concepts only from specific OMOP domains.",
        rich_help_panel="Concept Filters",
    )] = None,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)"
    )] = 0,
):
    configure_logging_level(verbosity)  
    load_dotenv()

    queries_generator = consolidate_queries(queries=queries, queries_file=queries_file)
    engine = resolve_engine()

    embedding_reader = EmbeddingReaderInterface(
        model=model,
        api_key=api_key,
        api_base=api_base,
        backend_name_or_type=backend_type,
        storage_base_dir=storage_base_dir,
    )

    embedding_client = EmbeddingClient(
        model=model,
        api_base=api_base,
        api_key=api_key,
        embedding_batch_size=batch_size
    )

    concept_filter = EmbeddingConceptFilter(
        require_standard=standard_only,
        domains=tuple(domains) if domains else None,
        vocabularies=tuple(vocabularies) if vocabularies else None,
        limit=k,
    )

    embedding_reader.initialise_store(engine)

    with Session(engine) as session:
        for batch_id, batched_queries in enumerate(itertools.batched(
            queries_generator,
            batch_size
        )): 
            batched_matches = embedding_reader.get_nearest_concepts_from_query_texts(
                session=session,
                query_texts=batched_queries,
                metric_type=metric_type,
                concept_filter=concept_filter,
                embedding_client=embedding_client,
                index_type=index_type
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
