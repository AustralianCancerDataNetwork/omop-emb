import sqlalchemy as sa
from sqlalchemy.orm import Session

from orm_loader.helpers import get_logger, configure_logging

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import time
from typing import Annotated, Optional, Sequence
import csv
import os
import logging
from dotenv import load_dotenv
from tqdm import tqdm
import typer

from omop_emb.utils.embedding_utils import EmbeddingConceptFilter
from omop_emb.embedding_client import OpenAICompatibleEmbeddingClient
from omop_emb.interface import EmbeddingInterface
from omop_emb.config import BackendType, IndexType, MetricType
from omop_emb.backends.faiss.faiss_backend import (
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_HNSW_NUM_NEIGHBORS,
    FaissEmbeddingBackend,
    build_faiss_index_metadata,
)
from omop_emb.model_registry import get_metadata_schema

app = typer.Typer()
logger = get_logger(__name__)

SNAPSHOT_MANIFEST_NAME = "manifest.json"


def configure_logging_level(verbosity: int) -> None:
    """Configure global logging based on CLI verbosity flags."""
    level_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    log_level = level_map.get(min(verbosity, 2), logging.DEBUG)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _load_legacy_rows(engine: sa.Engine, legacy_table: str) -> list[dict[str, object]]:
    inspector = sa.inspect(engine)
    if not inspector.has_table(legacy_table):
        raise RuntimeError(f"Legacy table '{legacy_table}' does not exist in source database.")

    query = sa.text(f'SELECT * FROM "{legacy_table}"')
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    return [dict(row) for row in rows]


def _legacy_row_fields(row: dict[str, object]) -> tuple[str, int, IndexType, str, dict[str, object]]:
    model_name_raw = row.get("model_name")
    dimensions_raw = row.get("dimensions")
    if model_name_raw is None or dimensions_raw is None:
        raise ValueError("Legacy row missing required fields 'model_name' and/or 'dimensions'.")

    index_raw = row.get("index_type") or row.get("index_method") or IndexType.FLAT.value
    index_type = IndexType(str(index_raw))

    storage_identifier_raw = row.get("storage_identifier") or row.get("table_name")
    if storage_identifier_raw is None:
        raise ValueError("Legacy row missing required storage field ('storage_identifier' or 'table_name').")
    storage_identifier = str(storage_identifier_raw)

    details_raw = row.get("details") or row.get("metadata") or {}
    if isinstance(details_raw, dict):
        metadata = dict(details_raw)
    else:
        metadata = {}

    model_name = str(model_name_raw)
    dimensions = int(str(dimensions_raw))

    return model_name, dimensions, index_type, storage_identifier, metadata


def _resolve_engine() -> sa.Engine:
    engine_string = os.getenv('OMOP_DATABASE_URL')
    if engine_string is None:
        raise RuntimeError("OMOP_DATABASE_URL environment variable not set. Please set it in your .env file to point to your database.")

    engine = sa.create_engine(engine_string, future=True, echo=False)
    assert engine.dialect.name == "postgresql", "Only PostgreSQL databases are supported for embedding storage with the current backends. Please check your `OMOP_DATABASE_URL` environment variable and ensure it points to a PostgreSQL database."
    return engine


def _build_pgvector_interface(storage_base_dir: Optional[str]) -> EmbeddingInterface:
    interface = EmbeddingInterface.from_backend_name(
        backend_name=BackendType.PGVECTOR,
        storage_base_dir=storage_base_dir,
    )
    if interface.backend.backend_type != BackendType.PGVECTOR:
        raise RuntimeError("Resolved embedding backend is not pgvector. Set --storage-base-dir and/or OMOP_EMB_BACKEND appropriately.")
    return interface


def _log_timing(stage: str, started_at: float) -> float:
    elapsed = time.monotonic() - started_at
    logger.info("Timing: %s completed in %.3fs", stage, elapsed)
    return elapsed


def _resolve_model_name(model: Optional[str]) -> str:
    resolved_model = model or os.getenv("OMOP_EMB_MODEL") or "text-embedding-3-small"
    if not resolved_model:
        raise RuntimeError(
            "No embedding model configured. Pass `--model` or set `OMOP_EMB_MODEL`."
        )
    return resolved_model


def _resolve_api_key(api_key: Optional[str]) -> Optional[str]:
    resolved_api_key = api_key
    if resolved_api_key is None:
        resolved_api_key = os.getenv("OMOP_EMB_API_KEY")
    if resolved_api_key == "":
        return None
    return resolved_api_key


def _resolve_embedding_dim(
    interface: EmbeddingInterface,
    embedding_dim: Optional[int],
) -> int:
    if embedding_dim is None:
        raw_dim = os.getenv("OMOP_EMB_EMBEDDING_DIM")
        if raw_dim:
            try:
                embedding_dim = int(raw_dim)
            except ValueError as exc:
                raise RuntimeError(
                    "OMOP_EMB_EMBEDDING_DIM must be an integer."
                ) from exc

    if embedding_dim is not None:
        if embedding_dim <= 0:
            raise RuntimeError("Embedding dimension override must be a positive integer.")
        return embedding_dim

    try:
        resolved_dim = interface.embedding_dim
    except NotImplementedError as exc:
        raise RuntimeError(
            "Embedding dimension could not be inferred from the configured API client. "
            "Pass `--embedding-dim` or set `OMOP_EMB_EMBEDDING_DIM`."
        ) from exc

    if resolved_dim is None:
        raise RuntimeError(
            "Embedding dimensions could not be determined from the embedding client. "
            "Pass `--embedding-dim` or set `OMOP_EMB_EMBEDDING_DIM`."
        )
    return resolved_dim


def _normalize_embedding_path(embedding_path: str) -> str:
    normalized = embedding_path.strip()
    if not normalized:
        raise RuntimeError("Embedding path must not be empty.")
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized


def _normalize_api_base(api_base: str, embedding_path: str) -> str:
    normalized = api_base.rstrip("/")
    normalized_path = _normalize_embedding_path(embedding_path)
    if normalized.endswith(normalized_path):
        normalized = normalized[: -len(normalized_path)]
        logger.warning(
            "API base ended with the embedding path `%s`. Normalizing to `%s` so the configured path is not duplicated.",
            normalized_path,
            normalized,
        )
    return normalized


def _build_concept_filter(
    *,
    standard_only: bool,
    vocabularies: Optional[list[str]],
) -> Optional[EmbeddingConceptFilter]:
    resolved_vocabularies = tuple(vocabularies) if vocabularies else None
    if not standard_only and resolved_vocabularies is None:
        return None
    return EmbeddingConceptFilter(
        require_standard=standard_only,
        vocabularies=resolved_vocabularies,
    )


def _build_backend_metadata(
    *,
    backend_type: BackendType,
    index_type: IndexType,
    existing_metadata: Optional[dict[str, object]] = None,
    hnsw_num_neighbors: Optional[int] = None,
    hnsw_ef_search: Optional[int] = None,
    hnsw_ef_construction: Optional[int] = None,
) -> dict[str, object]:
    if backend_type != BackendType.FAISS:
        return dict(existing_metadata or {})
    return build_faiss_index_metadata(
        index_type=index_type,
        existing_metadata=existing_metadata,
        hnsw_num_neighbors=hnsw_num_neighbors,
        hnsw_ef_search=hnsw_ef_search,
        hnsw_ef_construction=hnsw_ef_construction,
    )


def _compile_sql(query: sa.Select, engine: sa.Engine) -> str:
    try:
        return str(
            query.compile(
                dialect=engine.dialect,
                compile_kwargs={"literal_binds": True},
            )
        )
    except Exception:
        return str(query)


def _create_engine_from_env() -> sa.Engine:
    engine_string = os.getenv('OMOP_DATABASE_URL')
    if engine_string is None:
        raise RuntimeError("OMOP_DATABASE_URL environment variable not set. Please set it in your .env file to point to your database.")

    engine = sa.create_engine(engine_string, future=True, echo=False)
    assert engine.dialect.name == "postgresql", "Only PostgreSQL databases are supported for embedding storage with the current backends. Please check your `OMOP_DATABASE_URL` environment variable and ensure it points to a PostgreSQL database."
    return engine


def _create_embedding_interface(
    *,
    api_base: str,
    api_key: Optional[str],
    batch_size: int,
    model: Optional[str],
    backend_name: Optional[str],
    storage_base_dir: Optional[str],
    embedding_path: str,
) -> tuple[EmbeddingInterface, str]:
    resolved_embedding_path = _normalize_embedding_path(
        os.getenv("OMOP_EMB_EMBEDDING_PATH") or embedding_path
    )
    resolved_api_base = _normalize_api_base(api_base, resolved_embedding_path)
    resolved_api_key = _resolve_api_key(api_key)
    resolved_model = _resolve_model_name(model)

    interface = EmbeddingInterface.from_backend_name(
        backend_name=backend_name,
        storage_base_dir=storage_base_dir,
        embedding_client=OpenAICompatibleEmbeddingClient(
            model=resolved_model,
            api_base=resolved_api_base,
            api_key=resolved_api_key,
            embedding_batch_size=batch_size,
            embedding_path=resolved_embedding_path,
            encoding_format="float",
        ),
    )
    return interface, resolved_model


def _load_batch_queries(query_file: str) -> list[tuple[str, str]]:
    loaded_queries: list[tuple[str, str]] = []
    with Path(query_file).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" in line:
                query_id, query_text = line.split("\t", 1)
            else:
                query_id = str(line_number)
                query_text = line
            query_text = query_text.strip()
            if not query_text:
                continue
            loaded_queries.append((query_id.strip() or str(line_number), query_text))
    if not loaded_queries:
        raise RuntimeError(f"No non-empty queries found in {query_file}.")
    return loaded_queries


def _render_search_results(
    *,
    query_id: str,
    query_text: str,
    matches: Sequence[object],
) -> list[str]:
    if not matches:
        return [f"{query_id}\t{query_text}\t0\t\t\t"]
    rendered_rows: list[str] = []
    for rank, match in enumerate(matches, start=1):
        rendered_rows.append(
            f"{query_id}\t{query_text}\t{rank}\t{match.concept_id}\t{match.similarity:.6f}\t{match.concept_name}"
        )
    return rendered_rows


def _warm_search_backend(
    *,
    interface: EmbeddingInterface,
    session: Session,
    model_name: str,
    metric_type: MetricType,
) -> None:
    backend = interface.backend
    if backend.backend_type != BackendType.FAISS:
        return
    matching_models = backend.embedding_model_registry.get_registered_models_from_db(
        backend_type=backend.backend_type,
        model_name=model_name,
    )
    if not matching_models:
        raise RuntimeError(f"Embedding model '{model_name}' is not registered.")
    if len(matching_models) != 1:
        raise RuntimeError(
            f"Expected exactly one registered model for '{model_name}', found "
            f"{len(matching_models)}. Narrow the model selection or clean up stale registrations."
        )
    model_record = matching_models[0]
    storage_manager = backend.get_storage_manager(
        model_name=model_name,
        dimensions=model_record.dimensions,
        index_type=model_record.index_type,
        metadata=model_record.metadata,
    )
    started_at = time.monotonic()
    storage_manager.get_index_manager(
        index_type=model_record.index_type,
        metric_type=metric_type,
    )
    _log_timing(
        f"warm faiss index model={model_name} metric={metric_type.value}",
        started_at,
    )


def _run_search_query(
    *,
    interface: EmbeddingInterface,
    session: Session,
    model_name: str,
    query_texts: Sequence[str],
    batch_size: int,
    metric_type: MetricType,
    concept_filter: Optional[EmbeddingConceptFilter],
    k: int,
) -> tuple[tuple[object, ...], ...]:
    total_started_at = time.monotonic()
    embed_started_at = time.monotonic()
    backend = interface.backend
    matching_models = backend.embedding_model_registry.get_registered_models_from_db(
        backend_type=backend.backend_type,
        model_name=model_name,
    )
    if not matching_models:
        raise RuntimeError(f"Embedding model '{model_name}' is not registered.")
    if len(matching_models) != 1:
        raise RuntimeError(
            f"Expected exactly one registered model for '{model_name}', found "
            f"{len(matching_models)}. Narrow the model selection or clean up stale registrations."
        )
    model_record = matching_models[0]

    query_embeddings = interface.embed_texts(
        list(query_texts),
        batch_size=batch_size,
        text_role="query",
    )
    _log_timing("embed query text", embed_started_at)

    search_started_at = time.monotonic()
    matches = backend.get_nearest_concepts(
        session=session,
        model_name=model_name,
        index_type=model_record.index_type,
        query_embedding=query_embeddings,
        metric_type=metric_type,
        concept_filter=concept_filter,
    )
    _log_timing("nearest-concept lookup", search_started_at)
    _log_timing("full search request", total_started_at)
    return tuple(tuple(match for match in matches_per_query[:k]) for matches_per_query in matches)


def _search_response_payload(
    *,
    query_id: str,
    query_text: str,
    matches: Sequence[object],
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query_text": query_text,
        "matches": [
            {
                "rank": rank,
                "concept_id": int(match.concept_id),
                "similarity": float(match.similarity),
                "concept_name": match.concept_name,
            }
            for rank, match in enumerate(matches, start=1)
        ],
    }


@app.command()
def add_embeddings(
    api_base: Annotated[str, typer.Option(
        "--api-base",
        help="Base URL for the API to use for generating embeddings.",
    )],
    api_key: Annotated[Optional[str], typer.Option(
        "--api-key",
        help="Optional API key for the embedding API. Can also be set with `OMOP_EMB_API_KEY`."
    )] = None,
    index_type: Annotated[IndexType, typer.Option(
        "--index-type",
        help="Backend-specific index type for newly registered models and how it should be stored.",
    )] = IndexType.FLAT,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size to use when generating and inserting embeddings. Adjust based on your system's memory capacity.")] = 100,
    model: Annotated[Optional[str], typer.Option(
        "--model", "-m",
        help="Name of the embedding model to use for generating concept embeddings. Can also be set with `OMOP_EMB_MODEL`."
    )] = None,
    embedding_dim: Annotated[Optional[int], typer.Option(
        "--embedding-dim",
        help="Explicit embedding dimension for the configured model. Use this for OpenAI-compatible endpoints when the client cannot infer dimensions automatically. Can also be set with `OMOP_EMB_EMBEDDING_DIM`."
    )] = None,
    embedding_path: Annotated[str, typer.Option(
        "--embedding-path",
        help="Embedding endpoint path relative to `--api-base`, for example `/embeddings` or `/embed`. Can also be set with `OMOP_EMB_EMBEDDING_PATH`."
    )] = "/embeddings",
    overwrite_model_registration: Annotated[bool, typer.Option(
        "--overwrite-model-registration",
        help="If set, delete any existing registration and backend storage for this model name before re-registering it. Use with care when changing dimensions or endpoint behavior."
    )] = False,
    backend_name: Annotated[Optional[str], typer.Option(
        "--backend",
        help="Embedding backend to use. Can be replaced by the `OMOP_EMB_BACKEND` environment variable."
    )] = None,
    storage_base_dir: Annotated[Optional[str], typer.Option(
        "--storage-base-dir",
        help="Optional base directory for embedding backend storage. Reverts to `OMOP_EMB_BASE_STORAGE_DIR` if not provided, or defaults to ./.omop_emb in the current working directory. Paths with `~` are expanded."
    )] = None,
    faiss_base_dir: Annotated[Optional[str], typer.Option(
        "--faiss-base-dir",
        help="Deprecated alias for `--storage-base-dir` retained for compatibility."
    )] = None,
    hnsw_num_neighbors: Annotated[int, typer.Option(
        "--hnsw-num-neighbors",
        help="FAISS HNSW M parameter. Used when `--index-type hnsw`."
    )] = DEFAULT_HNSW_NUM_NEIGHBORS,
    hnsw_ef_search: Annotated[int, typer.Option(
        "--hnsw-ef-search",
        help="FAISS HNSW efSearch parameter. Used when `--index-type hnsw`."
    )] = DEFAULT_HNSW_EF_SEARCH,
    hnsw_ef_construction: Annotated[int, typer.Option(
        "--hnsw-ef-construction",
        help="FAISS HNSW efConstruction parameter. Used when `--index-type hnsw`."
    )] = DEFAULT_HNSW_EF_CONSTRUCTION,
    standard_only: Annotated[bool, typer.Option(
        "--standard-only",
        help="If set, only generate embeddings for OMOP standard concepts (standard_concept = 'S')."
    )] = False,
    vocabularies: Annotated[Optional[list[str]], typer.Option(
        "--vocabulary",
        help="Optional vocabulary filter. Repeat the option to embed concepts only from specific OMOP vocabularies."
    )] = None,
    domains: Annotated[Optional[list[str]], typer.Option(
        "--domain",
        help="Optional domain filter. Repeat the option to embed concepts only from specific OMOP domains."
    )] = None,
    num_embeddings: Annotated[Optional[int], typer.Option(
        "--num-embeddings", "-n",
        help="If set, limits the number of concepts for which embeddings are generated. Useful for testing and development to speed up the embedding generation step.")] = None,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)"
    )] = 0,
):
    configure_logging_level(verbosity)
    load_dotenv()
    engine = _create_engine_from_env()
    logger.info("Embedding metadata schema: %s", get_metadata_schema())
    interface, resolved_model = _create_embedding_interface(
        api_base=api_base,
        api_key=api_key,
        batch_size=batch_size,
        model=model,
        backend_name=backend_name,
        storage_base_dir=storage_base_dir or faiss_base_dir,
        embedding_path=embedding_path,
    )
    resolved_embedding_dim = _resolve_embedding_dim(interface, embedding_dim)
    backend_metadata = _build_backend_metadata(
        backend_type=interface.backend.backend_type,
        index_type=index_type,
        hnsw_num_neighbors=hnsw_num_neighbors,
        hnsw_ef_search=hnsw_ef_search,
        hnsw_ef_construction=hnsw_ef_construction,
    )

    concept_filter = _build_concept_filter(
        standard_only=standard_only,
        vocabularies=vocabularies,
    )
    
    # Initialize only the embedding store metadata for this project.
    interface.initialise_store(engine)

    with Session(engine) as reader, Session(engine) as writer:
        interface.ensure_model_registered(
            engine=engine,
            session=reader,
            model_name=resolved_model,
            index_type=index_type,
            dimensions=resolved_embedding_dim,
            metadata=backend_metadata,
            overwrite_existing_conflicts=overwrite_model_registration,
        )

        estimated_total_concepts = num_embeddings or interface.get_concepts_without_embedding_count(
            session=reader,
            model_name=resolved_model,
            concept_filter=concept_filter,
            index_type=index_type
        )
        concepts_without_embedding = interface.get_concepts_without_embedding_query(
            session=reader,
            model_name=resolved_model,
            concept_filter=concept_filter,
            limit=num_embeddings,
            index_type=index_type
        )
        actual_total_concepts = reader.scalar(
            sa.select(sa.func.count()).select_from(concepts_without_embedding.subquery())
        )

        logger.info(
            "Embedding source query: %s",
            _compile_sql(concepts_without_embedding, engine),
        )
        if num_embeddings is not None:
            logger.info(
                "Requested up to %s concepts; query returned %s concepts.",
                num_embeddings,
                actual_total_concepts,
            )
        else:
            logger.info("Query returned %s concepts.", actual_total_concepts)

        logger.info(
            "Progress bar total: %s concepts.",
            actual_total_concepts if num_embeddings is not None else estimated_total_concepts,
        )
        processed_concepts = 0
        with tqdm(
            total=actual_total_concepts if num_embeddings is not None else estimated_total_concepts,
            desc="Processing",
            unit="concept",
        ) as pbar:
            result = reader.execute(concepts_without_embedding)
            
            for row_chunk in result.partitions(batch_size):
                batch_concepts = {row.concept_id: row.concept_name for row in row_chunk}
                if not batch_concepts:
                    continue
                
                interface.embed_and_upsert_concepts(
                    session=writer,
                    model_name=resolved_model,
                    concept_ids=tuple(batch_concepts.keys()),
                    concept_texts=tuple(batch_concepts.values()),
                    batch_size=batch_size,
                    index_type=index_type
                )
                
                processed_concepts += len(batch_concepts)
                pbar.update(len(batch_concepts))

    logger.info("Completed embedding generation and storage. Wrote %s embeddings.", processed_concepts)
    if interface.backend.backend_type == BackendType.FAISS:
        typer.echo(
            "FAISS note: raw embeddings were written to HDF5 storage. "
            "Rebuild the metric-specific FAISS index before search, for example: "
            f"omop-emb rebuild-index --model {resolved_model} --backend faiss --metric-type cosine"
        )


@app.command()
def search(
    query_text: Annotated[str, typer.Argument(
        help="Text to embed and search against stored concept embeddings."
    )],
    api_base: Annotated[str, typer.Option(
        "--api-base",
        help="Base URL for the API to use for generating query embeddings.",
    )],
    api_key: Annotated[Optional[str], typer.Option(
        "--api-key",
        help="Optional API key for the embedding API. Can also be set with `OMOP_EMB_API_KEY`."
    )] = None,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size to use when generating query embeddings."
    )] = 32,
    model: Annotated[Optional[str], typer.Option(
        "--model", "-m",
        help="Name of the embedding model to query."
    )] = None,
    embedding_path: Annotated[str, typer.Option(
        "--embedding-path",
        help="Embedding endpoint path relative to `--api-base`, for example `/embeddings` or `/embed`."
    )] = "/embeddings",
    backend_name: Annotated[Optional[str], typer.Option(
        "--backend",
        help="Embedding backend to query. Can be replaced by the `OMOP_EMB_BACKEND` environment variable."
    )] = None,
    faiss_base_dir: Annotated[Optional[str], typer.Option(
        "--faiss-base-dir",
        help="Optional base directory for FAISS backend storage."
    )] = None,
    metric_type: Annotated[MetricType, typer.Option(
        "--metric-type",
        help="Similarity or distance metric to use for nearest-neighbor search."
    )] = MetricType.COSINE,
    k: Annotated[int, typer.Option(
        "--k",
        help="Number of nearest concepts to return."
    )] = 10,
    standard_only: Annotated[bool, typer.Option(
        "--standard-only",
        help="If set, only search within OMOP standard concepts."
    )] = False,
    vocabularies: Annotated[Optional[list[str]], typer.Option(
        "--vocabulary",
        help="Optional vocabulary filter. Repeat to restrict the search space."
    )] = None,
):
    configure_logging()
    load_dotenv()

    engine = _create_engine_from_env()
    logger.info("Embedding metadata schema: %s", get_metadata_schema())
    interface, resolved_model = _create_embedding_interface(
        api_base=api_base,
        api_key=api_key,
        batch_size=batch_size,
        model=model,
        backend_name=backend_name,
        storage_base_dir=faiss_base_dir,
        embedding_path=embedding_path,
    )
    concept_filter = _build_concept_filter(
        standard_only=standard_only,
        vocabularies=vocabularies,
    )

    interface.initialise_store(engine)

    with Session(engine) as session:
        matches = _run_search_query(
            interface=interface,
            session=session,
            model_name=resolved_model,
            query_texts=(query_text,),
            batch_size=batch_size,
            metric_type=metric_type,
            concept_filter=concept_filter,
            k=k,
        )

    typer.echo("rank\tconcept_id\tsimilarity\tconcept_name")
    if not matches or not matches[0]:
        logger.info("No search results found for model '%s'.", resolved_model)
        return

    for rank, match in enumerate(matches[0], start=1):
        typer.echo(
            f"{rank}\t{match.concept_id}\t{match.similarity:.6f}\t{match.concept_name}"
        )


@app.command("search-batch")
def search_batch(
    query_file: Annotated[str, typer.Argument(
        help="Path to a UTF-8 text file containing one query per line, or `query_id<TAB>query_text` per line."
    )],
    api_base: Annotated[str, typer.Option(
        "--api-base",
        help="Base URL for the API to use for generating query embeddings.",
    )],
    api_key: Annotated[Optional[str], typer.Option(
        "--api-key",
        help="Optional API key for the embedding API. Can also be set with `OMOP_EMB_API_KEY`."
    )] = None,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size to use when generating query embeddings and executing batched search."
    )] = 32,
    model: Annotated[Optional[str], typer.Option(
        "--model", "-m",
        help="Name of the embedding model to query."
    )] = None,
    embedding_path: Annotated[str, typer.Option(
        "--embedding-path",
        help="Embedding endpoint path relative to `--api-base`, for example `/embeddings` or `/embed`."
    )] = "/embeddings",
    backend_name: Annotated[Optional[str], typer.Option(
        "--backend",
        help="Embedding backend to query. Can be replaced by the `OMOP_EMB_BACKEND` environment variable."
    )] = None,
    faiss_base_dir: Annotated[Optional[str], typer.Option(
        "--faiss-base-dir",
        help="Optional base directory for FAISS backend storage."
    )] = None,
    metric_type: Annotated[MetricType, typer.Option(
        "--metric-type",
        help="Similarity or distance metric to use for nearest-neighbor search."
    )] = MetricType.COSINE,
    k: Annotated[int, typer.Option(
        "--k",
        help="Number of nearest concepts to return."
    )] = 10,
    standard_only: Annotated[bool, typer.Option(
        "--standard-only",
        help="If set, only search within OMOP standard concepts."
    )] = False,
    vocabularies: Annotated[Optional[list[str]], typer.Option(
        "--vocabulary",
        help="Optional vocabulary filter. Repeat to restrict the search space."
    )] = None,
    warm_index: Annotated[bool, typer.Option(
        "--warm-index/--no-warm-index",
        help="Preload the FAISS index once before processing the batch."
    )] = True,
):
    configure_logging()
    load_dotenv()

    queries = _load_batch_queries(query_file)
    engine = _create_engine_from_env()
    logger.info("Embedding metadata schema: %s", get_metadata_schema())
    interface, resolved_model = _create_embedding_interface(
        api_base=api_base,
        api_key=api_key,
        batch_size=batch_size,
        model=model,
        backend_name=backend_name,
        storage_base_dir=faiss_base_dir,
        embedding_path=embedding_path,
    )
    concept_filter = _build_concept_filter(
        standard_only=standard_only,
        vocabularies=vocabularies,
    )
    interface.initialise_store(engine)

    typer.echo("query_id\tquery_text\trank\tconcept_id\tsimilarity\tconcept_name")
    with Session(engine) as session:
        if warm_index:
            _warm_search_backend(
                interface=interface,
                session=session,
                model_name=resolved_model,
                metric_type=metric_type,
            )

        for batch_start in range(0, len(queries), batch_size):
            query_batch = queries[batch_start:batch_start + batch_size]
            matches_batch = _run_search_query(
                interface=interface,
                session=session,
                model_name=resolved_model,
                query_texts=[query_text for _, query_text in query_batch],
                batch_size=batch_size,
                metric_type=metric_type,
                concept_filter=concept_filter,
                k=k,
            )
            for (query_id, query_text), matches_per_query in zip(query_batch, matches_batch):
                for row in _render_search_results(
                    query_id=query_id,
                    query_text=query_text,
                    matches=matches_per_query,
                ):
                    typer.echo(row)


@app.command("serve-search")
def serve_search(
    api_base: Annotated[str, typer.Option(
        "--api-base",
        help="Base URL for the API to use for generating query embeddings.",
    )],
    api_key: Annotated[Optional[str], typer.Option(
        "--api-key",
        help="Optional API key for the embedding API. Can also be set with `OMOP_EMB_API_KEY`."
    )] = None,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size to use when generating query embeddings."
    )] = 32,
    model: Annotated[Optional[str], typer.Option(
        "--model", "-m",
        help="Name of the embedding model to query."
    )] = None,
    embedding_path: Annotated[str, typer.Option(
        "--embedding-path",
        help="Embedding endpoint path relative to `--api-base`, for example `/embeddings` or `/embed`."
    )] = "/embeddings",
    backend_name: Annotated[Optional[str], typer.Option(
        "--backend",
        help="Embedding backend to query. Can be replaced by the `OMOP_EMB_BACKEND` environment variable."
    )] = None,
    faiss_base_dir: Annotated[Optional[str], typer.Option(
        "--faiss-base-dir",
        help="Optional base directory for FAISS backend storage."
    )] = None,
    metric_type: Annotated[MetricType, typer.Option(
        "--metric-type",
        help="Similarity or distance metric to use for nearest-neighbor search."
    )] = MetricType.COSINE,
    k: Annotated[int, typer.Option(
        "--k",
        help="Number of nearest concepts to return."
    )] = 10,
    standard_only: Annotated[bool, typer.Option(
        "--standard-only",
        help="If set, only search within OMOP standard concepts."
    )] = False,
    vocabularies: Annotated[Optional[list[str]], typer.Option(
        "--vocabulary",
        help="Optional vocabulary filter. Repeat to restrict the search space."
    )] = None,
    host: Annotated[str, typer.Option(
        "--host",
        help="Host interface for the search service."
    )] = "127.0.0.1",
    port: Annotated[int, typer.Option(
        "--port",
        help="Port for the search service."
    )] = 8080,
    warm_index: Annotated[bool, typer.Option(
        "--warm-index/--no-warm-index",
        help="Preload the FAISS index before accepting requests."
    )] = True,
):
    configure_logging()
    load_dotenv()

    engine = _create_engine_from_env()
    logger.info("Embedding metadata schema: %s", get_metadata_schema())
    interface, resolved_model = _create_embedding_interface(
        api_base=api_base,
        api_key=api_key,
        batch_size=batch_size,
        model=model,
        backend_name=backend_name,
        storage_base_dir=faiss_base_dir,
        embedding_path=embedding_path,
    )
    concept_filter = _build_concept_filter(
        standard_only=standard_only,
        vocabularies=vocabularies,
    )
    interface.initialise_store(engine)

    if warm_index:
        with Session(engine) as warm_session:
            _warm_search_backend(
                interface=interface,
                session=warm_session,
                model_name=resolved_model,
                metric_type=metric_type,
            )

    class SearchHandler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "model": resolved_model,
                        "backend": interface.backend.backend_name,
                        "metric_type": metric_type.value,
                    },
                )
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/search":
                self._send_json(404, {"error": "not found"})
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            try:
                payload = json.loads(raw_body or b"{}")
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid json"})
                return

            query_text = payload.get("query_text")
            if not isinstance(query_text, str) or not query_text.strip():
                self._send_json(400, {"error": "`query_text` must be a non-empty string"})
                return

            request_k = payload.get("k", k)
            if not isinstance(request_k, int) or request_k <= 0:
                self._send_json(400, {"error": "`k` must be a positive integer"})
                return

            with Session(engine) as session:
                matches = _run_search_query(
                    interface=interface,
                    session=session,
                    model_name=resolved_model,
                    query_texts=(query_text,),
                    batch_size=batch_size,
                    metric_type=metric_type,
                    concept_filter=concept_filter,
                    k=request_k,
                )
            self._send_json(
                200,
                _search_response_payload(
                    query_id=str(payload.get("query_id", "1")),
                    query_text=query_text,
                    matches=matches[0] if matches else (),
                ),
            )

        def log_message(self, format: str, *args: object) -> None:
            logger.info("search-service " + format, *args)

    server = HTTPServer((host, port), SearchHandler)
    typer.echo(
        f"Serving search on http://{host}:{port} using model '{resolved_model}', backend '{interface.backend.backend_name}', metric '{metric_type.value}'."
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping search service.")
    finally:
        server.server_close()


@app.command("rebuild-index")
def rebuild_index(
    model: Annotated[Optional[str], typer.Option(
        "--model", "-m",
        help="Registered embedding model name to rebuild indexes for."
    )] = None,
    backend_name: Annotated[Optional[str], typer.Option(
        "--backend",
        help="Embedding backend to rebuild. Currently only FAISS supports explicit rebuild."
    )] = None,
    faiss_base_dir: Annotated[Optional[str], typer.Option(
        "--faiss-base-dir",
        help="Optional base directory for FAISS backend storage."
    )] = None,
    metric_types: Annotated[Optional[list[MetricType]], typer.Option(
        "--metric-type",
        help="Metric(s) to rebuild. Repeat to rebuild multiple metrics. Defaults to all metrics supported by the model's index type."
    )] = None,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size to use when streaming embeddings from disk during rebuild."
    )] = 100_000,
):
    configure_logging()
    load_dotenv()

    engine = _create_engine_from_env()
    logger.info("Embedding metadata schema: %s", get_metadata_schema())
    interface = EmbeddingInterface.from_backend_name(
        backend_name=backend_name,
        storage_base_dir=faiss_base_dir,
    )
    resolved_model = _resolve_model_name(model)
    interface.initialise_store(engine)

    with Session(engine) as session:
        interface.rebuild_model_indexes(
            session=session,
            model_name=resolved_model,
            metric_types=tuple(metric_types) if metric_types else None,
            batch_size=batch_size,
        )

    logger.info("Completed index rebuild for model '%s'.", resolved_model)


@app.command("switch-index-type")
def switch_index_type(
    model: Annotated[Optional[str], typer.Option(
        "--model", "-m",
        help="Registered embedding model name to update."
    )] = None,
    backend_name: Annotated[Optional[str], typer.Option(
        "--backend",
        help="Embedding backend to update. Currently only FAISS supports index switching."
    )] = None,
    faiss_base_dir: Annotated[Optional[str], typer.Option(
        "--faiss-base-dir",
        help="Optional base directory for FAISS backend storage."
    )] = None,
    index_type: Annotated[IndexType, typer.Option(
        "--index-type",
        help="New FAISS index type to store in the model registry."
    )] = IndexType.HNSW,
    metric_types: Annotated[Optional[list[MetricType]], typer.Option(
        "--metric-type",
        help="Metric(s) to rebuild after switching. Defaults to all metrics supported by the new index type."
    )] = None,
    hnsw_num_neighbors: Annotated[int, typer.Option(
        "--hnsw-num-neighbors",
        help="FAISS HNSW M parameter. Used when `--index-type hnsw`."
    )] = DEFAULT_HNSW_NUM_NEIGHBORS,
    hnsw_ef_search: Annotated[int, typer.Option(
        "--hnsw-ef-search",
        help="FAISS HNSW efSearch parameter. Used when `--index-type hnsw`."
    )] = DEFAULT_HNSW_EF_SEARCH,
    hnsw_ef_construction: Annotated[int, typer.Option(
        "--hnsw-ef-construction",
        help="FAISS HNSW efConstruction parameter. Used when `--index-type hnsw`."
    )] = DEFAULT_HNSW_EF_CONSTRUCTION,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size to use when streaming embeddings from disk during rebuild."
    )] = 100_000,
    rebuild: Annotated[bool, typer.Option(
        "--rebuild/--no-rebuild",
        help="If set, rebuild FAISS index files immediately after updating the registry."
    )] = True,
):
    configure_logging()
    load_dotenv()

    engine = _create_engine_from_env()
    logger.info("Embedding metadata schema: %s", get_metadata_schema())
    interface = EmbeddingInterface.from_backend_name(
        backend_name=backend_name,
        storage_base_dir=faiss_base_dir,
    )
    resolved_model = _resolve_model_name(model)
    interface.initialise_store(engine)

    if interface.backend.backend_type != BackendType.FAISS or not isinstance(interface.backend, FaissEmbeddingBackend):
        raise RuntimeError("Index-type switching is currently only supported for the FAISS backend.")

    with Session(engine) as session:
        matching_models = interface.backend.embedding_model_registry.get_registered_models_from_db(
            backend_type=interface.backend.backend_type,
            model_name=resolved_model,
        )
        if not matching_models:
            raise RuntimeError(f"Embedding model '{resolved_model}' is not registered.")
        if len(matching_models) != 1:
            raise RuntimeError(
                f"Expected exactly one registered model for '{resolved_model}', found "
                f"{len(matching_models)}. Narrow the model selection or clean up stale registrations."
            )
        existing_model = matching_models[0]

        metadata = _build_backend_metadata(
            backend_type=interface.backend.backend_type,
            index_type=index_type,
            existing_metadata=dict(existing_model.metadata),
            hnsw_num_neighbors=hnsw_num_neighbors,
            hnsw_ef_search=hnsw_ef_search,
            hnsw_ef_construction=hnsw_ef_construction,
        )
        interface.backend.update_model_index_configuration(
            session=session,
            model_name=resolved_model,
            index_type=index_type,
            metadata=metadata,
        )
        if rebuild:
            interface.rebuild_model_indexes(
                session=session,
                model_name=resolved_model,
                metric_types=tuple(metric_types) if metric_types else None,
                batch_size=batch_size,
            )

    typer.echo(
        f"Updated model '{resolved_model}' to index_type={index_type.value}."
        + (" Rebuilt FAISS index files." if rebuild else "")
    )


@app.command()
def export_pgvector(
    output_dir: Annotated[str, typer.Option(
        "--output-dir", "-o",
        help="Directory where pgvector snapshot files (CSV + manifest) are written.",
    )],
    storage_base_dir: Annotated[Optional[str], typer.Option(
        "--storage-base-dir",
        help="Optional base directory for omop-emb metadata registry (metadata.db). Reverts to `OMOP_EMB_BASE_STORAGE_DIR` if not provided, or defaults to ./.omop_emb in the current working directory. Paths with `~` are expanded.",
    )] = None,
    model: Annotated[Optional[list[str]], typer.Option(
        "--model", "-m",
        help="Optional model-name filter. Repeat to export specific models only.",
    )] = None,
    index_type: Annotated[Optional[IndexType], typer.Option(
        "--index-type",
        help="Optional index-type filter.",
    )] = None,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)"
    )] = 0,
):
    """Export pgvector embedding tables to file for checkpoint/restore workflows."""
    configure_logging_level(verbosity)
    load_dotenv()

    engine = _resolve_engine()
    interface = _build_pgvector_interface(storage_base_dir=storage_base_dir)
    interface.initialise_store(engine)

    records = interface.backend.embedding_model_registry.get_registered_models_from_db(
        backend_type=BackendType.PGVECTOR
    )
    if records is None:
        raise RuntimeError("No pgvector models found in local registry metadata. Nothing to export.")

    model_filter = set(model) if model else None
    selected_records = [
        record for record in records
        if (model_filter is None or record.model_name in model_filter)
        and (index_type is None or record.index_type == index_type)
    ]
    if not selected_records:
        raise RuntimeError("No pgvector models matched the provided filters.")

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format_version": 1,
        "backend": BackendType.PGVECTOR.value,
        "tables": [],
    }

    with engine.connect() as conn:
        for record in selected_records:
            table_name = record.storage_identifier
            quoted_table_name = f'"{table_name}"'
            row_count = conn.execute(sa.text(f"SELECT COUNT(*) FROM {quoted_table_name}"))
            n_rows = int(row_count.scalar_one())

            csv_filename = f"{table_name}.csv"
            csv_path = output_path / csv_filename

            query = sa.text(
                f"SELECT concept_id, embedding::text AS embedding FROM {quoted_table_name} ORDER BY concept_id"
            )
            result = conn.execution_options(stream_results=True).execute(query)

            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["concept_id", "embedding"])
                for row in result:
                    writer.writerow([int(row.concept_id), str(row.embedding)])

            logger.info(f"Exported {n_rows} rows from {table_name} to {csv_path}")
            manifest["tables"].append(
                {
                    "model_name": record.model_name,
                    "index_type": record.index_type.value,
                    "dimensions": record.dimensions,
                    "storage_identifier": record.storage_identifier,
                    "metadata": dict(record.metadata),
                    "rows": n_rows,
                    "file": csv_filename,
                }
            )

    manifest_path = output_path / SNAPSHOT_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Wrote pgvector snapshot manifest to {manifest_path}")


@app.command()
def import_pgvector(
    input_dir: Annotated[str, typer.Option(
        "--input-dir", "-i",
        help="Directory containing pgvector snapshot files and manifest.json.",
    )],
    storage_base_dir: Annotated[Optional[str], typer.Option(
        "--storage-base-dir",
        help="Optional base directory for omop-emb metadata registry (metadata.db). Reverts to `OMOP_EMB_BASE_STORAGE_DIR` if not provided, or defaults to ./.omop_emb in the current working directory. Paths with `~` are expanded.",
    )] = None,
    replace: Annotated[bool, typer.Option(
        "--replace",
        help="If set, truncate each destination embedding table before import.",
    )] = False,
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Number of rows per INSERT batch.",
    )] = 5000,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)"
    )] = 0,
):
    """Import pgvector embedding tables from a previously exported snapshot."""
    configure_logging_level(verbosity)
    load_dotenv()

    input_path = Path(input_dir).resolve()
    manifest_path = input_path / SNAPSHOT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Snapshot manifest not found at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("backend") != BackendType.PGVECTOR.value:
        raise ValueError("Snapshot manifest backend is not pgvector.")

    table_specs = manifest.get("tables")
    if not isinstance(table_specs, list) or not table_specs:
        raise ValueError("Snapshot manifest does not contain any tables.")

    engine = _resolve_engine()
    interface = _build_pgvector_interface(storage_base_dir=storage_base_dir)

    for table_spec in table_specs:
        model_name = str(table_spec["model_name"])
        index_type_enum = IndexType(str(table_spec["index_type"]))
        dimensions = int(table_spec["dimensions"])
        expected_table_name = str(table_spec["storage_identifier"])
        metadata = table_spec.get("metadata") or {}

        interface.register_model(
            engine=engine,
            model_name=model_name,
            dimensions=dimensions,
            index_type=index_type_enum,
            metadata=metadata,
        )
        actual_table_name = interface.get_model_table_name(
            model_name=model_name,
            index_type=index_type_enum,
        )
        if actual_table_name != expected_table_name:
            raise RuntimeError(
                f"Storage identifier mismatch for model '{model_name}': expected '{expected_table_name}', got '{actual_table_name}'."
            )

    interface.initialise_store(engine)

    with Session(engine) as session:
        for table_spec in table_specs:
            table_name = str(table_spec["storage_identifier"])
            csv_file = input_path / str(table_spec["file"])
            if not csv_file.is_file():
                raise FileNotFoundError(f"Missing snapshot table file: {csv_file}")

            quoted_table_name = f'"{table_name}"'
            if replace:
                session.execute(sa.text(f"TRUNCATE TABLE {quoted_table_name}"))
                session.commit()

            insert_sql = sa.text(
                f"""
                INSERT INTO {quoted_table_name} (concept_id, embedding)
                VALUES (:concept_id, CAST(:embedding AS vector))
                ON CONFLICT (concept_id) DO UPDATE
                SET embedding = EXCLUDED.embedding
                """
            )

            total_rows = 0
            batch: list[dict[str, object]] = []
            with csv_file.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    batch.append(
                        {
                            "concept_id": int(row["concept_id"]),
                            "embedding": row["embedding"],
                        }
                    )
                    if len(batch) >= batch_size:
                        session.execute(insert_sql, batch)
                        session.commit()
                        total_rows += len(batch)
                        batch = []

                if batch:
                    session.execute(insert_sql, batch)
                    session.commit()
                    total_rows += len(batch)

            logger.info(f"Imported {total_rows} rows into {table_name} from {csv_file}")


@app.command()
def migrate_legacy_pgvector_registry(
    storage_base_dir: Annotated[Optional[str], typer.Option(
        "--storage-base-dir",
        help="Optional base directory for omop-emb metadata registry (metadata.db). Reverts to `OMOP_EMB_BASE_STORAGE_DIR` if not provided, or defaults to ./.omop_emb in the current working directory. Paths with `~` are expanded.",
    )] = None,
    source_database_url: Annotated[Optional[str], typer.Option(
        "--source-database-url",
        help="Source database URL containing the legacy model_registry table. Defaults to OMOP_DATABASE_URL.",
    )] = None,
    legacy_table: Annotated[str, typer.Option(
        "--legacy-table",
        help="Name of the legacy table to migrate from.",
    )] = "model_registry",
    dry_run: Annotated[bool, typer.Option(
        "--dry-run",
        help="If set, report rows that would be migrated without writing to local metadata.",
    )] = False,
    drop_legacy_registry: Annotated[bool, typer.Option(
        "--drop-legacy-registry",
        help="If set, drop the legacy table after successful migration.",
    )] = False,
    verbosity: Annotated[int, typer.Option(
        "--verbose", "-v", count=True,
        help="Increase verbosity (up to two levels)"
    )] = 0,
):
    """Migrate legacy pgvector registry rows into local metadata.db registry."""
    configure_logging_level(verbosity)
    load_dotenv()

    source_url = source_database_url or os.getenv("OMOP_DATABASE_URL")
    if source_url is None:
        raise RuntimeError("OMOP_DATABASE_URL is not set. Provide --source-database-url.")

    source_engine = sa.create_engine(source_url, future=True, echo=False)
    interface = _build_pgvector_interface(storage_base_dir=storage_base_dir)

    legacy_rows = _load_legacy_rows(source_engine, legacy_table=legacy_table)
    if not legacy_rows:
        logger.info("No legacy registry rows found. Nothing to migrate.")
        return

    logger.info(f"Found {len(legacy_rows)} legacy registry rows in '{legacy_table}'.")

    migrated = 0
    for row in legacy_rows:
        model_name, dimensions, index_type, storage_identifier, metadata = _legacy_row_fields(row)

        if dry_run:
            logger.info(
                f"[DRY RUN] Would migrate model={model_name}, index={index_type.value}, "
                f"dimensions={dimensions}, storage_identifier={storage_identifier}"
            )
            migrated += 1
            continue

        interface.backend.embedding_model_registry.register_model(
            model_name=model_name,
            dimensions=dimensions,
            backend_type=BackendType.PGVECTOR,
            index_type=index_type,
            metadata=metadata,
            storage_identifier=storage_identifier,
        )
        migrated += 1

    logger.info(f"Migrated {migrated} legacy registry rows into local metadata registry.")

    if drop_legacy_registry and not dry_run:
        with source_engine.begin() as conn:
            conn.execute(sa.text(f'DROP TABLE IF EXISTS "{legacy_table}"'))
        logger.info(f"Dropped legacy registry table '{legacy_table}'.")


if __name__ == "__main__":
    app()
