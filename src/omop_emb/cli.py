import sqlalchemy as sa
from sqlalchemy.orm import Session

from orm_loader.helpers import get_logger, configure_logging

from typing import Annotated, Optional
import os
from dotenv import load_dotenv
from tqdm import tqdm
import typer

from omop_emb.backends import (
    EmbeddingConceptFilter,
)
from omop_emb.embedding_client import OpenAICompatibleEmbeddingClient
from omop_emb.interface import EmbeddingInterface
from omop_emb.backends.config import IndexType, MetricType
from omop_emb.backends.registry import get_metadata_schema

app = typer.Typer()
logger = get_logger(__name__)


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
    faiss_base_dir: Optional[str],
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
        faiss_base_dir=faiss_base_dir,
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

    faiss_base_dir: Annotated[Optional[str], typer.Option(
        "--faiss-base-dir",
        help="Optional base directory for FAISS backend storage."
    )] = None,
    standard_only: Annotated[bool, typer.Option(
        "--standard-only",
        help="If set, only generate embeddings for OMOP standard concepts (standard_concept = 'S')."
    )] = False,
    vocabularies: Annotated[Optional[list[str]], typer.Option(
        "--vocabulary",
        help="Optional vocabulary filter. Repeat the option to embed concepts only from specific OMOP vocabularies."
    )] = None,
    num_embeddings: Annotated[Optional[int], typer.Option(
        "--num-embeddings", "-n",
        help="If set, limits the number of concepts for which embeddings are generated. Useful for testing and development to speed up the embedding generation step.")] = None,
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
        faiss_base_dir=faiss_base_dir,
        embedding_path=embedding_path,
    )
    resolved_embedding_dim = _resolve_embedding_dim(interface, embedding_dim)

    concept_filter = EmbeddingConceptFilter(
        require_standard=standard_only,
        vocabularies=tuple(vocabularies) if vocabularies else None,
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
            overwrite_existing_conflicts=overwrite_model_registration,
        )

        estimated_total_concepts = num_embeddings or interface.get_concepts_without_embedding_count(
            session=reader,
            model_name=resolved_model,
            concept_filter=concept_filter,
        )
        concepts_without_embedding = interface.get_concepts_without_embedding_query(
            session=reader,
            model_name=resolved_model,
            concept_filter=concept_filter,
            limit=num_embeddings,
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
                )
                
                processed_concepts += len(batch_concepts)
                pbar.update(len(batch_concepts))

    logger.info("Completed embedding generation and storage. Wrote %s embeddings.", processed_concepts)


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
        faiss_base_dir=faiss_base_dir,
        embedding_path=embedding_path,
    )
    concept_filter = EmbeddingConceptFilter(
        require_standard=standard_only,
        vocabularies=tuple(vocabularies) if vocabularies else None,
    )

    interface.initialise_store(engine)

    with Session(engine) as session:
        query_embeddings = interface.embed_texts(query_text, batch_size=batch_size)
        matches = interface.backend.get_nearest_concepts(
            session=session,
            model_name=resolved_model,
            query_embeddings=query_embeddings,
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
        faiss_base_dir=faiss_base_dir,
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


if __name__ == "__main__":
    app()
