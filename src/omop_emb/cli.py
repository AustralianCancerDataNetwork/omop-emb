from omop_llm import LLMClient

import sqlalchemy as sa
from sqlalchemy.orm import Session

from orm_loader.helpers import get_logger, configure_logging, create_db

from typing import Annotated, Optional
import os
from dotenv import load_dotenv
from tqdm import tqdm
import typer

from omop_emb.backends import (
    EmbeddingConceptFilter,
)
from omop_emb.interface import EmbeddingInterface
from omop_emb.backends.config import IndexType

app = typer.Typer()
logger = get_logger(__name__)


def _resolve_model_name(model: Optional[str]) -> str:
    resolved_model = model or os.getenv("OMOP_EMB_MODEL") or "text-embedding-3-small"
    if not resolved_model:
        raise RuntimeError(
            "No embedding model configured. Pass `--model` or set `OMOP_EMB_MODEL`."
        )
    return resolved_model


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


@app.command()
def add_embeddings(
    api_base: Annotated[str, typer.Option(
        "--api-base",
        help="Base URL for the API to use for generating embeddings.",
    )],
    api_key: Annotated[str, typer.Option(
        "--api-key",
        help="API key for the embedding API.")],
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

    engine_string = os.getenv('OMOP_DATABASE_URL')
    if engine_string is None:
        raise RuntimeError("OMOP_DATABASE_URL environment variable not set. Please set it in your .env file to point to your database.")
    
    engine = sa.create_engine(engine_string, future=True, echo=False)
    assert engine.dialect.name == "postgresql", "Only PostgreSQL databases are supported for embedding storage with the current backends. Please check your `OMOP_DATABASE_URL` environment variable and ensure it points to a PostgreSQL database."
    resolved_model = _resolve_model_name(model)

    interface = EmbeddingInterface.from_backend_name(
        backend_name=backend_name,
        faiss_base_dir=faiss_base_dir,
        embedding_client=LLMClient(
            model=resolved_model,
            api_base=api_base,
            api_key=api_key,
            embedding_batch_size=batch_size
        ),
    )
    resolved_embedding_dim = _resolve_embedding_dim(interface, embedding_dim)

    concept_filter = EmbeddingConceptFilter(
        require_standard=standard_only,
        vocabularies=tuple(vocabularies) if vocabularies else None,
    )
    
    # Ensure OMOP metadata tables exist, then initialize the embedding store.
    create_db(engine)
    interface.initialise_store(engine)

    with Session(engine) as reader, Session(engine) as writer:
        interface.ensure_model_registered(
            engine=engine,
            session=reader,
            model_name=resolved_model,
            index_type=index_type,
            dimensions=resolved_embedding_dim
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


if __name__ == "__main__":
    app()
