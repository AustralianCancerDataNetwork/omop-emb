from omop_llm import LLMClient

import sqlalchemy as sa
from sqlalchemy.orm import Session

from orm_loader.helpers import get_logger, configure_logging, create_db
from omop_alchemy.cdm.model.vocabulary import Concept

from typing import Annotated, Optional
import os
from dotenv import load_dotenv
from tqdm import tqdm
import typer

from omop_emb.backends import (
    EmbeddingBackendConfigurationError,
)
from omop_emb.interface import EmbeddingInterface
from omop_emb.backends.config import BackendType, IndexType

app = typer.Typer()
logger = get_logger(__name__)


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
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Name of the embedding model to use for generating concept embeddings (e.g., 'text-embedding-3-small'). If not provided, embeddings will not be generated.")] = "text-embedding-3-small",
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

    interface = EmbeddingInterface.from_backend_name(
        backend_name=backend_name,
        faiss_base_dir=faiss_base_dir,
        embedding_client=LLMClient(
            model=model,
            api_base=api_base,
            api_key=api_key,
            embedding_batch_size=batch_size
        ),
    )
    
    # Ensure OMOP metadata tables exist, then initialize the embedding store.
    create_db(engine)
    interface.initialise_store(engine)

    with Session(engine) as reader, Session(engine) as writer:
        interface.ensure_model_registered(
            engine=engine,
            session=reader,
            model_name=model,
            index_type=index_type,
            dimensions=interface.embedding_dim
        )

        total_concepts = num_embeddings or interface.get_concepts_without_embedding_count(
            session=reader,
            model_name=model,
            require_standard=standard_only,
            vocabularies=vocabularies,
        )
        concepts_without_embedding = interface.get_concepts_without_embedding_query(
            session=reader,
            model_name=model,
            require_standard=standard_only,
            vocabularies=vocabularies,
            limit=num_embeddings,
        )

        logger.info(f"Total concepts to process: {total_concepts}")
        with tqdm(total=total_concepts, desc="Processing", unit="concept") as pbar:
            result = reader.execute(concepts_without_embedding)
            
            for row_chunk in result.partitions(batch_size):
                batch_concepts = {row.concept_id: row.concept_name for row in row_chunk}
                
                interface.embed_and_upsert_concepts(
                    session=writer,
                    model_name=model,
                    concept_ids=tuple(batch_concepts.keys()),
                    concept_texts=tuple(batch_concepts.values()),
                    batch_size=batch_size,
                )
                
                pbar.update(len(batch_concepts))

    logger.info("Completed embedding generation and storage.")


if __name__ == "__main__":
    app()
