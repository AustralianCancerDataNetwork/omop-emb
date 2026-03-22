from omop_llm import LLMClient

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import insert

from orm_loader.helpers import get_logger, configure_logging, create_db
from omop_alchemy.cdm.model.vocabulary import Concept

from typing import Annotated, Optional
import os
from dotenv import load_dotenv
from tqdm import tqdm
import typer

from omop_emb.cdm.embeddings import (
    initialize_embedding_tables, 
    register_new_model, 
    get_embedding_model,
    add_embeddings_to_registered_table
)

from omop_emb.queries import (
    q_get_concepts_without_embedding,
    q_count_missing_concepts
)

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
    batch_size: Annotated[int, typer.Option(
        "--batch-size", "-b",
        help="Batch size to use when generating and inserting embeddings. Adjust based on your system's memory capacity.")] = 100,
    model: Annotated[str, typer.Option(
        "--model", "-m",
        help="Name of the embedding model to use for generating concept embeddings (e.g., 'text-embedding-3-small'). If not provided, embeddings will not be generated.")] = "text-embedding-3-small",
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
    
    # Re-init tables
    create_db(engine)
    initialize_embedding_tables(engine)

    if engine.dialect.name != 'postgresql':
        logger.warning(f"Embedding generation is currently only supported for PostgreSQL with pgvectorscale. Detected dialect: {engine.dialect.name}. Skipping embedding generation.")
        return
    
    # Prepare the database for storing embeddings
    embedding_client = LLMClient(
        model=model,
        api_base=api_base,
        api_key=api_key,
        embedding_batch_size=batch_size
    )

    register_new_model(
        engine=engine,
        model_name=model,
        dimensions=embedding_client.embedding_dim,
    )
    EmbModelType = get_embedding_model(model)
    
    
    logger.info(f"Generating and storing embeddings for concepts. Using batch_size {batch_size}")
    with Session(engine) as reader, Session(engine) as writer:
        concepts_without_embeddings = reader.execute(q_get_concepts_without_embedding(embedding_table=EmbModelType, limit=num_embeddings))
        num_concepts_without_embeddings = reader.execute(q_count_missing_concepts(embedding_table=EmbModelType)).scalar() if num_embeddings is None else num_embeddings

        logger.info(f"Total concepts to process: {num_concepts_without_embeddings}")
        pbar = tqdm(total=num_concepts_without_embeddings, desc="Processing concept embeddings")
        for row_chunk in concepts_without_embeddings.partitions(batch_size):
            concept_ids = [row.concept_id for row in row_chunk]
            concept_names = [row.concept_name for row in row_chunk]
            embeddings = embedding_client.embeddings(concept_names)
            
            add_embeddings_to_registered_table(
                session=writer,
                concept_ids=tuple(concept_ids),
                embeddings=embeddings,
                model=EmbModelType
            )
            pbar.update(len(concept_ids))
    pbar.close()
    logger.info("Completed embedding generation and storage.")
    


if __name__ == "__main__":
    app()