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
    get_embedding_model
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

    select_concept_query = (
        sa.select(
            Concept.concept_id,
            Concept.concept_name,
        )
        .where(Concept.concept_id.notin_(
            sa.select(EmbModelType.concept_id)
        ))
    )

    total_concepts_query = (
        sa.select(
            sa.func.count()
        ).select_from(Concept)
        .where(Concept.concept_id.notin_(
            sa.select(EmbModelType.concept_id)
        ))
    )

    if num_embeddings is not None:
        select_concept_query = select_concept_query.limit(num_embeddings)

    logger.info(f"Generating and storing embeddings for concepts. Using batch_size {batch_size}")
    with Session(engine) as reader, Session(engine) as writer:
        result = reader.execute(select_concept_query)
        total_concepts = reader.execute(total_concepts_query).scalar()
        if num_embeddings is not None:
            assert isinstance(total_concepts, int), " Expected total_concepts to be an integer"
            total_concepts = min(total_concepts, num_embeddings)

        logger.info(f"Total concepts to process: {total_concepts}")
        pbar = tqdm(total=total_concepts, desc="Processing concept embeddings")
        for row_chunk in result.partitions(batch_size):
            concept_ids = [row.concept_id for row in row_chunk]
            concept_names = [row.concept_name for row in row_chunk]
            embeddings = embedding_client.embeddings(concept_names)
            
            insert_values = [
                {
                    EmbModelType.concept_id.key: cid,
                    EmbModelType.embedding.key: emb.tolist(),
                }
                for cid, emb in zip(concept_ids, embeddings)
            ]

            # Define the upsert
            stmt = insert(EmbModelType).values(insert_values)
            upsert_stmt = stmt.on_conflict_do_update(
                index_elements=[EmbModelType.concept_id.key], 
                set_={EmbModelType.embedding.key: stmt.excluded.embedding}
            )

            # Execute on the WRITER
            writer.execute(upsert_stmt)
            writer.commit()
            pbar.update(len(insert_values))
    pbar.close()
    logger.info("Completed embedding generation and storage.")
    


if __name__ == "__main__":
    app()