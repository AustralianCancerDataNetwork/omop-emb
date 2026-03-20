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
    index_method: Annotated[str, typer.Option(
        "--index-method",
        help="Vector index backend to use for new embedding tables. Options: auto, diskann, hnsw, ivfflat, none. Defaults to OMOP_EMB_INDEX_METHOD or auto."
    )] = "auto",
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
    
    # Re-init tables
    create_db(engine)
    initialize_embedding_tables(engine)

    if engine.dialect.name != 'postgresql':
        logger.warning(
            "Embedding generation is currently only supported for PostgreSQL "
            f"with pgvector or pgvectorscale. Detected dialect: {engine.dialect.name}. Skipping embedding generation."
        )
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
        index_method=index_method,
    )
    EmbModelType = get_embedding_model(model)
    missing_embedding_clause = ~sa.exists(
        sa.select(sa.literal(1))
        .select_from(EmbModelType)
        .where(EmbModelType.concept_id == Concept.concept_id)
    )

    select_concept_query = (
        sa.select(
            Concept.concept_id,
            Concept.concept_name,
        )
        .where(missing_embedding_clause)
    )

    if standard_only:
        select_concept_query = select_concept_query.where(Concept.standard_concept == "S")

    if vocabularies:
        select_concept_query = select_concept_query.where(Concept.vocabulary_id.in_(vocabularies))

    if num_embeddings is not None:
        select_concept_query = select_concept_query.limit(num_embeddings)

    logger.info(
        "Generating and storing embeddings for concepts. Using batch_size %s (standard_only=%s, vocabularies=%s)",
        batch_size,
        standard_only,
        vocabularies,
    )
    with Session(engine) as reader, Session(engine) as writer:
        pbar = tqdm(
            total=num_embeddings,
            desc="Processing concept embeddings",
            unit="concept",
            dynamic_ncols=True,
        )
        pbar.set_postfix_str("querying")
        result = reader.execute(select_concept_query)
        processed_concepts = 0
        for row_chunk in result.partitions(batch_size):
            concept_ids = [row.concept_id for row in row_chunk]
            concept_names = [row.concept_name for row in row_chunk]
            pbar.set_postfix_str(f"embedding batch={len(concept_ids)}")
            embeddings = embedding_client.embeddings(concept_names)
            
            add_embeddings_to_registered_table(
                session=writer,
                concept_ids=tuple(concept_ids),
                embeddings=embeddings,
                model=EmbModelType
            )
            processed_concepts += len(concept_ids)
            pbar.update(len(concept_ids))
        if pbar.total is not None and pbar.total != processed_concepts:
            pbar.total = processed_concepts
            pbar.refresh()
    pbar.close()
    logger.info("Completed embedding generation and storage.")
    


if __name__ == "__main__":
    app()
