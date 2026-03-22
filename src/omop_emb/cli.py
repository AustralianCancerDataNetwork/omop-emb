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
    EmbeddingIndexConfig,
    EmbeddingBackendConfigurationError,
    normalize_backend_name,
)
from omop_emb.service import EmbeddingService

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
    backend_name: Annotated[Optional[str], typer.Option(
        "--backend",
        help="Embedding backend to use. Defaults to OMOP_EMB_BACKEND or the package default."
    )] = None,
    index_method: Annotated[str, typer.Option(
        "--index-method",
        help="Backend-specific index type for newly registered models. For PostgreSQL this maps to pgvector index settings. For the current FAISS backend, auto maps to IndexFlatIP."
    )] = "auto",
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
    resolved_backend_name = normalize_backend_name(backend_name)
    service = EmbeddingService.from_backend_name(
        backend_name=resolved_backend_name,
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
    service.initialise_store(engine)

    if resolved_backend_name == 'postgres' and engine.dialect.name != 'postgresql':
        logger.warning(
            "Postgres embedding backend requested, but the configured SQLAlchemy "
            f"dialect is {engine.dialect.name}. Skipping embedding generation."
        )
        return

    backend_index_type = index_method
    if resolved_backend_name == "faiss" and index_method == "auto":
        backend_index_type = "IndexFlatIP"

    select_concept_query = (
        sa.select(
            Concept.concept_id,
            Concept.concept_name,
        )
    )

    if standard_only:
        select_concept_query = select_concept_query.where(Concept.standard_concept == "S")

    if vocabularies:
        select_concept_query = select_concept_query.where(Concept.vocabulary_id.in_(vocabularies))

    logger.info(
        "Generating and storing embeddings for concepts. Using batch_size %s (standard_only=%s, vocabularies=%s)",
        batch_size,
        standard_only,
        vocabularies,
    )
    with Session(engine) as reader, Session(engine) as writer:
        try:
            service.ensure_model_registered(
                engine=engine,
                session=reader,
                model_name=model,
                index_config=EmbeddingIndexConfig(
                    index_type=backend_index_type,
                    distance_metric="cosine",
                ),
            )
        except EmbeddingBackendConfigurationError as exc:
            requested = backend_index_type
            raise typer.BadParameter(
                f"{exc} Requested backend={resolved_backend_name!r}, "
                f"model={model!r}, index_method={requested!r}.",
                param_hint="--index-method",
            ) from exc
        total_concepts = _count_missing_concepts(
            reader=reader,
            service=service,
            model_name=model,
            select_concept_query=select_concept_query,
            batch_size=batch_size,
            max_missing=num_embeddings,
        )

        logger.info(f"Total concepts to process: {total_concepts}")
        pbar = tqdm(
            total=total_concepts,
            desc="Processing concept embeddings",
            unit="concept",
            dynamic_ncols=True,
        )
        pbar.set_postfix_str("querying")
        result = reader.execute(select_concept_query)
        processed_concepts = 0
        for row_chunk in result.partitions(batch_size):
            row_by_id = {int(row.concept_id): row for row in row_chunk}
            candidate_ids = tuple(row_by_id.keys())
            missing_ids = list(service.get_missing_concept_ids(
                session=reader,
                model_name=model,
                concept_ids=candidate_ids,
            ))
            if not missing_ids:
                continue

            if num_embeddings is not None:
                remaining = num_embeddings - processed_concepts
                if remaining <= 0:
                    break
                missing_ids = missing_ids[:remaining]

            concept_names = [row_by_id[concept_id].concept_name for concept_id in missing_ids]
            pbar.set_postfix_str(f"embedding batch={len(missing_ids)}")
            service.embed_and_upsert_concepts(
                session=writer,
                model_name=model,
                concept_ids=tuple(missing_ids),
                concept_texts=concept_names,
                batch_size=batch_size,
            )
            processed_concepts += len(missing_ids)
            pbar.update(len(missing_ids))
            if num_embeddings is not None and processed_concepts >= num_embeddings:
                break
        if pbar.total is not None and pbar.total != processed_concepts:
            pbar.total = processed_concepts
            pbar.refresh()
    pbar.close()
    logger.info("Completed embedding generation and storage.")
    


def _count_missing_concepts(
    *,
    reader: Session,
    service: EmbeddingService,
    model_name: str,
    select_concept_query,
    batch_size: int,
    max_missing: Optional[int] = None,
) -> int:
    """
    Backend-neutral precount for concepts without stored embeddings.

    This deliberately uses backend lookup rather than a backend-specific SQL
    anti-join so the same CLI flow can work across multiple storage engines.
    """

    missing = 0
    result = reader.execute(select_concept_query)
    for row_chunk in result.partitions(batch_size):
        concept_ids = tuple(int(row.concept_id) for row in row_chunk)
        missing += len(service.get_missing_concept_ids(
            session=reader,
            model_name=model_name,
            concept_ids=concept_ids,
        ))
        if max_missing is not None and missing >= max_missing:
            return max_missing
    return missing


if __name__ == "__main__":
    app()
