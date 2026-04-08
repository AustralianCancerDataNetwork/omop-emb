from __future__ import annotations

from typing import Mapping, Optional, Sequence, Type, Tuple

from numpy import ndarray
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from .pgvector_sql import (
    q_embedding_nearest_concepts,
    q_embedding_vectors_by_concept_ids,
    PGVectorConceptIDEmbeddingTable,
    create_pg_embedding_table,
    add_embeddings_to_registered_table,
)
from omop_emb.config import BackendType, MetricType, IndexType
from ..base import EmbeddingBackend, require_registered_model
from omop_emb.utils.embedding_utils import (
    EmbeddingConceptFilter,
    NearestConceptMatch,
)
from omop_emb.model_registry import EmbeddingModelRecord

class PGVectorEmbeddingBackend(EmbeddingBackend[PGVectorConceptIDEmbeddingTable]):
    """
    pgvector-backed embedding backend for postgresql databases.

    This class is intentionally a thin structural layer over the existing
    ``omop_emb`` implementation. It is not wired into the current accessor or
    CLI yet, but it demonstrates how the present behavior maps onto the new
    backend abstraction.
    """

    @property
    def backend_type(self) -> BackendType:
        return BackendType.PGVECTOR

    def _create_storage_table(self, engine: Engine, model_record: EmbeddingModelRecord) -> Type[PGVectorConceptIDEmbeddingTable]:
        return create_pg_embedding_table(engine=engine, model_record=model_record)

    def pre_initialise_store(self, engine: Engine) -> None:
        with Session(engine, expire_on_commit=False) as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector CASCADE;"))
            session.commit()

    @require_registered_model
    def upsert_embeddings(
        self,
        model_name: str,
        index_type: IndexType,
        model_record: EmbeddingModelRecord,
        session: Session,
        concept_ids: Sequence[int],
        embeddings: ndarray,
    ) -> None:
        concept_id_tuple = tuple(concept_ids)

        self.validate_embeddings_and_concept_ids(
            concept_ids=concept_id_tuple,
            embeddings=embeddings,
            dimensions=model_record.dimensions,
        )

        table = self.get_embedding_table(
            model_name=model_name,
            index_type=index_type,
        )

        try:
            add_embeddings_to_registered_table(
                session=session,
                concept_ids=concept_id_tuple,
                embeddings=embeddings,
                registered_table=table,
            )
        except IntegrityError as e:
            session.rollback()
            raise ValueError(
                f"Failed to upsert embeddings for model '{model_name}'. This may be due to a mismatch between the provided concept_ids and the existing entries in the database. Original error: {str(e)}"
            ) from e

    @require_registered_model
    def get_embeddings_by_concept_ids(
        self,
        model_name: str,
        index_type: IndexType,
        model_record: EmbeddingModelRecord,
        session: Session,
        concept_ids: Sequence[int],
    ) -> Mapping[int, Sequence[float]]:
        concept_id_tuple = tuple(concept_ids)
        if not concept_id_tuple:
            return {}

        embedding_table = self.get_embedding_table(
            model_name=model_name,
            index_type=index_type,
        )
        query = q_embedding_vectors_by_concept_ids(
            embedding_table=embedding_table,
            concept_ids=concept_id_tuple,
        )

        return_dict = {
            int(row.concept_id): list(row.embedding)
            for row in session.execute(query)
        }

        missing_ids = set(concept_id_tuple) - set(return_dict.keys())
        if missing_ids:
            raise ValueError(
                f"Requested concept IDs {missing_ids} not found in the database for model '{model_name}'."
            )
        return return_dict

    @require_registered_model
    def get_nearest_concepts(
        self,
        model_name: str,
        index_type: IndexType,
        model_record: EmbeddingModelRecord,
        session: Session,
        query_embeddings: ndarray,
        *,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        k: int = 10,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        
        embedding_table = self.get_embedding_table(
            model_name=model_name,
            index_type=index_type,
        )
        self.validate_embeddings(embeddings=query_embeddings, dimensions=model_record.dimensions)

        query_list = query_embeddings.tolist()
        query = q_embedding_nearest_concepts(
            embedding_table=embedding_table,
            query_embeddings=query_list,
            metric_type=metric_type,
            concept_filter=concept_filter,
            limit=k
        )

        rows = session.execute(query).all()
        results = [[] for _ in range(len(query_list))]

        for row in rows:
            results[row.q_id].append(
                NearestConceptMatch(
                    concept_id=int(row.concept_id),
                    concept_name=row.concept_name,
                    similarity=float(row.similarity),
                    is_standard=bool(row.is_standard),
                    is_active=bool(row.is_active),
                )
            )

        return tuple(tuple(matches) for matches in results)

