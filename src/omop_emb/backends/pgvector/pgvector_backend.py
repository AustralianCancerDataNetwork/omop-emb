from __future__ import annotations

from typing import Mapping, Optional, Sequence, Type

from numpy import ndarray
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from .pgvector_sql import (
    q_embedding_nearest_concepts,
    q_embedding_vectors_by_concept_ids,
    initialise_pg_embedding_tables,
    PGVectorConceptIDEmbeddingTable,
    create_pg_embedding_table,
    add_embeddings_to_registered_table,
)
from ..registry import ModelRegistry
from ..config import BackendType, MetricType

from ..base import (
    EmbeddingBackend,
    EmbeddingBackendCapabilities,
    EmbeddingConceptFilter,
    EmbeddingModelRecord,
    NearestConceptMatch,
    require_registered_model
)

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

    @property
    def capabilities(self) -> EmbeddingBackendCapabilities:
        return EmbeddingBackendCapabilities(
            stores_embeddings=True,
            supports_incremental_upsert=True,
            supports_nearest_neighbor_search=True,
            supports_server_side_similarity=True,
            supports_filter_by_concept_ids=True,
            supports_filter_by_domain=True,
            supports_filter_by_vocabulary=True,
            supports_filter_by_standard_flag=True,
            requires_explicit_index_refresh=False,
        )

    def initialise_store(self, engine: Engine) -> None:
        return initialise_pg_embedding_tables(engine, model_cache=self.model_cache)

    def _create_storage_table(self, engine: Engine, entry: ModelRegistry) -> Type[PGVectorConceptIDEmbeddingTable]:
        return create_pg_embedding_table(engine=engine, model_registry_entry=entry)

    @require_registered_model
    def upsert_embeddings(
        self,
        session: Session,
        model_name: str,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
        embeddings: ndarray,
    ) -> None:
        concept_id_tuple = tuple(concept_ids)

        self.validate_embeddings_and_concept_ids(
            concept_ids=concept_id_tuple,
            embeddings=embeddings,
            dimensions=model_record.dimensions,
        )

        table = self._get_embedding_table(
            session=session,
            model_name=model_name,
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

    def get_embeddings_by_concept_ids(
        self,
        session: Session,
        model_name: str,
        concept_ids: Sequence[int],
    ) -> Mapping[int, Sequence[float]]:
        concept_id_tuple = tuple(concept_ids)
        if not concept_id_tuple:
            return {}

        embedding_table = self._get_embedding_table(
            session=session,
            model_name=model_name,
        )
        query = q_embedding_vectors_by_concept_ids(
            embedding_table=embedding_table,
            concept_ids=concept_id_tuple,
        )
        return {
            int(row.concept_id): list(row.embedding)
            for row in session.execute(query)
        }

    @require_registered_model
    def get_nearest_concepts(
        self,
        session: Session,
        model_name: str,
        query_embedding: ndarray,
        *,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: int = 10,
    ) -> Sequence[NearestConceptMatch]:
        embedding_table = self._get_embedding_table(
            session=session,
            model_name=model_name,
        )
        concept_filter = concept_filter

        assert query_embedding.shape[0] == 1, f"Expected query_embedding to have shape (1, dimension), got {query_embedding.shape}"

        query = q_embedding_nearest_concepts(
            embedding_table=embedding_table,
            text_embedding=query_embedding.tolist()[0],
            concept_filter=concept_filter,
            limit=limit,
            metric_type=metric_type
        )
        return tuple(
            NearestConceptMatch(
                concept_id=int(row.concept_id),
                concept_name=row.concept_name,
                similarity=float(row.similarity),
                is_standard=bool(row.is_standard),
                is_active=bool(row.is_active),
            )
            for row in session.execute(query).all()
        )

    def refresh_model_index(self, session: Session, model_name: str) -> None:
        # pgvector indexes update as rows are written, so no explicit refresh is
        # required for the current PostgreSQL backend.
        return None

    def _record_from_registry_row(self, row: ModelRegistry) -> EmbeddingModelRecord:
        return EmbeddingModelRecord(
            model_name=row.model_name,
            dimensions=row.dimensions,
            backend_type=row.backend_type,
            storage_identifier=row.storage_identifier,
            index_type=row.index_type,
            metadata=self._coerce_registry_metadata(row.metadata),
        )
