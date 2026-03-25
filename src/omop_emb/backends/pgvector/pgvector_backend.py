from __future__ import annotations

from typing import Mapping, Optional, Sequence, Type

from numpy import ndarray
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from .pgvector_sql import (
    q_embedding_cosine_similarity,
    q_embedding_nearest_concepts,
    q_embedding_vectors_by_concept_ids,
    initialise_pg_embedding_tables,
    PGVectorConceptIDEmbeddingTable,
    create_pg_embedding_table,
    add_embeddings_to_registered_table,
)
from ..registry import ModelRegistry
from ..config import BackendType

from ..base import (
    EmbeddingBackend,
    EmbeddingBackendCapabilities,
    EmbeddingConceptFilter,
    EmbeddingIndexConfig,
    EmbeddingModelRecord,
    NearestConceptMatch,
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

    def upsert_embeddings(
        self,
        session: Session,
        model_name: str,
        concept_ids: Sequence[int],
        embeddings: ndarray,
    ) -> None:
        concept_id_tuple = tuple(concept_ids)
        assert embeddings.ndim == 2, f"Expected 2 dimensions of embeddings. Got {embeddings.ndim}"
        assert len(concept_id_tuple) == embeddings.shape[0], (
            f"Mismatch between #concept_ids ({len(concept_id_tuple)}) "
            f"and embedding dimensionality ({embeddings.shape[0]})"
        )

        registered_table = self._get_embedding_table(
            session=session,
            model_name=model_name,
        )

        add_embeddings_to_registered_table(
            session=session,
            concept_ids=concept_id_tuple,
            embeddings=embeddings,
            model=registered_table,
        )

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

    def get_similarities(
        self,
        session: Session,
        model_name: str,
        query_embedding: Sequence[float],
        *,
        concept_ids: Optional[Sequence[int]] = None,
    ) -> Mapping[int, float]:
        embedding_table = self._get_embedding_table(
            session=session,
            model_name=model_name,
        )
        query = q_embedding_cosine_similarity(
            embedding_table=embedding_table,
            text_embedding=list(query_embedding),
            concept_ids=tuple(concept_ids) if concept_ids is not None else None,
        )
        return {
            int(row.concept_id): float(row.similarity)
            for row in session.execute(query).all()
        }

    def get_nearest_concepts(
        self,
        session: Session,
        model_name: str,
        query_embedding: Sequence[float],
        *,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: int = 10,
    ) -> Sequence[NearestConceptMatch]:
        embedding_table = self._get_embedding_table(
            session=session,
            model_name=model_name,
        )
        concept_filter = concept_filter or EmbeddingConceptFilter()

        query = q_embedding_nearest_concepts(
            embedding_table=embedding_table,
            text_embedding=list(query_embedding),
            concept_ids=concept_filter.concept_ids,
            domains=concept_filter.domains,
            vocabularies=concept_filter.vocabularies,
            require_standard=concept_filter.require_standard,
            limit=limit,
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
            backend_name=row.backend_type,
            storage_identifier=row.storage_identifier,
            index_config=EmbeddingIndexConfig(
                index_type=row.index_type,
                distance_metric="cosine",
            ),
            metadata=self._coerce_registry_metadata(row.details),
        )
