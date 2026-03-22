from __future__ import annotations

from typing import Mapping, Optional, Sequence

from numpy import ndarray
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from omop_emb.cdm.embeddings import (
    _MODEL_CACHE,
    _resolve_index_method,
    add_embeddings_to_registered_table,
    initialize_embedding_tables,
    register_new_model,
)
from omop_emb.queries import (
    q_embedding_cosine_similarity,
    q_embedding_nearest_concepts,
    q_embedding_vectors_by_concept_ids,
)
from omop_emb.registry import ModelRegistry

from .errors import EmbeddingBackendConfigurationError
from .base import (
    EmbeddingBackend,
    EmbeddingBackendCapabilities,
    EmbeddingConceptFilter,
    EmbeddingIndexConfig,
    EmbeddingModelRecord,
    NearestConceptMatch,
)


class PostgresEmbeddingBackend(EmbeddingBackend):
    """
    PostgreSQL/pgvector-backed embedding backend.

    This class is intentionally a thin structural layer over the existing
    ``omop_emb`` implementation. It is not wired into the current accessor or
    CLI yet, but it demonstrates how the present behavior maps onto the new
    backend abstraction.
    """

    @property
    def backend_name(self) -> str:
        return "postgres"

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
        initialize_embedding_tables(engine)

    def list_registered_models(self, session: Session) -> Sequence[EmbeddingModelRecord]:
        return tuple(
            self._record_from_registry_row(row)
            for row in session.scalars(
                select(ModelRegistry).where(ModelRegistry.backend_name == self.backend_name)
            ).all()
        )

    def get_registered_model(
        self,
        session: Session,
        model_name: str,
    ) -> Optional[EmbeddingModelRecord]:
        row = session.scalar(
            select(ModelRegistry).where(
                ModelRegistry.name == model_name,
                ModelRegistry.backend_name == self.backend_name,
            )
        )
        if row is None:
            return None
        return self._record_from_registry_row(row)

    def register_model(
        self,
        engine: Engine,
        model_name: str,
        dimensions: int,
        *,
        index_config: Optional[EmbeddingIndexConfig] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> EmbeddingModelRecord:
        with Session(engine, expire_on_commit=False) as session:
            existing_row = session.scalar(
                select(ModelRegistry).where(ModelRegistry.name == model_name)
            )
            if existing_row is not None:
                if existing_row.backend_name != self.backend_name:
                    raise EmbeddingBackendConfigurationError(
                        f"Model '{model_name}' is already registered for backend "
                        f"'{existing_row.backend_name}', not '{self.backend_name}'."
                    )
                if existing_row.dimensions != dimensions:
                    raise EmbeddingBackendConfigurationError(
                        f"Model '{model_name}' is already registered with dimensions "
                        f"{existing_row.dimensions}, not {dimensions}."
                    )
                if index_config is not None:
                    requested_index_method = _resolve_index_method(
                        session,
                        dimensions=dimensions,
                        index_method=index_config.index_type,
                    )
                    session.commit()
                    if existing_row.index_method != requested_index_method:
                        raise EmbeddingBackendConfigurationError(
                            f"Model '{model_name}' is already registered with "
                            f"index_method='{existing_row.index_method}', not "
                            f"'{requested_index_method}'. Reuse the existing model "
                            "configuration or register a new model name."
                        )
                return self._record_from_registry_row(existing_row)

        dynamic_model = register_new_model(
            engine=engine,
            model_name=model_name,
            dimensions=dimensions,
            index_method=index_config.index_type if index_config is not None else None,
        )
        storage_identifier = dynamic_model.__tablename__
        return EmbeddingModelRecord(
            model_name=model_name,
            dimensions=dimensions,
            backend_name=self.backend_name,
            storage_identifier=storage_identifier,
            index_config=index_config,
            metadata=dict(metadata or {}),
        )

    def has_any_embeddings(self, session: Session, model_name: str) -> bool:
        embedding_table = self._get_embedding_table(
            session=session,
            model_name=model_name,
        )
        return session.query(embedding_table.concept_id).limit(1).first() is not None

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

        add_embeddings_to_registered_table(
            session=session,
            concept_ids=concept_id_tuple,
            embeddings=embeddings,
            model=model_name,
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

    def _get_embedding_table(
        self,
        session: Session,
        model_name: str,
    ):
        embedding_table = _MODEL_CACHE.get(model_name)
        if embedding_table is not None:
            return embedding_table

        bind = session.get_bind()
        if bind is None:
            raise RuntimeError("Session is not bound to an engine.")

        initialize_embedding_tables(bind)
        embedding_table = _MODEL_CACHE.get(model_name)
        if embedding_table is None:
            raise ValueError(f"Embedding model '{model_name}' not found in cache.")
        return embedding_table

    def _record_from_registry_row(self, row: ModelRegistry) -> EmbeddingModelRecord:
        return EmbeddingModelRecord(
            model_name=row.name,
            dimensions=row.dimensions,
            backend_name=row.backend_name,
            storage_identifier=row.storage_identifier,
            index_config=EmbeddingIndexConfig(
                index_type=row.index_method,
                distance_metric="cosine",
            ),
            metadata={},
        )
