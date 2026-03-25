from __future__ import annotations

from typing import Optional, Sequence, Tuple

from numpy import ndarray
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from .backends import EmbeddingBackend, EmbeddingConceptFilter, get_embedding_backend


class EmbeddingInterface:
    """
    Compatibility adapter over the backend selection layer.

    The goal of this class is to preserve the current ``kg.emb`` public surface
    while delegating the actual work to a concrete embedding backend selected by
    configuration.
    """

    def __init__(
        self,
        backend: Optional[EmbeddingBackend] = None,
        *,
        backend_name: Optional[str] = None,
        faiss_base_dir: Optional[str] = None,
    ):
        self._backend = backend or get_embedding_backend(
            backend_name=backend_name,
            faiss_base_dir=faiss_base_dir,
        )

    @property
    def backend(self) -> EmbeddingBackend:
        """Expose the selected backend for advanced callers."""
        return self._backend

    def get_model_table_name(
        self,
        session: Session,
        model_name: str,
    ) -> Optional[str]:
        """
        Legacy helper preserved for compatibility.

        Historically this returned a PostgreSQL table name. Under the backend
        abstraction it returns the backend-specific storage identifier.
        """
        record = self.backend.get_registered_model(session=session, model_name=model_name)
        return record.storage_identifier if record is not None else None

    def is_model_registered(
        self,
        session: Session,
        model_name: str,
    ) -> bool:
        return self.backend.is_model_registered(session=session, model_name=model_name)

    def get_similarities(
        self,
        session: Session,
        embedding_model_name: str,
        text_embedding: Sequence[float],
        concept_ids: Optional[Tuple[int, ...]] = None,
    ):
        return self.backend.get_similarities(
            session=session,
            model_name=embedding_model_name,
            query_embedding=text_embedding,
            concept_ids=concept_ids,
        )

    def has_any_embeddings(
        self,
        session: Session,
        embedding_model_name: str,
    ) -> bool:
        return self.backend.has_any_embeddings(
            session=session,
            model_name=embedding_model_name,
        )

    def get_nearest_concepts(
        self,
        session: Session,
        embedding_model_name: str,
        text_embedding: Sequence[float],
        concept_ids: Optional[Tuple[int, ...]] = None,
        domains: Optional[Tuple[str, ...]] = None,
        vocabularies: Optional[Tuple[str, ...]] = None,
        require_standard: bool = False,
        limit: int = 10,
    ):
        concept_filter = EmbeddingConceptFilter(
            concept_ids=concept_ids,
            domains=domains,
            vocabularies=vocabularies,
            require_standard=require_standard,
        )
        return self.backend.get_nearest_concepts(
            session=session,
            model_name=embedding_model_name,
            query_embedding=text_embedding,
            concept_filter=concept_filter,
            limit=limit,
        )

    def get_embeddings_by_concept_ids(
        self,
        session: Session,
        embedding_model_name: str,
        concept_ids: Tuple[int, ...],
    ) -> dict[int, list[float]]:
        rows = self.backend.get_embeddings_by_concept_ids(
            session=session,
            model_name=embedding_model_name,
            concept_ids=concept_ids,
        )
        return {
            int(concept_id): list(embedding)
            for concept_id, embedding in rows.items()
        }

    def initialise_tables(self, engine: Engine):
        """
        Legacy name preserved for compatibility.
        """

        return self.backend.initialise_store(engine)

    def add_to_db(
        self,
        session: Session,
        concept_ids: Tuple[int, ...],
        embeddings: ndarray,
        model: str,
    ):
        assert embeddings.ndim == 2, f"Expected 2 dimensions of embeddings. Got {embeddings.ndim}"
        assert len(concept_ids) == embeddings.shape[0], (
            f"Mismatch between #concept_ids ({len(concept_ids)}) and embedding "
            f"dimensionality ({embeddings.shape[0]})"
        )

        return self.backend.upsert_embeddings(
            session=session,
            model_name=model,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )