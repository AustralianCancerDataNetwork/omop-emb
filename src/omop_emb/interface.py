"""Public reader and writer interfaces for omop-emb.

Key changes from the previous design
--------------------------------------
* Two mandatory engines instead of one:
  - ``emb_engine``: the dedicated pgvector Postgres instance
  - ``omop_cdm_engine``: the user's OMOP CDM (any SQLAlchemy dialect, read-only)
* No ``backend_name_or_type`` — pgvector is the only production backend.
* No ``storage_base_dir`` or ``registry_db_name`` — the registry lives in
  the pgvector instance, not a local SQLite file.
* The backend is fully initialized inside ``__init__`` (no separate
  ``initialise_store`` call required by the caller).
"""
from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray
from sqlalchemy import Engine

from omop_emb.embeddings import (
    EmbeddingClient,
    EmbeddingRole,
    get_provider_from_provider_type,
)
from omop_emb.embeddings.embedding_providers import get_provider_for_api_base
from omop_emb.storage import EmbeddingBackend, IndexConfig, PGVectorEmbeddingBackend
from omop_emb.storage.postgres.pg_registry import EmbeddingModelRecord, PostgresRegistryManager
from omop_emb.config import BackendType, IndexType, MetricType, ProviderType
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter, NearestConceptMatch


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------

def list_registered_models(
    emb_engine: Engine,
    provider_type: Optional[ProviderType] = None,
    model_name: Optional[str] = None,
    index_type: Optional[IndexType] = None,
) -> tuple[EmbeddingModelRecord, ...]:
    """List registered models from the pgvector registry.

    Parameters
    ----------
    emb_engine : Engine
        Engine for the pgvector instance that holds the registry.
    provider_type : ProviderType, optional
        Filter by provider type.
    model_name : str, optional
        Filter by model name.
    index_type : IndexType, optional
        Filter by index type.

    Returns
    -------
    tuple[EmbeddingModelRecord, ...]
        Matching registered models (empty tuple if none found).
    """
    registry = PostgresRegistryManager(emb_engine=emb_engine)
    return registry.get_registered_models_from_db(
        backend_type=BackendType.PGVECTOR,
        provider_type=provider_type,
        model_name=model_name,
        index_type=index_type,
    )


# ---------------------------------------------------------------------------
# Reader interface
# ---------------------------------------------------------------------------

class EmbeddingReaderInterface:
    """Backend-neutral reader interface for embedding query operations.

    Parameters
    ----------
    emb_engine : Engine
        SQLAlchemy engine for the **dedicated pgvector Postgres instance**.
        The embedding tables and model registry live here.
    omop_cdm_engine : Engine
        SQLAlchemy engine for the **user's OMOP CDM** (read-only).  May be
        any SQLAlchemy-supported dialect.
    model : str, optional
        Raw model name.  Required unless *canonical_model_name* is provided.
    api_base : str, optional
        API base URL for automatic provider detection.
    api_key : str, optional
        API key for automatic provider detection.
    canonical_model_name : str, optional
        Pre-computed canonical model name.  Supply this to bypass the
        provider canonicalization round-trip.
    provider_name_or_type : str | ProviderType, optional
        Embedding provider.  Required if the provider cannot be inferred
        from *api_base*.
    """

    def __init__(
        self,
        emb_engine: Engine,
        omop_cdm_engine: Engine,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        canonical_model_name: Optional[str] = None,
        provider_name_or_type: Optional[Union[str, ProviderType]] = None,
    ):
        # Resolve provider type
        if isinstance(provider_name_or_type, str):
            provider_type = ProviderType(provider_name_or_type)
        elif isinstance(provider_name_or_type, ProviderType):
            provider_type = provider_name_or_type
        elif provider_name_or_type is None:
            if api_base is None or api_key is None:
                raise ValueError(
                    "Either 'provider_name_or_type' or both 'api_base' and 'api_key' "
                    "must be provided."
                )
            provider_type = get_provider_for_api_base(api_base, api_key).provider_type
        else:
            raise ValueError(
                f"Invalid provider_name_or_type: expected str or ProviderType, "
                f"got {type(provider_name_or_type).__name__}."
            )

        provider = get_provider_from_provider_type(provider_type)

        if canonical_model_name is None:
            if model is None:
                raise ValueError("Either 'model' or 'canonical_model_name' must be provided.")
            canonical_model_name = provider.canonical_model_name(model)

        # Idempotency check
        if canonical_model_name != provider.canonical_model_name(canonical_model_name):
            raise ValueError(
                f"Canonical model name validation failed for provider {provider_type}: "
                f"'{canonical_model_name}' is not in canonical form."
            )

        self._backend: EmbeddingBackend = PGVectorEmbeddingBackend(
            emb_engine=emb_engine,
            omop_cdm_engine=omop_cdm_engine,
        )
        self._provider_type = provider_type
        self._canonical_model_name = canonical_model_name

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def backend_type(self) -> BackendType:
        return self._backend.backend_type

    @property
    def canonical_model_name(self) -> str:
        return self._canonical_model_name

    @property
    def provider_type(self) -> ProviderType:
        return self._provider_type

    # ------------------------------------------------------------------
    # Registry queries
    # ------------------------------------------------------------------

    def get_model_table_name(self, index_type: IndexType) -> Optional[str]:
        record = self._backend.get_registered_model(
            model_name=self.canonical_model_name,
            index_type=index_type,
            provider_type=self.provider_type,
        )
        return record.storage_identifier if record is not None else None

    def is_model_registered(self, index_type: IndexType) -> bool:
        return self._backend.is_model_registered(
            model_name=self.canonical_model_name,
            index_type=index_type,
            provider_type=self.provider_type,
        )

    def has_any_embeddings(self, index_type: IndexType) -> bool:
        return self._backend.has_any_embeddings(
            model_name=self.canonical_model_name,
            provider_type=self.provider_type,
            index_type=index_type,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def get_nearest_concepts(
        self,
        index_type: IndexType,
        query_embedding: np.ndarray,
        *,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        """Return nearest stored concepts for the query embedding.

        Parameters
        ----------
        query_embedding : ndarray
            Shape ``(Q, D)`` — one row per query vector.
        metric_type : MetricType
            Distance / similarity metric.
        concept_filter : EmbeddingConceptFilter, optional
            Optional constraints (domain, vocabulary, standard flag, limit).

        Returns
        -------
        Tuple[Tuple[NearestConceptMatch, ...], ...]
            Shape ``(Q, K)``.
        """
        if not isinstance(metric_type, MetricType):
            raise TypeError(
                f"metric_type must be MetricType, got {type(metric_type).__name__}."
            )
        return self._backend.get_nearest_concepts(
            model_name=self.canonical_model_name,
            provider_type=self.provider_type,
            index_type=index_type,
            query_embeddings=query_embedding,
            concept_filter=concept_filter,
            metric_type=metric_type,
        )

    def get_nearest_concepts_from_query_texts(
        self,
        index_type: IndexType,
        query_texts: str | Tuple[str, ...] | List[str],
        embedding_client: EmbeddingClient,
        *,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        """Embed *query_texts* then return nearest concepts."""
        if isinstance(query_texts, str):
            query_texts = (query_texts,)
        else:
            query_texts = tuple(query_texts)
        query_embeddings = embedding_client.embeddings(
            query_texts,
            batch_size=batch_size,
            embedding_role=EmbeddingRole.QUERY,
        )
        return self.get_nearest_concepts(
            index_type=index_type,
            query_embedding=query_embeddings,
            metric_type=metric_type,
            concept_filter=concept_filter,
        )

    def get_embeddings_by_concept_ids(
        self,
        index_type: IndexType,
        concept_ids: Tuple[int, ...],
    ) -> Mapping[int, Sequence[float]]:
        return self._backend.get_embeddings_by_concept_ids(
            model_name=self.canonical_model_name,
            index_type=index_type,
            concept_ids=concept_ids,
            provider_type=self.provider_type,
        )

    def get_concepts_without_embedding(
        self,
        *,
        index_type: IndexType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> Mapping[int, str]:
        return self._backend.get_concepts_without_embedding(
            model_name=self.canonical_model_name,
            provider_type=self.provider_type,
            index_type=index_type,
            concept_filter=concept_filter,
        )

    def get_concepts_without_embedding_batched(
        self,
        *,
        index_type: IndexType,
        batch_size: int,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> Iterable[Mapping[int, str]]:
        return self._backend.get_concepts_without_embedding_batched(
            model_name=self.canonical_model_name,
            provider_type=self.provider_type,
            index_type=index_type,
            concept_filter=concept_filter,
            batch_size=batch_size,
        )

    def get_concepts_without_embedding_count(
        self,
        *,
        index_type: IndexType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> int:
        return self._backend.get_concepts_without_embedding_count(
            model_name=self.canonical_model_name,
            provider_type=self.provider_type,
            index_type=index_type,
            concept_filter=concept_filter,
        )


# ---------------------------------------------------------------------------
# Writer interface
# ---------------------------------------------------------------------------

class EmbeddingWriterInterface(EmbeddingReaderInterface):
    """Reader interface extended with embedding generation and write operations.

    Parameters
    ----------
    emb_engine : Engine
        Engine for the dedicated pgvector Postgres instance.
    omop_cdm_engine : Engine
        Engine for the user's OMOP CDM (read-only).
    embedding_client : EmbeddingClient
        Required.  Provides the provider, model name, and dimension, and
        is used to generate embeddings for upsert and query operations.
    """

    def __init__(
        self,
        emb_engine: Engine,
        omop_cdm_engine: Engine,
        embedding_client: EmbeddingClient,
    ):
        self._embedding_client = embedding_client
        super().__init__(
            emb_engine=emb_engine,
            omop_cdm_engine=omop_cdm_engine,
            canonical_model_name=embedding_client.canonical_model_name,
            provider_name_or_type=embedding_client.provider.provider_type,
        )

    @property
    def embedding_dim(self) -> int:
        return self._embedding_client.embedding_dim

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def register_model(
        self,
        index_config: IndexConfig,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> EmbeddingModelRecord:
        """Register the embedding model in the pgvector registry.

        Idempotent: if the model is already registered with the same
        dimensions and metadata, the existing record is returned.
        """
        return self._backend.register_model(
            model_name=self.canonical_model_name,
            provider_type=self._embedding_client.provider.provider_type,
            dimensions=self.embedding_dim,
            index_config=index_config,
            metadata=metadata,
        )

    def delete_model(self, index_type: IndexType) -> None:
        """Irreversibly delete the model and all associated embeddings."""
        self._backend.delete_model(
            model_name=self.canonical_model_name,
            provider_type=self._provider_type,
            index_type=index_type,
        )

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    def embed_texts(
        self,
        texts: str | Tuple[str, ...] | List[str],
        *,
        embedding_role: EmbeddingRole,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        return self._embedding_client.embeddings(
            texts, batch_size=batch_size, embedding_role=embedding_role
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert_concept_embeddings(
        self,
        *,
        index_type: IndexType,
        concept_ids: Sequence[int],
        embeddings: ndarray,
        metric_type: Optional[MetricType] = None,
    ) -> None:
        self._backend.upsert_embeddings(
            model_name=self.canonical_model_name,
            provider_type=self._embedding_client.provider.provider_type,
            index_type=index_type,
            concept_ids=concept_ids,
            embeddings=embeddings,
            metric_type=metric_type,
        )

    def bulk_upsert_concept_embeddings(
        self,
        *,
        index_type: IndexType,
        batches: Iterable[Tuple[Sequence[int], ndarray]],
        metric_type: Optional[MetricType] = None,
    ) -> None:
        """Upsert embeddings from a lazy ``(concept_ids, embeddings)`` iterable."""
        self._backend.bulk_upsert_embeddings(
            model_name=self.canonical_model_name,
            provider_type=self._embedding_client.provider.provider_type,
            index_type=index_type,
            batches=batches,
            metric_type=metric_type,
        )

    def embed_and_upsert_concepts(
        self,
        *,
        index_type: IndexType,
        concept_ids: Sequence[int],
        concept_texts: Sequence[str],
        batch_size: Optional[int] = None,
    ) -> ndarray:
        """Generate embeddings for concepts and upsert them.

        Validates dimension against the registered model *before* the API
        call so a misconfigured client fails fast rather than wasting credits.
        """
        if len(concept_ids) != len(concept_texts):
            raise ValueError(
                f"Mismatch between #concept_ids ({len(concept_ids)}) and "
                f"#concept_texts ({len(concept_texts)})."
            )
        record = self._backend.get_registered_model(
            model_name=self.canonical_model_name,
            index_type=index_type,
            provider_type=self.provider_type,
        )
        if record is not None and record.dimensions != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch for model '{self.canonical_model_name}': "
                f"client produces {self.embedding_dim}-dimensional vectors but the "
                f"registered model declares {record.dimensions} dimensions."
            )
        embeddings = self.embed_texts(
            list(concept_texts),
            batch_size=batch_size,
            embedding_role=EmbeddingRole.DOCUMENT,
        )
        self.upsert_concept_embeddings(
            index_type=index_type,
            concept_ids=concept_ids,
            embeddings=embeddings,
        )
        return embeddings

    def get_nearest_concepts_from_query_texts(
        self,
        index_type: IndexType,
        query_texts: str | Tuple[str, ...] | List[str],
        *,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        return super().get_nearest_concepts_from_query_texts(
            index_type=index_type,
            query_texts=query_texts,
            embedding_client=self._embedding_client,
            metric_type=metric_type,
            concept_filter=concept_filter,
            batch_size=batch_size,
        )
