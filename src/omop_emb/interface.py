"""Public reader and writer interfaces for omop-emb.

Design
------
* The interface accepts a pre-constructed ``EmbeddingBackend`` (sqlite-vec
  default, or pgvector optional) so it is backend-agnostic.
* Table identity is ``(model_name, provider_type)`` — one row per model in the
  registry. ``metric_type`` is supplied by the caller at query time.
* ``omop_cdm_engine`` is **optional** on the reader interface.  When provided,
  KNN results are enriched with concept names and flags from the CDM.
  When absent, ``NearestConceptMatch.concept_name`` etc. are ``None``.
* ``omop_cdm_engine`` is **required** for ingestion methods
  (``embed_and_upsert_concepts``) because concept metadata must be fetched
  from the CDM to populate the embedding table filter columns.
"""
from __future__ import annotations

from dataclasses import replace as dc_replace
from itertools import batched
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray
from sqlalchemy import Engine, select
from sqlalchemy.orm import sessionmaker

from omop_alchemy.cdm.model.vocabulary import Concept

from omop_emb.embeddings import (
    EmbeddingClient,
    EmbeddingRole,
    get_provider_from_provider_type,
)
from omop_emb.embeddings.embedding_providers import get_provider_for_api_base
from omop_emb.backends.base_backend import (
    ConceptEmbeddingRecord,
    EmbeddingBackend,
    EmbeddingModelRecord,
)
from omop_emb.backends.index_config import IndexConfig
from omop_emb.config import BackendType, MetricType, ProviderType
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter, NearestConceptMatch


# ---------------------------------------------------------------------------
# Standalone utility
# ---------------------------------------------------------------------------

def list_registered_models(
    backend: EmbeddingBackend,
    provider_type: Optional[ProviderType] = None,
    model_name: Optional[str] = None,
) -> tuple[EmbeddingModelRecord, ...]:
    """List models registered by the backend, optionally filtered by provider and/or model name.

    Parameters
    ----------
    backend : EmbeddingBackend
        Backend to query.
    provider_type : ProviderType, optional
        Provider that serves the model.
    model_name : str, optional
        Filter by canonical model name.

    Returns
    -------
    tuple[EmbeddingModelRecord, ...]
    """
    return backend.get_registered_models(
        model_name=model_name,
        provider_type=provider_type,
    )


# ---------------------------------------------------------------------------
# CDM enrichment helper (private)
# ---------------------------------------------------------------------------

def _fetch_cdm_concept_metadata(
    concept_ids: set[int],
    cdm_session_factory: sessionmaker,
) -> dict[int, object]:
    if not concept_ids:
        return {}
    query = select(
        Concept.concept_id,
        Concept.concept_name,
        Concept.standard_concept,
        Concept.invalid_reason,
    ).where(Concept.concept_id.in_(concept_ids))
    with cdm_session_factory() as session:
        return {row.concept_id: row for row in session.execute(query)}


def _fetch_cdm_concepts_for_filter(
    concept_filter: Optional[EmbeddingConceptFilter],
    cdm_session_factory: sessionmaker,
) -> dict[int, str]:
    """Return {concept_id: concept_name} from CDM matching the filter."""
    query = select(Concept.concept_id, Concept.concept_name)
    if concept_filter is not None:
        query = concept_filter.apply(query)
    with cdm_session_factory() as session:
        return {row.concept_id: row.concept_name for row in session.execute(query)}


# ---------------------------------------------------------------------------
# Reader interface
# ---------------------------------------------------------------------------

class EmbeddingReaderInterface:
    """Backend-neutral read interface for embedding search and retrieval.

    Parameters
    ----------
    backend : EmbeddingBackend
        Pre-constructed backend (SQLiteVecBackend or PGVectorEmbeddingBackend).
    metric_type : MetricType
        Distance metric used for KNN queries and validated against the registry.
    omop_cdm_engine : Engine, optional
        Engine for the user's OMOP CDM.  When provided, KNN results are
        enriched with ``concept_name``, ``is_standard``, and ``is_active``
        from the CDM.  When absent, those fields are ``None``.
    model : str, optional
        Raw model name (canonicalized via the provider).
    canonical_model_name : str, optional
        Pre-computed canonical model name.
    provider_name_or_type : str | ProviderType, optional
        Embedding provider.
    api_base / api_key : str, optional
        Used for automatic provider detection when *provider_name_or_type*
        is not given.
    k : int
        Default number of nearest neighbors to return.
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        metric_type: MetricType,
        *,
        omop_cdm_engine: Optional[Engine] = None,
        model: Optional[str] = None,
        canonical_model_name: Optional[str] = None,
        provider_name_or_type: Optional[Union[str, ProviderType]] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        k: int = EmbeddingBackend.DEFAULT_K_NEAREST,
    ):
        # Resolve provider type
        if isinstance(provider_name_or_type, str):
            provider_type = ProviderType(provider_name_or_type)
        elif isinstance(provider_name_or_type, ProviderType):
            provider_type = provider_name_or_type
        elif provider_name_or_type is None:
            if api_base is None or api_key is None:
                raise ValueError(
                    "Either 'provider_name_or_type' or both 'api_base' and 'api_key' must be provided."
                )
            provider_type = get_provider_for_api_base(api_base, api_key).provider_type
        else:
            raise ValueError(f"Invalid provider_name_or_type: {type(provider_name_or_type).__name__}.")

        provider = get_provider_from_provider_type(provider_type)

        if canonical_model_name is None:
            if model is None:
                raise ValueError("Either 'model' or 'canonical_model_name' must be provided.")
            canonical_model_name = provider.canonical_model_name(model)

        self._backend = backend
        self._metric_type = metric_type
        self._provider_type = provider_type
        self._canonical_model_name = canonical_model_name
        self._k = k
        self._cdm_engine = omop_cdm_engine
        self._cdm_session_factory = sessionmaker(omop_cdm_engine) if omop_cdm_engine else None

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_sqlitevec(
        cls,
        db_path: str,
        metric_type: MetricType,
        **kwargs,
    ) -> "EmbeddingReaderInterface":
        from omop_emb.backends.sqlitevec import SQLiteVecBackend
        return cls(backend=SQLiteVecBackend.from_path(db_path), metric_type=metric_type, **kwargs)

    @classmethod
    def from_pgvector(
        cls,
        emb_engine: Engine,
        metric_type: MetricType,
        **kwargs,
    ) -> "EmbeddingReaderInterface":
        from omop_emb.backends.pgvector import PGVectorEmbeddingBackend
        return cls(backend=PGVectorEmbeddingBackend(emb_engine=emb_engine), metric_type=metric_type, **kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def backend(self) -> EmbeddingBackend:
        return self._backend

    @property
    def backend_type(self) -> BackendType:
        return self._backend.backend_type

    @property
    def metric_type(self) -> MetricType:
        return self._metric_type

    @property
    def canonical_model_name(self) -> str:
        return self._canonical_model_name

    @property
    def provider_type(self) -> ProviderType:
        return self._provider_type

    # ------------------------------------------------------------------
    # Registry queries
    # ------------------------------------------------------------------

    def get_model_table_name(self) -> Optional[str]:
        record = self._backend.get_registered_model(
            model_name=self.canonical_model_name,
            provider_type=self.provider_type,
        )
        return record.storage_identifier if record is not None else None

    def is_model_registered(self) -> bool:
        return self._backend.is_model_registered(
            model_name=self.canonical_model_name,
            provider_type=self.provider_type,
        )

    def has_any_embeddings(self) -> bool:
        return self._backend.has_any_embeddings(
            model_name=self.canonical_model_name,
            provider_type=self.provider_type,
            metric_type=self._metric_type,
        )

    def get_embedding_count(self) -> int:
        return self._backend.get_embedding_count(
            model_name=self.canonical_model_name,
            provider_type=self.provider_type,
            metric_type=self._metric_type,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def get_nearest_concepts(
        self,
        query_embedding: np.ndarray,
        *,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        k: Optional[int] = None,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        """Return nearest stored concepts for each query embedding row.

        Parameters
        ----------
        query_embedding : ndarray
            Shape ``(Q, D)`` — one row per query vector.
        concept_filter : EmbeddingConceptFilter, optional
            In-DB pre-filter applied during KNN (domain, vocabulary, standard).
        k : int, optional
            Number of nearest neighbors (defaults to interface-level *k*).

        Returns
        -------
        Tuple[Tuple[NearestConceptMatch, ...], ...]
            Shape ``(Q, ≤k)``.  Enrichment fields are ``None`` if no CDM engine.
        """
        effective_k = k or (concept_filter.limit if concept_filter else None) or self._k
        raw = self._backend.get_nearest_concepts(
            model_name=self.canonical_model_name,
            provider_type=self.provider_type,
            metric_type=self._metric_type,
            query_embeddings=query_embedding,
            concept_filter=concept_filter,
            k=effective_k,
        )
        return self._enrich(raw)

    def get_nearest_concepts_from_query_texts(
        self,
        query_texts: Union[str, Tuple[str, ...], List[str]],
        embedding_client: EmbeddingClient,
        *,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        batch_size: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        """Embed *query_texts* then search for nearest concepts."""
        if isinstance(query_texts, str):
            query_texts = (query_texts,)
        query_embeddings = embedding_client.embeddings(
            tuple(query_texts),
            batch_size=batch_size,
            embedding_role=EmbeddingRole.QUERY,
        )
        return self.get_nearest_concepts(
            query_embedding=query_embeddings,
            concept_filter=concept_filter,
            k=k,
        )

    def get_embeddings_by_concept_ids(
        self,
        concept_ids: Tuple[int, ...],
    ) -> Mapping[int, Sequence[float]]:
        return self._backend.get_embeddings_by_concept_ids(
            model_name=self.canonical_model_name,
            provider_type=self.provider_type,
            metric_type=self._metric_type,
            concept_ids=concept_ids,
        )

    # ------------------------------------------------------------------
    # Concepts without embedding (requires CDM)
    # ------------------------------------------------------------------

    def get_concepts_without_embedding(
        self,
        omop_cdm_engine: Engine,
        *,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> Mapping[int, str]:
        """Return ``{concept_id: concept_name}`` for concepts lacking embeddings.

        Requires *omop_cdm_engine* to query the CDM for candidate concepts.
        """
        cdm_factory = sessionmaker(omop_cdm_engine)
        all_concepts = _fetch_cdm_concepts_for_filter(concept_filter, cdm_factory)
        embedded_ids = self._backend.get_all_stored_concept_ids(
            model_name=self.canonical_model_name,
            provider_type=self.provider_type,
            metric_type=self._metric_type,
        )
        return {cid: name for cid, name in all_concepts.items() if cid not in embedded_ids}

    def get_concepts_without_embedding_batched(
        self,
        omop_cdm_engine: Engine,
        *,
        batch_size: int,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> Iterable[Mapping[int, str]]:
        missing = self.get_concepts_without_embedding(
            omop_cdm_engine=omop_cdm_engine,
            concept_filter=concept_filter,
        )
        items = list(missing.items())
        for batch in batched(items, batch_size):
            yield dict(batch)

    # ------------------------------------------------------------------
    # CDM enrichment (internal)
    # ------------------------------------------------------------------

    def _enrich(
        self,
        raw: Tuple[Tuple[NearestConceptMatch, ...], ...],
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        """Enrich backend results with CDM concept metadata if available."""
        if not self._cdm_session_factory:
            return raw

        unique_ids = {r.concept_id for results in raw for r in results}
        meta = _fetch_cdm_concept_metadata(unique_ids, self._cdm_session_factory)

        return tuple(
            tuple(
                dc_replace(
                    r,
                    concept_name=meta[r.concept_id].concept_name if r.concept_id in meta else None,
                    is_standard=meta[r.concept_id].standard_concept in ("S", "C") if r.concept_id in meta else None,
                    is_active=meta[r.concept_id].invalid_reason not in ("D", "U") if r.concept_id in meta else None,
                )
                for r in query_results
            )
            for query_results in raw
        )


# ---------------------------------------------------------------------------
# Writer interface
# ---------------------------------------------------------------------------

class EmbeddingWriterInterface(EmbeddingReaderInterface):
    """Reader interface extended with embedding generation and write operations.

    Parameters
    ----------
    backend : EmbeddingBackend
        Pre-constructed backend.
    metric_type : MetricType
        Distance metric for the table.
    embedding_client : EmbeddingClient
        Required.  Supplies model name, provider type, and dimensionality, and
        is used for generating embeddings.
    omop_cdm_engine : Engine, optional
        CDM engine used for result enrichment.  Pass to write methods directly
        when needed for ingestion.
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        metric_type: MetricType,
        embedding_client: EmbeddingClient,
        *,
        omop_cdm_engine: Optional[Engine] = None,
    ):
        self._embedding_client = embedding_client
        super().__init__(
            backend=backend,
            metric_type=metric_type,
            omop_cdm_engine=omop_cdm_engine,
            canonical_model_name=embedding_client.canonical_model_name,
            provider_name_or_type=embedding_client.provider.provider_type,
        )

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_sqlitevec(
        cls,
        db_path: str,
        metric_type: MetricType,
        embedding_client: EmbeddingClient,
        **kwargs,
    ) -> "EmbeddingWriterInterface":
        from omop_emb.backends.sqlitevec import SQLiteVecBackend
        return cls(
            backend=SQLiteVecBackend.from_path(db_path),
            metric_type=metric_type,
            embedding_client=embedding_client,
            **kwargs,
        )

    @classmethod
    def from_pgvector(
        cls,
        emb_engine: Engine,
        metric_type: MetricType,
        embedding_client: EmbeddingClient,
        **kwargs,
    ) -> "EmbeddingWriterInterface":
        from omop_emb.backends.pgvector import PGVectorEmbeddingBackend
        return cls(
            backend=PGVectorEmbeddingBackend(emb_engine=emb_engine),
            metric_type=metric_type,
            embedding_client=embedding_client,
            **kwargs,
        )

    @property
    def embedding_dim(self) -> int:
        return self._embedding_client.embedding_dim

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def register_model(
        self,
        *,
        index_config: Optional[IndexConfig] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> EmbeddingModelRecord:
        """Register the embedding model in the backend registry.

        Parameters
        ----------
        index_config : IndexConfig, optional
            Index configuration. Defaults to ``FlatIndexConfig()`` when not
            provided.
        metadata : Mapping[str, object], optional
            Free-form operational metadata. Must not contain reserved keys.

        Returns
        -------
        EmbeddingModelRecord
        """
        return self._backend.register_model(
            model_name=self.canonical_model_name,
            provider_type=self._embedding_client.provider.provider_type,
            dimensions=self.embedding_dim,
            index_config=index_config,
            metadata=metadata,
        )

    def delete_model(self) -> None:
        """Irreversibly delete the model and all associated embeddings."""
        self._backend.delete_model(
            model_name=self.canonical_model_name,
            provider_type=self._provider_type,
        )

    def rebuild_index(self, index_config: IndexConfig) -> EmbeddingModelRecord:
        """Build or rebuild the index on the embedding table.

        Parameters
        ----------
        index_config : IndexConfig
            New index configuration.

        Returns
        -------
        EmbeddingModelRecord
            Updated registry record.
        """
        return self._backend.rebuild_index(
            model_name=self.canonical_model_name,
            provider_type=self._provider_type,
            index_config=index_config,
        )

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    def embed_texts(
        self,
        texts: Union[str, Tuple[str, ...], List[str]],
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
        records: Sequence[ConceptEmbeddingRecord],
        embeddings: ndarray,
    ) -> None:
        """Upsert pre-built ConceptEmbeddingRecords with their embeddings."""
        self._backend.upsert_embeddings(
            model_name=self.canonical_model_name,
            provider_type=self._embedding_client.provider.provider_type,
            metric_type=self._metric_type,
            records=records,
            embeddings=embeddings,
        )

    def bulk_upsert_concept_embeddings(
        self,
        batches: Iterable[Tuple[Sequence[ConceptEmbeddingRecord], ndarray]],
    ) -> None:
        """Upsert from a lazy ``(records, embeddings)`` iterable."""
        self._backend.bulk_upsert_embeddings(
            model_name=self.canonical_model_name,
            provider_type=self._embedding_client.provider.provider_type,
            metric_type=self._metric_type,
            batches=batches,
        )

    def embed_and_upsert_concepts(
        self,
        *,
        omop_cdm_engine: Engine,
        concept_ids: Sequence[int],
        concept_texts: Sequence[str],
        batch_size: Optional[int] = None,
    ) -> ndarray:
        """Generate embeddings from CDM concepts and upsert with filter metadata.

        Parameters
        ----------
        omop_cdm_engine : Engine
            CDM engine used to fetch concept filter attributes
            (domain_id, vocabulary_id, standard_concept) for each concept_id.
        concept_ids : Sequence[int]
            OMOP concept IDs to embed.
        concept_texts : Sequence[str]
            Text strings to embed (aligned with *concept_ids*).
        """
        if len(concept_ids) != len(concept_texts):
            raise ValueError(
                f"concept_ids ({len(concept_ids)}) and concept_texts ({len(concept_texts)}) "
                "must have the same length."
            )

        # Fetch concept metadata from CDM for filter columns
        cdm_factory = sessionmaker(omop_cdm_engine)
        meta = _fetch_cdm_concept_metadata(set(concept_ids), cdm_factory)

        records = [
            ConceptEmbeddingRecord(
                concept_id=cid,
                domain_id=meta[cid].domain_id if cid in meta else "",
                vocabulary_id=meta[cid].vocabulary_id if cid in meta else "",
                is_standard=meta[cid].standard_concept in ("S", "C") if cid in meta else False,
            )
            for cid in concept_ids
        ]

        # Check registered dimensions
        record = self._backend.get_registered_model(
            model_name=self.canonical_model_name,
            provider_type=self.provider_type,
        )
        if record is not None and record.dimensions != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: client produces {self.embedding_dim}d "
                f"but registered model declares {record.dimensions}d."
            )

        embeddings = self.embed_texts(
            list(concept_texts),
            batch_size=batch_size,
            embedding_role=EmbeddingRole.DOCUMENT,
        )
        self.upsert_concept_embeddings(records=records, embeddings=embeddings)
        return embeddings

    def get_nearest_concepts_from_query_texts(
        self,
        query_texts: Union[str, Tuple[str, ...], List[str]],
        *,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        batch_size: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        return super().get_nearest_concepts_from_query_texts(
            query_texts=query_texts,
            embedding_client=self._embedding_client,
            concept_filter=concept_filter,
            batch_size=batch_size,
            k=k,
        )
