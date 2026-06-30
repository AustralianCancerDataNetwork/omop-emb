"""Public reader and writer interfaces for omop-emb.

Design
------
* The interface accepts a pre-constructed ``EmbeddingBackend`` (sqlite-vec
  default, or pgvector optional) so it is backend-agnostic.
* Table identity is ``(model_name, provider_type)``: one row per model in the
  registry. ``metric_type`` is supplied by the caller at query time.
* ``omop_cdm_engine`` is **optional** on the reader interface.  When provided,
  KNN results are enriched with concept names and flags from the CDM.
  When absent, ``NearestConceptMatch.concept_name`` etc. are ``None``.
* ``omop_cdm_engine`` is **required** for ingestion methods
  (``embed_and_upsert_concepts``) because concept metadata must be fetched
  from the CDM to populate the embedding table filter columns.
"""

from __future__ import annotations

import logging
from dataclasses import replace as dc_replace
from typing import (
    TYPE_CHECKING,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy import ndarray
from sqlalchemy import Engine, Row

from omop_emb.utils.cdm import (
    count_missing_concepts,
    fetch_cdm_concepts_for_filter,
    iter_cdm_concepts_for_filter,
)
from omop_emb.embeddings import (
    EmbeddingClient,
    EmbeddingRole,
    get_provider_from_provider_type,
)
from omop_emb.backends.base_backend import (
    ConceptEmbeddingRecord,
    EmbeddingBackend,
    EmbeddingModelRecord,
)
from omop_emb.backends.index_config import IndexConfig
from omop_emb.config import BackendType, MetricType, ProviderType
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter, NearestConceptMatch

if TYPE_CHECKING:
    from omop_emb.storage.faiss import FAISSCache

logger = logging.getLogger(__name__)


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
# Reader interface
# ---------------------------------------------------------------------------


class EmbeddingReaderInterface:
    """Backend-neutral read interface for embedding search and retrieval.

    Parameters
    ----------
    backend : EmbeddingBackend
        Pre-constructed backend (SQLiteVecEmbeddingBackend or PGVectorEmbeddingBackend).
    metric_type : MetricType
        Distance metric used for KNN queries and validated against the registry.
    omop_cdm_engine : Engine, optional
        Engine for the user's OMOP CDM.  When provided, KNN results are
        enriched with ``concept_name``, ``is_standard``, and ``is_active``
        from the CDM.  When absent, those fields are ``None``.
    model : str
        Model name in canonical form.
    canonical_model_name : str, optional
    provider_name_or_type : str | ProviderType, optional
        Embedding provider.
    k : int
        Default number of nearest neighbors to return.
    faiss_cache_dir : str, optional
        Optional directory for FAISS index caching.  If provided, the interface
        will attempt to use FAISS for faster KNN search.  Only supported
        if the 'faiss' package is installed.  
    """

    def __init__(
        self,
        model: str,
        backend: EmbeddingBackend,
        metric_type: MetricType,
        *,
        omop_cdm_engine: Optional[Engine] = None,
        provider_name_or_type: Optional[Union[str, ProviderType]] = None,
        k: int = EmbeddingBackend.DEFAULT_K_NEAREST,
        faiss_cache_dir: Optional[str] = None,
    ):
        # Resolve provider type
        if isinstance(provider_name_or_type, str):
            provider_type = ProviderType(provider_name_or_type)
        elif isinstance(provider_name_or_type, ProviderType):
            provider_type = provider_name_or_type
        elif provider_name_or_type is None:
            provider_type = ProviderType.OLLAMA
        else:
            raise ValueError(
                f"Invalid provider_name_or_type: {type(provider_name_or_type).__name__}."
            )

        provider = get_provider_from_provider_type(provider_type)
        canonical_model_name = provider.canonical_model_name(model)

        self._backend = backend

        if not isinstance(metric_type, MetricType):
            raise ValueError(
                f"metric_type must be an instance of MetricType Enum, got {type(metric_type).__name__}"
            )
        self._metric_type = metric_type
        self._provider_type = provider_type
        self._canonical_model_name = canonical_model_name
        self._k = k
        self._cdm_engine = omop_cdm_engine

        self._faiss_cache: Optional["FAISSCache"] = None
        if faiss_cache_dir is not None:
            try:
                from omop_emb.storage.faiss import FAISSCache as _FAISSCache

                self._faiss_cache = _FAISSCache(
                    model_name=canonical_model_name,
                    cache_dir=faiss_cache_dir,
                )
            except ImportError as exc:
                raise ImportError(
                    "faiss_cache_dir was provided but the 'faiss' package is not installed. "
                    "Install it with: pip install omop-emb[faiss-cpu]"
                ) from exc

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
            model_name=self.canonical_model_name
        )
        return record.storage_identifier if record is not None else None

    def is_model_registered(self) -> bool:
        return self._backend.is_model_registered(model_name=self.canonical_model_name)

    def has_any_embeddings(self) -> bool:
        return self._backend.has_any_embeddings(
            model_name=self.canonical_model_name,
            metric_type=self._metric_type,
        )

    def get_embedding_count(self) -> int:
        return self._backend.get_embedding_count(
            model_name=self.canonical_model_name,
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
        faiss_index_config: Optional[IndexConfig] = None,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        """Return nearest stored concepts for each query embedding row.

        Parameters
        ----------
        query_embedding : ndarray
            Shape ``(Q, D)``, Q query vectors of dimension D.  D must match the embedding dimension for the registered model.
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

        if self._faiss_cache is not None:
            if faiss_index_config is None:
                raise ValueError(
                    "faiss_index_config is required when a FAISS cache is configured. "
                    "Pass FlatIndexConfig() for exact search or HNSWIndexConfig(metric_type=...) "
                    "for approximate search."
                )
            record = self._backend.get_registered_model(
                model_name=self.canonical_model_name
            )
            if record is not None and self._faiss_cache.is_fresh(
                record, self._metric_type, faiss_index_config
            ):
                logger.info(
                    "Using FAISS cache for search (model='%s', cache='%s').",
                    self.canonical_model_name,
                    self._faiss_cache.model_dir,
                )
                raw = self._faiss_cache.search(
                    query_embedding,
                    effective_k,
                    self._metric_type,
                    faiss_index_config,
                    concept_filter=concept_filter,
                    backend=self._backend,
                )
                return self._enrich(raw)

        raw = self._backend.get_nearest_concepts(
            model_name=self.canonical_model_name,
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
        faiss_index_config: Optional[IndexConfig] = None,
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
            faiss_index_config=faiss_index_config,
        )

    def get_embeddings_by_concept_ids(
        self,
        concept_ids: Tuple[int, ...],
    ) -> Mapping[int, Sequence[float]]:
        return self._backend.get_embeddings_by_concept_ids(
            model_name=self.canonical_model_name,
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
    ) -> Mapping[int, Row]:
        """Return CDM rows for concepts lacking embeddings, keyed by concept_id.

        Each row contains concept_name, domain_id, vocabulary_id,
        standard_concept, and invalid_reason — all columns needed for both
        text lookup and embedding-record metadata.
        """
        all_concepts = fetch_cdm_concepts_for_filter(
            concept_filter=concept_filter,
            cdm_engine=omop_cdm_engine,
        )
        embedded_ids = self._backend.get_all_stored_concept_ids(
            model_name=self.canonical_model_name,
            metric_type=self._metric_type,
        )
        return {
            cid: row for cid, row in all_concepts.items() if cid not in embedded_ids
        }

    def count_concepts_without_embedding(
        self,
        omop_cdm_engine: Engine,
        *,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> int:
        """Return how many CDM concepts match *concept_filter* but lack an embedding."""
        embedded_ids = self._backend.get_all_stored_concept_ids(
            model_name=self.canonical_model_name,
            metric_type=self._metric_type,
        )
        return count_missing_concepts(concept_filter, omop_cdm_engine, embedded_ids)

    def get_concepts_without_embedding_batched(
        self,
        omop_cdm_engine: Engine,
        *,
        batch_size: int,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: Optional[int] = None,
    ) -> Iterable[Mapping[int, Row]]:
        """Yield ``{concept_id: Row}`` batches for concepts lacking embeddings.

        Streams CDM rows and filters against already-embedded IDs on-the-fly,
        so only one batch of CDM rows is in memory at a time.
        """
        embedded_ids = self._backend.get_all_stored_concept_ids(
            model_name=self.canonical_model_name,
            metric_type=self._metric_type,
        )
        batch: dict[int, Row] = {}
        n_yielded = 0
        for row in iter_cdm_concepts_for_filter(concept_filter, omop_cdm_engine):
            if row.concept_id in embedded_ids:
                continue
            batch[row.concept_id] = row
            if len(batch) >= batch_size:
                yield batch
                n_yielded += len(batch)
                batch = {}
                if limit is not None and n_yielded >= limit:
                    return
        if batch:
            if limit is not None:
                trimmed = dict(list(batch.items())[: limit - n_yielded])
                if trimmed:
                    yield trimmed
            else:
                yield batch

    # ------------------------------------------------------------------
    # CDM enrichment (internal)
    # ------------------------------------------------------------------

    def _enrich(
        self,
        raw: Tuple[Tuple[NearestConceptMatch, ...], ...],
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        """Enrich backend results with concept names from the CDM.

        Only ``concept_name`` is populated here.  ``is_standard`` is already
        set by the backend from the embedding table filter columns.
        """
        if not self._cdm_engine:
            return raw

        unique_ids = {r.concept_id for results in raw for r in results}
        concept_filter = EmbeddingConceptFilter(concept_ids=tuple(unique_ids))
        rows = fetch_cdm_concepts_for_filter(
            concept_filter=concept_filter, cdm_engine=self._cdm_engine
        )

        return tuple(
            tuple(
                dc_replace(
                    r,
                    concept_name=rows[r.concept_id].concept_name
                    if r.concept_id in rows
                    else None,
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
            model=embedding_client.canonical_model_name,
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
        self._backend.delete_model(model_name=self.canonical_model_name)

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
            metric_type=self._metric_type,
            records=records,
            embeddings=embeddings,
        )
        self._backend.refresh_model_updated_at_timestamp(model_name=self.canonical_model_name)

    def bulk_upsert_concept_embeddings(
        self,
        batches: Iterable[Tuple[Sequence[ConceptEmbeddingRecord], ndarray]],
        total_n_batches: Optional[int] = None,
    ) -> None:
        """Upsert from a lazy ``(records, embeddings)`` iterable."""
        self._backend.bulk_upsert_embeddings(
            model_name=self.canonical_model_name,
            metric_type=self._metric_type,
            batches=batches,
            total_n_batches=total_n_batches,
        )
        self._backend.refresh_model_updated_at_timestamp(model_name=self.canonical_model_name)

    def embed_and_upsert_concepts(
        self,
        *,
        concept_ids: Sequence[int],
        concept_texts: Sequence[str],
        concept_meta: Mapping[int, Row],
        batch_size: Optional[int] = None,
    ) -> ndarray:
        """Generate embeddings from CDM concepts and upsert with filter metadata.
        Concept_ids, concept_texts and concept_meta must be aligned (same length, same order).
        `fetch_cdm_concepts_for_filter` can be used to get aligned concept_meta for a set of concept_ids.

        Parameters
        ----------
        concept_ids : Sequence[int]
            OMOP concept IDs to embed.
        concept_texts : Sequence[str]
            Text strings to embed (aligned with *concept_ids*).
        concept_meta : Mapping[int, Row]
            CDM rows keyed by concept_id, as returned by
            ``get_concepts_without_embedding``.  Used to populate
            domain_id, vocabulary_id, is_standard, and is_valid.
        """
        if len(concept_ids) != len(concept_texts):
            raise ValueError(
                f"concept_ids ({len(concept_ids)}) and concept_texts ({len(concept_texts)}) "
                "must have the same length."
            )

        records = [
            ConceptEmbeddingRecord(
                concept_id=cid,
                domain_id=concept_meta[cid].domain_id if cid in concept_meta else "",
                vocabulary_id=concept_meta[cid].vocabulary_id
                if cid in concept_meta
                else "",
                is_standard=concept_meta[cid].standard_concept in ("S", "C")
                if cid in concept_meta
                else False,
                is_valid=concept_meta[cid].invalid_reason not in ("D", "U")
                if cid in concept_meta
                else True,
            )
            for cid in concept_ids
        ]

        # Check registered dimensions
        record = self._backend.get_registered_model(
            model_name=self.canonical_model_name
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
        embedding_client: Optional[EmbeddingClient] = None,
        *,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        batch_size: Optional[int] = None,
        k: Optional[int] = None,
        faiss_index_config: Optional[IndexConfig] = None,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        return super().get_nearest_concepts_from_query_texts(
            query_texts=query_texts,
            embedding_client=embedding_client or self._embedding_client,
            concept_filter=concept_filter,
            batch_size=batch_size,
            k=k,
            faiss_index_config=faiss_index_config,
        )
