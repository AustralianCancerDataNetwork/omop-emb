"""Public reader and writer interfaces for omop-emb.

Design
------
* The interface accepts a pre-constructed ``EmbeddingBackend`` (sqlite-vec
  default, or pgvector optional) so it is backend-agnostic.
* Table identity is ``(model_name, provider_type)``: one row per model in the
  registry. ``metric_type`` is supplied by the caller at query time.
* ``omop_cdm_engine`` is **optional** on the reader interface.  When provided,
  KNN results are enriched with ``concept_name`` from the CDM.  When absent,
  ``NearestConceptMatch.concept_name`` is ``None``.
  ``domain_id``, ``vocabulary_id``, ``is_standard``, and ``is_active`` come
  from the embedding table directly and are always populated regardless.
  These attributes are necessary to be in the embedding table for filtering
  without round-tripping to the CDM, and are populated from the CDM at ingestion time.
* ``omop_cdm_engine`` is **required** for ingestion methods
  (``embed_and_upsert_concepts``) because concept metadata must be fetched
  from the CDM to populate the embedding table filter columns.
* Model calling is entirely ``omop_llm.ModelBackend``'s job (construction,
  canonicalization, dimension discovery, batched embedding calls). The writer
  interface builds one at construction time and owns only the domain logic
  ``omop_llm`` doesn't: role-based text prefixing and shape validation.
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
from oa_configurator import ResolvedModel
from omop_llm import EmbeddingRole, ModelBackend, build_model_backend_from_resolved
from omop_llm.providers import canonical_model_name as resolve_canonical_model_name

from omop_emb.utils.cdm import (
    count_missing_concepts,
    fetch_cdm_concepts_for_filter,
    iter_cdm_concepts_for_filter,
)
from omop_emb.backends.base_backend import (
    ConceptEmbeddingRecord,
    EmbeddingBackend,
    EmbeddingModelRecord,
)
from omop_emb.backends.index_config import IndexConfig
from omop_emb.config import BackendType, MetricType
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter, NearestConceptMatch

if TYPE_CHECKING:
    from omop_emb.storage.faiss import FAISSCache

logger = logging.getLogger(__name__)


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
        enriched with ``concept_name`` from the CDM.  When absent,
        ``concept_name`` is ``None``. ``domain_id``, ``vocabulary_id``,
        ``is_standard``, and ``is_active`` are populated directly from the
        embedding table by the backend.
    model : str
        Model name in canonical form.
    provider_type : str, optional
        omop-llm provider key. Defaults to ``'ollama'``.
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
        provider_type: str = "ollama",
        k: int = EmbeddingBackend.DEFAULT_K_NEAREST,
        faiss_cache_dir: Optional[str] = None,
    ):
        canonical_model_name = resolve_canonical_model_name(provider_type, model)

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
    def provider_type(self) -> str:
        return self._provider_type

    # ------------------------------------------------------------------
    # Embedding generation (static -> no interface instance required)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_embeddings(
        model_backend: ModelBackend,
        text: Union[str, List[str], Tuple[str, ...]],
        *,
        role: EmbeddingRole,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Embed *text* against *model_backend*, validating the resulting shape.

        Callable without an interface instance for callers that hold a
        ``ModelBackend`` directly (e.g. on-the-fly query embedding against a
        registry-resolved model, decoupled from any specific write session).
        Role-prefix application is ``model_backend``'s own job (see
        ``ModelBackend.embed_texts``'s ``role`` parameter); this method only
        forwards *role* and validates the returned shape.

        Parameters
        ----------
        model_backend : omop_llm.ModelBackend
            Backend to generate embeddings with.
        text : str | list[str] | tuple[str, ...]
            Input text(s) to embed.
        role : omop_llm.EmbeddingRole
            Role of the input text(s), used by *model_backend* to apply a
            configured prefix.
        batch_size : int, optional
            Forwarded to ``ModelBackend.embed_texts``.

        Returns
        -------
        np.ndarray
            2-D float array of shape ``(n_texts, embedding_dim)``.
        """
        if isinstance(text, str):
            text_tuple: Tuple[str, ...] = (text,)
        else:
            text_tuple = tuple(text)

        vectors = model_backend.embed_texts(list(text_tuple), role=role, batch_size=batch_size)

        result = np.array(vectors)
        if result.ndim != 2:
            raise RuntimeError(f"Expected 2-D embedding array, got shape {result.shape}")
        if result.shape[0] != len(text_tuple):
            raise RuntimeError(f"Expected {len(text_tuple)} embeddings, got {result.shape[0]}")
        return result

    # ------------------------------------------------------------------
    # Registry queries
    # ------------------------------------------------------------------

    @staticmethod
    def list_registered_models(
        backend: EmbeddingBackend,
        provider_type: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> tuple[EmbeddingModelRecord, ...]:
        """List models registered by the backend, optionally filtered by provider and/or model name.

        Parameters
        ----------
        backend : EmbeddingBackend
            Backend to query.
        provider_type : str, optional
            omop-llm provider key that serves the model.
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

    def get_embedding_count_by_vocabulary(self) -> Mapping[str, int]:
        """Return stored embedding counts grouped by vocabulary_id.

        Returns
        -------
        Mapping[str, int]
            ``vocabulary_id`` to embedding count, for every vocabulary with at
            least one stored embedding.
        """
        return self._backend.get_embedding_count_by_vocabulary(
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
            Shape ``(Q, ≤k)``. A row has fewer than *k* entries only when fewer
            than *k* stored concepts exist that match *concept_filter* (or
            exist at all). ``domain_id``, ``vocabulary_id``, ``is_standard``,
            and ``is_active`` are always populated from the embedding table.
            ``concept_name`` is ``None`` if no CDM engine was provided to the
            interface.
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
        model_backend: Optional[ModelBackend] = None,
        *,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        batch_size: Optional[int] = None,
        k: Optional[int] = None,
        faiss_index_config: Optional[IndexConfig] = None,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        """Embed *query_texts* then search for nearest concepts."""
        if model_backend is None:
            raise ValueError(
                "model_backend is required (EmbeddingReaderInterface has no default backend)."
            )
        if isinstance(query_texts, str):
            query_texts = (query_texts,)
        query_embeddings = self.generate_embeddings(
            model_backend,
            tuple(query_texts),
            role=EmbeddingRole.QUERY,
            batch_size=batch_size,
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

    def get_indexed_concept_ids(
        self,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> set[int]:
        """Return every stored concept_id matching *concept_filter*.

        Parameters
        ----------
        concept_filter : EmbeddingConceptFilter, optional
            Filter constraints to evaluate (domain, vocabulary, standard,
            concept ID allowlist). When omitted, every stored concept_id is
            returned.

        Returns
        -------
        set[int]
        """
        return self._backend.get_concept_ids_matching_filter(
            model_name=self.canonical_model_name,
            metric_type=self._metric_type,
            concept_filter=concept_filter or EmbeddingConceptFilter(),
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
        standard_concept, and invalid_reason, all columns needed for both
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

        Notes
        -----
        Enriched concepts (if a CDM engine is provided) will have the following attributes:
            - `concept_name` (str): The name of the concept from the CDM.
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

    Builds and owns an ``omop_llm.ModelBackend`` directly: there is no
    separate client object between this interface and the model backend.

    Parameters
    ----------
    backend : EmbeddingBackend
        Pre-constructed backend.
    metric_type : MetricType
        Distance metric for the table.
    resolved_model : oa_configurator.ResolvedModel
        A model resolved via ``oa_configurator.Resolver.resolve_model()``,
        e.g. ``Resolver(load_stack_config()).resolve_model(cfg.embedding_model_name)``.
        Provider, connection details, ``embedding_dim``, and
        ``document_prefix``/``query_prefix`` all come from this, not from
        ``omop-emb``'s own config.
    embedding_batch_size : int, optional
        Default number of texts per API call. Default is 32.
    omop_cdm_engine : Engine, optional
        CDM engine used for result enrichment.  Pass to write methods directly
        when needed for ingestion.
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        metric_type: MetricType,
        resolved_model: ResolvedModel,
        embedding_batch_size: int = 32,
        *,
        omop_cdm_engine: Optional[Engine] = None,
    ):
        self._model_backend: ModelBackend = build_model_backend_from_resolved(resolved_model)
        self._embedding_batch_size = embedding_batch_size
        self._embedding_dim: Optional[int] = None

        super().__init__(
            backend=backend,
            metric_type=metric_type,
            omop_cdm_engine=omop_cdm_engine,
            model=self._model_backend.model,
            provider_type=self._model_backend.provider,
        )

        logger.info(
            f"{EmbeddingWriterInterface.__name__} initialised for model={self.canonical_model_name!r}.\n"
            f"Provider: {self._model_backend.provider!r}"
        )

    @property
    def model_backend(self) -> ModelBackend:
        return self._model_backend

    @property
    def embedding_dim(self) -> int:
        """Embedding vector dimension, resolved on first access and cached.

        Delegates entirely to ``ModelBackend.dimensions()``'s three-tier
        lookup (configured override, provider fast path, live probe); the
        configured override was already passed in as ``configuration`` at
        construction time.
        """
        if self._embedding_dim is None:
            self._embedding_dim = self._model_backend.dimensions()
        return self._embedding_dim

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
            provider_type=self._model_backend.provider,
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
        role: EmbeddingRole,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        return self.generate_embeddings(
            self._model_backend,
            texts,
            role=role,
            batch_size=batch_size if batch_size is not None else self._embedding_batch_size,
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
            role=EmbeddingRole.DOCUMENT,
        )
        self.upsert_concept_embeddings(records=records, embeddings=embeddings)
        return embeddings

    def get_nearest_concepts_from_query_texts(
        self,
        query_texts: Union[str, Tuple[str, ...], List[str]],
        model_backend: Optional[ModelBackend] = None,
        *,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        batch_size: Optional[int] = None,
        k: Optional[int] = None,
        faiss_index_config: Optional[IndexConfig] = None,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        return super().get_nearest_concepts_from_query_texts(
            query_texts=query_texts,
            model_backend=model_backend if model_backend is not None else self._model_backend,
            concept_filter=concept_filter,
            batch_size=batch_size,
            k=k,
            faiss_index_config=faiss_index_config,
        )
