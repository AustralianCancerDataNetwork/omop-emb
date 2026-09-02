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
from omop_emb.utils.embedding_utils import (
    CDMConceptFilter,
    EmbeddingConceptFilter,
    NearestConceptMatch,
)

if TYPE_CHECKING:
    from omop_emb.storage.faiss import FAISSCache

logger = logging.getLogger(__name__)


def _stored_metadata_matches(
    row: Row,
    stored: Mapping[str, object] | None,
) -> bool:
    if stored is None:
        return False
    return (
        str(row.domain_id) == str(stored["domain_id"])
        and str(row.vocabulary_id) == str(stored["vocabulary_id"])
        and bool(row.is_standard) == bool(stored["is_standard"])
        and bool(row.is_valid) == bool(stored["is_valid"])
    )


def _resolve_k(k: Optional[int], default: int) -> int:
    """Resolve the number of nearest neighbours to request.

    *k* wins when given; falls back to *default*. The resolved value must be
    positive.
    """
    resolved = default if k is None else k
    if resolved <= 0:
        raise ValueError("k must be greater than zero.")
    return resolved


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
        effective_k = _resolve_k(k, self._k)

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

    def get_similar_concepts(
        self,
        concept_ids: Union[int, Sequence[int]],
        k: Optional[int] = None,
        *,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        faiss_index_config: Optional[IndexConfig] = None,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        """Return nearest stored concepts for one or more already-embedded concepts.

        Convenience wrapper around :meth:`get_nearest_concepts`: resolves each
        of *concept_ids* to its own stored embedding (via
        :meth:`get_embeddings_by_concept_ids`) instead of requiring the caller
        to fetch and pass a raw vector. Each concept is excluded from its own
        row of results.

        Parameters
        ----------
        concept_ids : int or sequence of int
            One or more concept IDs to search neighbours for. Every ID must
            already have a stored embedding. A bare ``int`` is treated as a
            single-element sequence.
        k : int, optional
            Number of nearest neighbours to return per query concept (defaults
            to interface-level *k*).
        concept_filter : EmbeddingConceptFilter, optional
            In-DB pre-filter applied during KNN (domain, vocabulary, standard).
        faiss_index_config : IndexConfig, optional
            Required only if a FAISS cache is configured on this interface.

        Returns
        -------
        Tuple[Tuple[NearestConceptMatch, ...], ...]
            Shape ``(Q, ≤k)`` where ``Q == len(concept_ids)`` (``1`` for a bare
            ``int``), in the same order as *concept_ids*. Each row excludes its
            own query concept.

        Raises
        ------
        ValueError
            If *concept_ids* is empty, or any entry has no stored embedding.
        """
        ids = (concept_ids,) if isinstance(concept_ids, int) else tuple(concept_ids)
        if not ids:
            raise ValueError("concept_ids must be non-empty.")

        stored = self.get_embeddings_by_concept_ids(ids)
        missing = [cid for cid in ids if cid not in stored]
        if missing:
            raise ValueError(f"No stored embedding for concept_ids: {missing}")

        vectors = np.asarray([stored[cid] for cid in ids], dtype=np.float64)
        effective_k = _resolve_k(k, self._k)

        raw = self.get_nearest_concepts(
            vectors,
            concept_filter=concept_filter,
            k=effective_k + 1,  # +1 because we filter out the query concept itself from results
            faiss_index_config=faiss_index_config,
        )
        return tuple(
            tuple(m for m in matches if m.concept_id != cid)[:effective_k]
            for cid, matches in zip(ids, raw)
        )

    def get_joint_embedding(
        self,
        concept_ids: Tuple[int, ...],
        weights: Optional[Tuple[float, ...]] = None,
    ) -> np.ndarray:
        """Return the (optionally weighted) centroid of stored concept embeddings.

        Parameters
        ----------
        concept_ids : tuple of int
            Concept IDs whose stored embeddings should be combined. Must be
            non-empty, and every ID must already have a stored embedding for
            the interface's model.
        weights : tuple of float, optional
            Per-concept weight, same length as *concept_ids*. Defaults to an
            unweighted mean.

        Returns
        -------
        ndarray
            Shape ``(D,)`` centroid vector, suitable as a single query row for
            :meth:`get_nearest_concepts`. Not normalised: backends compute
            cosine distance directly from raw vectors (and the FAISS cache
            normalises query vectors internally for ``COSINE``), so this
            method has no normalisation to do regardless of ``metric_type``.

        Raises
        ------
        ValueError
            If *concept_ids* is empty, *weights* has a mismatched length,
            *weights* sums to zero, or any *concept_ids* entry has no stored
            embedding.
        """
        if not concept_ids:
            raise ValueError("concept_ids must be non-empty.")
        if weights is not None and len(weights) != len(concept_ids):
            raise ValueError(
                f"weights must have the same length as concept_ids "
                f"({len(weights)} != {len(concept_ids)})."
            )
        if weights is not None and sum(weights) == 0:
            raise ValueError("weights sum to zero; cannot compute a weighted average.")

        vectors_by_id = self.get_embeddings_by_concept_ids(concept_ids)
        missing = [cid for cid in concept_ids if cid not in vectors_by_id]
        if missing:
            raise ValueError(f"No stored embedding for concept_ids: {missing}")

        vectors = np.asarray(
            [vectors_by_id[cid] for cid in concept_ids], dtype=np.float64
        )
        return np.average(vectors, axis=0, weights=weights)

    # ------------------------------------------------------------------
    # Concepts without embedding (requires CDM)
    # ------------------------------------------------------------------

    def get_concepts_without_embedding(
        self,
        omop_cdm_engine: Engine,
        *,
        concept_filter: Optional[CDMConceptFilter] = None,
    ) -> Mapping[int, Row]:
        """Return CDM rows for concepts lacking embeddings, keyed by concept_id.

        Each row contains concept name and filter metadata, including canonical
        standardness and validity derived by omop-alchemy.
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
        concept_filter: Optional[CDMConceptFilter] = None,
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
        concept_filter: Optional[CDMConceptFilter] = None,
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

    def count_concepts_requiring_embedding(
        self,
        omop_cdm_engine: Engine,
        *,
        concept_filter: Optional[CDMConceptFilter] = None,
        batch_size: int = 10_000,
    ) -> int:
        """Count missing concepts and concepts with changed filter metadata."""

        return sum(
            len(batch)
            for batch in self.get_concepts_requiring_embedding_batched(
                omop_cdm_engine,
                concept_filter=concept_filter,
                batch_size=batch_size,
            )
        )

    def get_concepts_requiring_embedding_batched(
        self,
        omop_cdm_engine: Engine,
        *,
        batch_size: int,
        concept_filter: Optional[CDMConceptFilter] = None,
        limit: Optional[int] = None,
    ) -> Iterable[Mapping[int, Row]]:
        """Yield concepts whose embedding is missing or has stale metadata.

        CDM rows are streamed and stored metadata is fetched in bounded
        batches. Existing embeddings are reprocessed only when their domain,
        vocabulary, standardness, or validity differs from the CDM.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than zero.")

        candidate_batch: dict[int, Row] = {}
        pending_batch: dict[int, Row] = {}
        yielded = 0

        def pending_candidates() -> dict[int, Row]:
            concept_ids = tuple(candidate_batch)
            stored = self._backend.get_concept_filter_metadata(
                model_name=self.canonical_model_name,
                metric_type=self._metric_type,
                concept_ids=concept_ids,
            )
            return {
                concept_id: row
                for concept_id, row in candidate_batch.items()
                if not _stored_metadata_matches(row, stored.get(concept_id))
            }

        for row in iter_cdm_concepts_for_filter(concept_filter, omop_cdm_engine):
            candidate_batch[int(row.concept_id)] = row
            if len(candidate_batch) < batch_size:
                continue
            pending_batch.update(pending_candidates())
            candidate_batch = {}
            while len(pending_batch) >= batch_size:
                output = dict(list(pending_batch.items())[:batch_size])
                if limit is not None:
                    output = dict(list(output.items())[: limit - yielded])
                if not output:
                    return
                yield output
                yielded += len(output)
                for concept_id in output:
                    pending_batch.pop(concept_id)
                if limit is not None and yielded >= limit:
                    return

        if candidate_batch:
            pending_batch.update(pending_candidates())
        if pending_batch:
            output = pending_batch
            if limit is not None:
                output = dict(list(output.items())[: limit - yielded])
            if output:
                yield output

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
        concept_filter = CDMConceptFilter(concept_ids=tuple(unique_ids))
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
        e.g. ``Resolver.from_active_config().resolve_model(cfg.embedding_model_name)``.
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
                is_standard=bool(concept_meta[cid].is_standard)
                if cid in concept_meta
                else False,
                is_classification=(
                    bool(concept_meta[cid].is_classification)
                    if cid in concept_meta
                    else False
                ),
                is_valid=bool(concept_meta[cid].is_valid)
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
