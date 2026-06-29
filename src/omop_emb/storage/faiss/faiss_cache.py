"""FAISS sidecar cache providing backend-agnostic read-acceleration layer,
built from local indices and (eventually) GPU support.

``FAISSCache`` is NOT a storage backend, and is NOT the source of
truth for embeddings. It builds on-disk FAISS indices directly from a live
:class:`~omop_emb.backends.base_backend.EmbeddingBackend` (see
:meth:`FAISSCache.build_from_backend`) for lower-latency approximate
search. ``faiss-cpu`` is the only optional dependency.

Disk layout
-----------
Each model gets its own sub-directory inside ``cache_dir``::

    <cache_dir>/<safe_model_name>/
        <index_type>_<metric_type>.faiss        : IndexIDMap wrapping Flat index
        <index_type>_<metric_type>.json         : per-index: exported_at, row_count, index_config
        ...

Staleness is tracked per-index via the ``.json`` sidecar: ``is_fresh()``
compares ``exported_at`` against ``model_record.updated_at``. A stale index
is never queried.


FAISS-only index configs
------------------------
:class:`IVFFlatIndexConfig` and :class:`IVFPQIndexConfig` are defined here for
FAISS-specific acceleration.  They subclass the backend
:class:`~omop_emb.backends.index_config.IndexConfig` with ``IndexType.IVFFLAT``
/ ``IndexType.IVFPQ`` and are NOT officially supported at the moment.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    import faiss
except ImportError as _faiss_err:
    raise ImportError(
        "FAISSCache requires the 'faiss-cpu' optional dependency. "
        "Install it with: pip install omop-emb[faiss-cpu]"
    ) from _faiss_err

from omop_emb.config import MetricType, IndexType
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter, NearestConceptMatch
from omop_emb.backends.base_backend import EmbeddingBackend
from omop_emb.backends.index_config import IndexConfig
from omop_emb.model_registry.model_registry_types import EmbeddingModelRecord
from omop_emb.storage.embedding_bundle import ExportMetadata, stream_embedding_batches

logger = logging.getLogger(__name__)


# Metrics supported by FAISS (L1, HAMMING, etc. are not available).
_FAISS_SUPPORTED_METRICS = frozenset({MetricType.L2, MetricType.COSINE})


# ---------------------------------------------------------------------------
# FAISS-only index configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class IVFFlatIndexConfig(IndexConfig):
    """Inverted-file flat index (FAISS only).

    Good for ~500 K-10 M vectors. Approximate search; faster than flat scan.

    Parameters
    ----------
    n_lists : int
        Number of Voronoi cells. Rule of thumb: ``sqrt(N)`` for N vectors.
    n_probe : int
        Cells visited at query time (higher → better recall, slower search).
    metric_type : MetricType
        Distance metric. Must be L2 or COSINE.
    """

    metric_type: MetricType
    index_type: IndexType = IndexType.IVFFLAT
    n_lists: int = 100
    n_probe: int = 10

    def _validate_metric_type_after_init(self) -> None:
        raise NotImplementedError("IVFFlatIndexConfig is currently not supported")


@dataclass(frozen=True, kw_only=True)
class IVFPQIndexConfig(IndexConfig):
    """Inverted-file product-quantisation index (FAISS only).

    Memory-efficient for 10 M+ vectors. Lossy compression.

    Parameters
    ----------
    n_lists : int
        Number of Voronoi cells.
    n_subquantizers : int
        Number of sub-quantisers (``m``).  Must divide ``dimensions`` evenly.
    n_bits : int
        Bits per sub-quantiser code (4 or 8).
    n_probe : int
        Cells visited at query time.
    metric_type : MetricType
        Distance metric. Must be L2 or COSINE.
    """

    metric_type: MetricType
    index_type: IndexType = IndexType.IVFPQ
    n_lists: int = 100
    n_subquantizers: int = 8
    n_bits: int = 8
    n_probe: int = 10

    def _validate_metric_type_after_init(self) -> None:
        raise NotImplementedError("IVFPQIndexConfig is currently not supported")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_faiss_metric(metric_type: MetricType) -> None:
    """Raise ``ValueError`` if *metric_type* is not supported by FAISS."""
    if metric_type not in _FAISS_SUPPORTED_METRICS:
        raise ValueError(
            f"FAISS only supports {[m.value for m in _FAISS_SUPPORTED_METRICS]} metrics, "
            f"got {metric_type.value!r}."
        )


def _safe_model_name(model_name: str) -> str:
    name = model_name.lower().strip()
    sanitized = re.sub(r"[^\w]+", "_", name)
    return re.sub(r"_+", "_", sanitized).strip("_")


def _index_key(metric_type: MetricType, index_config: IndexConfig) -> str:
    """Return the file stem for a given metric+index combo, e.g. ``'flat_cosine'``."""
    return f"{index_config.index_type.value}_{metric_type.value}"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# FAISSCache
# ---------------------------------------------------------------------------


class FAISSCache:
    """Model-level FAISS sidecar cache.

    A single instance manages **all** FAISS indices for one model under
    ``model_dir``.  The metric and index type are supplied per-call so that
    multiple indices can coexist and be queried independently.

    Notes
    -----
    Caches loaded index files in-memory, keyed on ``(metric_type, index_config)``,
    for faster subsequent queries. Both caches use LRU eviction with configurable
    size caps so a long-lived instance cannot accumulate unbounded memory.

    Parameters
    ----------
    model_name : str
        Registered canonical model name.
    cache_dir : Path | str
        Root cache directory. Each model gets its own sub-directory.
    max_cached_indices : int
        Maximum number of FAISS index objects to hold in memory simultaneously.
        Least-recently-used entries are evicted when the cap is reached.
    max_cached_filters : int
        Maximum number of concept-filter result sets to cache. Each entry is a
        numpy int64 array of matching concept IDs. LRU eviction applies.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: "Path | str",
        max_cached_indices: int = 4,
        max_cached_filters: int = 4,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = Path(cache_dir).expanduser().resolve()
        self._max_cached_indices = max_cached_indices
        self._max_cached_filters = max_cached_filters
        self._index_cache: OrderedDict[Tuple[MetricType, IndexConfig], "faiss.Index"] = OrderedDict()
        self._filter_cache: OrderedDict[EmbeddingConceptFilter, Optional[np.ndarray]] = OrderedDict()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def model_dir(self) -> Path:
        """Sub-directory for this model's cache files."""
        return self._cache_dir / _safe_model_name(self._model_name)

    def faiss_path(self, metric_type: MetricType, index_config: IndexConfig) -> Path:
        """Path to the ``.faiss`` index file for one metric+index combo."""
        return self.model_dir / f"{_index_key(metric_type, index_config)}.faiss"

    def json_path(self, metric_type: MetricType, index_config: IndexConfig) -> Path:
        """Path to the per-index ``.json`` staleness sidecar."""
        return self.model_dir / f"{_index_key(metric_type, index_config)}.json"

    def metadata_path(self) -> Path:
        """Path to a legacy ``metadata.npz`` file, if one exists.

        ``build_from_backend()`` no longer writes this file. Concept-filter
        predicates are now evaluated live against the backend (see
        :meth:`EmbeddingBackend.get_concept_ids_matching_filter`). This path
        is kept only so ``cli_legacy.py``'s ``import-legacy-faiss-cache``
        command can still locate ``metadata.npz`` on caches built before
        this change, for migration.
        """
        return self.model_dir / "metadata.npz"

    # ------------------------------------------------------------------
    # Staleness
    # ------------------------------------------------------------------

    def is_fresh(
        self,
        model_record: EmbeddingModelRecord,
        metric_type: MetricType,
        index_config: IndexConfig,
    ) -> bool:
        """Return ``True`` if this index exists and is up-to-date.

        Checks:

        1. ``{key}.faiss`` and ``{key}.json`` both exist.
        2. ``{key}.json`` is parseable with a non-negative ``row_count``.
        3. ``exported_at`` is strictly newer than ``model_record.updated_at``
           (when available).
        """
        if not self.faiss_path(metric_type, index_config).exists():
            return False

        json_path = self.json_path(metric_type, index_config)
        if not json_path.exists():
            return False

        try:
            meta = ExportMetadata.from_json(json_path.read_text())
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            return False

        if meta.row_count < 0:
            return False

        if model_record.updated_at is not None and meta.exported_at:
            try:
                exported_at = datetime.fromisoformat(meta.exported_at)
                if exported_at <= model_record.updated_at:
                    return False
            except ValueError:
                pass

        return True

    def staleness_info(
        self,
        model_record: EmbeddingModelRecord,
        metric_type: MetricType,
        index_config: IndexConfig,
    ) -> dict:
        """Return a summary dict describing the staleness state of one index."""
        meta: Optional[ExportMetadata] = None
        json_path = self.json_path(metric_type, index_config)
        if json_path.exists():
            try:
                meta = ExportMetadata.from_json(json_path.read_text())
            except (json.JSONDecodeError, OSError, ValueError, KeyError):
                pass
        return {
            "is_fresh": self.is_fresh(model_record, metric_type, index_config),
            "exported_at": meta.exported_at if meta else None,
            "cached_row_count": meta.row_count if meta else None,
            "model_updated_at": model_record.updated_at.isoformat()
            if model_record.updated_at
            else None,
            "cache_dir": str(self.model_dir),
        }

    # ------------------------------------------------------------------
    # Build from backend
    # ------------------------------------------------------------------

    def build_from_backend(
        self,
        backend: EmbeddingBackend,
        metric_type: MetricType,
        index_config: IndexConfig,
        batch_size: int = 100_000,
    ) -> None:
        """Build (or rebuild) one FAISS index directly from the live backend.

        Streams every embedding for this cache's model out of *backend* in
        bounded-memory batches (see
        :func:`omop_emb.storage.embedding_bundle.stream_embedding_batches`),
        normalizing a transient per-batch copy when *metric_type* is COSINE.
        Peak memory stays close to one batch's worth regardless of how many
        rows the model has.

        Writes (or overwrites):

        * ``<index_type>_<metric_type>.faiss``: the FAISS index for this metric+index combo.
        * ``<index_type>_<metric_type>.json``: per-index staleness metadata.

        Raises
        ------
        ValueError
            If ``metric_type`` is not supported by FAISS, the model isn't
            registered, or it has no stored embeddings.
        """
        _validate_faiss_metric(metric_type)

        record = backend.get_registered_model(model_name=self._model_name)
        if record is None:
            raise ValueError(f"Model '{self._model_name}' is not registered in the backend.")

        self.model_dir.mkdir(parents=True, exist_ok=True)

        all_ids = sorted(
            backend.get_all_stored_concept_ids(model_name=self._model_name, metric_type=metric_type)
        )
        if not all_ids:
            raise ValueError(
                f"No embeddings found for '{self._model_name}' (metric={metric_type.value}). "
                "Nothing to index."
            )

        n = len(all_ids)
        dimensions = record.dimensions

        inner = self._create_inner_index(
            dimensions, np.empty((0, dimensions), dtype=np.float32), metric_type, index_config
        )
        index = faiss.IndexIDMap(inner)

        row_count = 0
        for batch in tqdm(
            stream_embedding_batches(backend, self._model_name, metric_type, all_ids, batch_size),
            total=(n + batch_size - 1) // batch_size,
            desc="Building FAISS index from backend",
        ):
            batch_vecs = batch.embeddings
            if metric_type == MetricType.COSINE:
                batch_vecs = batch_vecs.copy()
                faiss.normalize_L2(batch_vecs)

            index.add_with_ids(batch_vecs, batch.concept_ids)  # type: ignore

            row_count += len(batch.concept_ids)

        faiss_path = self.faiss_path(metric_type, index_config)
        faiss.write_index(index, str(faiss_path))
        logger.info("Built FAISS index at '%s' (%d vectors).", faiss_path, row_count)

        meta = ExportMetadata(
            model_name=self._model_name,
            dimensions=dimensions,
            metric_type=metric_type,
            provider_type=record.provider_type,
            index_config=index_config,
            row_count=row_count,
            exported_at=_now_iso(),
        )
        self.json_path(metric_type, index_config).write_text(meta.to_json())
        logger.info(
            "FAISS cache build complete: %d vectors, metric=%s, index=%s, dir='%s'.",
            row_count,
            metric_type.value,
            index_config.index_type.value,
            self.model_dir,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int,
        metric_type: MetricType,
        index_config: IndexConfig,
        *,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        backend: Optional[EmbeddingBackend] = None,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        """Search a specific FAISS index for nearest concepts.

        All ``EmbeddingConceptFilter`` fields are applied as true pre-filters
        (via ``IDSelectorBatch``) using the supplied backend.

        Parameters
        ----------
        query_embeddings : ndarray
            Float32 array of shape ``(Q, D)``.
        k : int
            Maximum number of nearest neighbours per query.
        metric_type : MetricType
            Distance metric identifying which on-disk index to load.
        index_config : IndexConfig
            Index configuration identifying which on-disk index to load.
        concept_filter : EmbeddingConceptFilter, optional
            Applied as a pre-filter over the live backend.
        backend : EmbeddingBackend, optional
            Required when concept_filter is set and non-empty. Used to
            resolve the filter to a concept-ID set via
            :meth:`EmbeddingBackend.get_concept_ids_matching_filter`.

        Returns
        -------
        tuple[tuple[NearestConceptMatch, ...], ...]
            Shape ``(Q, <=k)``. CDM enrichment is the caller's responsibility.

        Notes
        -----
        When the underlying index is HNSW, ``IDSelectorBatch`` pre-filtering
        may lower recall: graph traversal can skip valid candidates that would
        only be reached through excluded nodes.
        """
        from omop_emb.utils.embedding_utils import get_similarity_from_distance

        query = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        if metric_type == MetricType.COSINE:
            faiss.normalize_L2(query)

        index = self._load_index(metric_type, index_config)

        params = None
        if concept_filter is not None and not concept_filter.is_empty():
            if backend is None:
                raise ValueError("backend is required when concept_filter is set.")
            selector_ids = self._build_filter_selector_ids(concept_filter, backend, metric_type, index)
            if selector_ids is not None:
                sel = faiss.IDSelectorBatch(selector_ids)  # type: ignore[reportCallIssue]
                params = faiss.SearchParameters()
                params.sel = sel

        distances, ids_matrix = index.search(query, k, params=params)  # type: ignore

        results: list[tuple[NearestConceptMatch, ...]] = []
        for dist_row, id_row in zip(distances, ids_matrix):
            row_results = tuple(
                NearestConceptMatch(
                    concept_id=int(cid),
                    similarity=float(
                        get_similarity_from_distance(
                            self._to_metric_dist(float(dist), metric_type), metric_type
                        )
                    ),
                )
                for dist, cid in zip(dist_row, id_row)
                if cid != -1
            )
            results.append(row_results)
        return tuple(results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_filter_selector_ids(
        self,
        concept_filter: EmbeddingConceptFilter,
        backend: EmbeddingBackend,
        metric_type: MetricType,
        index: "faiss.Index",
    ) -> Optional[np.ndarray]:
        """Get selector IDs for a given concept filter to supply to
        FAISS ``SearchParameters.sel``.  Returns ``None`` if the filter matches
        everything (no selector needed).

        Results are cached in an LRU cache keyed on ``concept_filter`` alone
        (metric and index do not affect which concept IDs match a filter).

        Notes
        -----
        Only works if the real index is wrapped using an ``IndexIDMap``
        as no translation is needed between FAISS positions and concept IDs.
        """
        if concept_filter in self._filter_cache:
            self._filter_cache.move_to_end(concept_filter)
            return self._filter_cache[concept_filter]

        matching_ids = backend.get_concept_ids_matching_filter(
            model_name=self._model_name,
            metric_type=metric_type,
            concept_filter=concept_filter,
        )
        result = (
            None
            if len(matching_ids) >= index.ntotal
            else np.fromiter(matching_ids, dtype=np.int64, count=len(matching_ids))
        )
        self._filter_cache[concept_filter] = result
        if len(self._filter_cache) > self._max_cached_filters:
            self._filter_cache.popitem(last=False)
        return result

    def _create_inner_index(
        self,
        dimensions: int,
        train_vecs: np.ndarray,
        metric_type: MetricType,
        index_config: IndexConfig,
    ):
        """Create the unwrapped FAISS inner index from *index_config*."""
        from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig

        faiss_metric = (
            faiss.METRIC_INNER_PRODUCT
            if metric_type == MetricType.COSINE
            else faiss.METRIC_L2
        )

        if isinstance(index_config, FlatIndexConfig):
            if metric_type == MetricType.L2:
                return faiss.IndexFlatL2(dimensions)
            return faiss.IndexFlatIP(dimensions)

        if isinstance(index_config, HNSWIndexConfig):
            idx = faiss.IndexHNSWFlat(
                dimensions, index_config.num_neighbors, faiss_metric
            )
            idx.hnsw.efConstruction = index_config.ef_construction
            return idx

        if isinstance(index_config, IVFFlatIndexConfig):
            quantizer = (
                faiss.IndexFlatL2(dimensions)
                if metric_type == MetricType.L2
                else faiss.IndexFlatIP(dimensions)
            )
            idx = faiss.IndexIVFFlat(
                quantizer, dimensions, index_config.n_lists, faiss_metric
            )
            idx.train(train_vecs)  # type: ignore
            idx.nprobe = index_config.n_probe
            return idx

        if isinstance(index_config, IVFPQIndexConfig):
            quantizer = (
                faiss.IndexFlatL2(dimensions)
                if metric_type == MetricType.L2
                else faiss.IndexFlatIP(dimensions)
            )
            idx = faiss.IndexIVFPQ(
                quantizer,
                dimensions,
                index_config.n_lists,
                index_config.n_subquantizers,
                index_config.n_bits,
            )
            idx.train(train_vecs)  # type: ignore
            idx.nprobe = index_config.n_probe
            return idx

        raise ValueError(f"No FAISS index factory for {type(index_config).__name__}.")

    @staticmethod
    def _to_metric_dist(raw: float, metric_type: MetricType) -> float:
        """Convert raw FAISS search output to the distance convention expected by
        get_similarity_from_distance.

        COSINE: FAISS returns inner product (L2-normalised IndexFlatIP) ∈ [-1, 1].
                Convert to cosine distance: d = 1 - ip.
        L2:     IndexFlatL2 returns squared Euclidean distance. Take sqrt.
        Others: pass through unchanged.
        """
        if metric_type == MetricType.COSINE:
            return 1.0 - raw
        if metric_type == MetricType.L2:
            return math.sqrt(max(0.0, raw))
        return raw

    def _load_index(self, metric_type: MetricType, index_config: IndexConfig):
        from omop_emb.backends.index_config import HNSWIndexConfig

        cache_key = (metric_type, index_config)
        cached_index = self._index_cache.get(cache_key)
        if cached_index is not None:
            self._index_cache.move_to_end(cache_key)
            return cached_index

        logger.debug(
            "Loading FAISS index for metric=%s, index=%s from disk.",
            metric_type.value,
            index_config.index_type.value,
        )
        path = self.faiss_path(metric_type, index_config)
        if not path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{path}'. Run FAISSCache.export() first."
            )
        index = faiss.read_index(str(path))
        assert isinstance(index, faiss.IndexIDMap), (
            f"FAISS index at '{path}' is not an IndexIDMap; "
            "this codebase only ever writes IndexIDMap-wrapped indices."
        )

        if isinstance(index_config, HNSWIndexConfig):
            # efSearch is a runtime parameter: apply the caller's config value
            # regardless of what was baked into the file during build.
            # index.index returns a generic faiss::Index* SWIG pointer; downcast
            # to the concrete IndexHNSWFlat to expose the .hnsw property.
            inner = faiss.downcast_index(index.index)
            if isinstance(inner, faiss.IndexHNSW):
                inner.hnsw.efSearch = index_config.ef_search

        self._index_cache[cache_key] = index
        if len(self._index_cache) > self._max_cached_indices:
            self._index_cache.popitem(last=False)
        return index
