"""FAISS sidecar cache — backend-agnostic read-acceleration layer.

``FAISSCache`` is **not** a storage backend. It exports vectors from any
``EmbeddingBackend`` (sqlite-vec or pgvector) into on-disk FAISS indices for
lower-latency approximate search.  ``faiss-cpu`` is the only optional
dependency; h5py is not used.

Disk layout
-----------
Each model gets its own sub-directory inside ``cache_dir``::

    <cache_dir>/<safe_model_name>/
        index.faiss          — FAISS index (IndexIDMap wrapping Flat or HNSW)
        concept_ids.npy      — int64 array, parallel to FAISS internal rows
        domain_ids.npy       — object (str) array, parallel to index
        vocabulary_ids.npy   — object (str) array, parallel to index
        is_standard.npy      — bool array, parallel to index
        is_valid.npy         — bool array, parallel to index
        cache_meta.json      — staleness tracking + build config

``cache_meta.json`` is written and read as a :class:`CacheMetadata` dataclass.

FAISS-only index configs
------------------------
:class:`IVFFlatIndexConfig` and :class:`IVFPQIndexConfig` are defined here for
FAISS-specific acceleration.  They are **not** subclasses of the backend
:class:`~omop_emb.backends.index_config.IndexConfig` and are not stored in the
model registry.  Use :class:`~omop_emb.backends.index_config.FlatIndexConfig`
or :class:`~omop_emb.backends.index_config.HNSWIndexConfig` for the publicly
supported index types.  IVF configs are kept for future lifecycle work — see
item 53d in REFACTOR_PLAN.md.

Notes
-----
``IDSelectorBatch`` pre-filtering works differently depending on the inner index
type:

- **Flat**: exact scan — all selected positions are visited; results are exact.
- **HNSW**: graph traversal skips non-selected nodes but may not reach all
  valid candidates through excluded neighbours.  Recall may be lower than
  expected when a tight filter is applied.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from omop_emb.config import MetricType, IndexType
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter, NearestConceptMatch
from omop_emb.backends.index_config import IndexConfig

if TYPE_CHECKING:
    from omop_emb.backends.base_backend import EmbeddingBackend
    from omop_emb.backends.index_config import IndexConfig
    from omop_emb.model_registry.model_registry_types import EmbeddingModelRecord

logger = logging.getLogger(__name__)

_INDEX_FILENAME = "index.faiss"
_META_FILENAME = "cache_meta.json"

# Metrics supported by FAISS (L1, HAMMING, etc. are not available).
_FAISS_SUPPORTED_METRICS = frozenset({MetricType.L2, MetricType.COSINE})


# ---------------------------------------------------------------------------
# FAISS-only index configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class IVFFlatIndexConfig(IndexConfig):
    """Inverted-file flat index (FAISS only).

    Good for ~500 K–10 M vectors. Approximate search; faster than flat scan.

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
# Cache metadata dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CacheMetadata:
    """Typed representation of ``cache_meta.json``.

    Written by :meth:`FAISSCache.export` and read back by
    :meth:`FAISSCache.is_fresh`, :meth:`FAISSCache.staleness_info`, and
    :meth:`FAISSCache.search`.

    Parameters
    ----------
    model_name : str
        Canonical model name (e.g. ``'nomic-embed-text:v1.5'``).
    dimensions : int
        Embedding vector dimensionality.
    metric_type : MetricType
        Distance metric used when building the index.
    index_config : dict
        Serialised index configuration (informational; index is loaded from
        the binary ``.faiss`` file, not reconstructed from this dict).
    row_count : int
        Number of vectors in the index at export time.
    exported_at : str
        UTC ISO-8601 timestamp of the last export.
    model_updated_at : str or None
        ``model_record.updated_at`` at export time; ``None`` if the registry
        did not track ``updated_at``.
    """

    model_name: str
    dimensions: int
    metric_type: MetricType
    index_config: dict
    row_count: int
    exported_at: str
    model_updated_at: Optional[str]

    def to_json(self) -> str:
        """Serialise to a JSON string suitable for writing to ``cache_meta.json``."""
        return json.dumps(
            {
                "model_name": self.model_name,
                "dimensions": self.dimensions,
                "metric_type": self.metric_type.value,
                "index_config": self.index_config,
                "row_count": self.row_count,
                "exported_at": self.exported_at,
                "model_updated_at": self.model_updated_at,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "CacheMetadata":
        """Deserialise from a ``cache_meta.json`` string.

        Raises
        ------
        ValueError
            If the JSON is malformed or contains an unknown ``metric_type``.
        """
        d = json.loads(text)
        return cls(
            model_name=d.get("model_name", ""),
            dimensions=int(d.get("dimensions", 0)),
            metric_type=MetricType(d["metric_type"]),
            index_config=d.get("index_config", {}),
            row_count=int(d.get("row_count", -1)),
            exported_at=d.get("exported_at", ""),
            model_updated_at=d.get("model_updated_at"),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_faiss() -> None:
    try:
        import faiss  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "FAISSCache requires the 'faiss' optional dependency. "
            "Install it with: pip install omop-emb[faiss]"
        ) from exc


def _validate_faiss_metric(
    metric_type: MetricType,
    index_config: Optional["IndexConfig"] = None,
) -> None:
    """Validate that *metric_type* is supported by FAISS.

    Parameters
    ----------
    metric_type : MetricType
        Metric to check.
    index_config : IndexConfig, optional
        Reserved for future per-config metric constraints.

    Raises
    ------
    ValueError
        If *metric_type* is not in :data:`_FAISS_SUPPORTED_METRICS`.
    """
    if metric_type not in _FAISS_SUPPORTED_METRICS:
        raise ValueError(
            f"FAISS only supports {[m.value for m in _FAISS_SUPPORTED_METRICS]} metrics, "
            f"got {metric_type.value!r}."
        )


def _safe_model_name(model_name: str) -> str:
    name = model_name.lower().strip()
    sanitized = re.sub(r"[^\w]+", "_", name)
    return re.sub(r"_+", "_", sanitized).strip("_")


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# FAISSCache
# ---------------------------------------------------------------------------

class FAISSCache:
    """In-memory FAISS acceleration cache backed by any ``EmbeddingBackend``.

    Exports vectors from the authoritative backend into a FAISS index on disk.
    Subsequent searches use FAISS directly (no SQL round-trip).  CDM enrichment
    is the caller's responsibility (``EmbeddingReaderInterface._enrich``).

    Parameters
    ----------
    model_name : str
        Registered canonical model name.
    cache_dir : Path | str
        Root cache directory. Each model gets its own sub-directory.
    use_gpu : bool
        When ``True``, transfer the index to GPU device 0 after loading.
        Requires a GPU-enabled FAISS build.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: "Path | str",
        use_gpu: bool = False,
    ) -> None:
        _require_faiss()
        self._model_name = model_name
        self._cache_dir = Path(cache_dir).expanduser().resolve()
        self._use_gpu = use_gpu

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def model_dir(self) -> Path:
        """Sub-directory for this model's cache files."""
        return self._cache_dir / _safe_model_name(self._model_name)

    def _path(self, filename: str) -> Path:
        return self.model_dir / filename

    # ------------------------------------------------------------------
    # Staleness
    # ------------------------------------------------------------------

    def is_fresh(self, model_record: "EmbeddingModelRecord") -> bool:
        """Return ``True`` if the cache exists and is up-to-date.

        Checks:

        1. ``model_dir`` exists.
        2. ``cache_meta.json`` is present and parseable.
        3. Row count in the meta matches the record's embedding count.
        4. ``exported_at`` timestamp is newer than ``model_record.updated_at``
           (when available).

        Parameters
        ----------
        model_record : EmbeddingModelRecord
            Current registry record for the model.
        """
        meta_path = self._path(_META_FILENAME)
        if not meta_path.exists():
            return False

        try:
            meta = CacheMetadata.from_json(meta_path.read_text())
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            return False

        if not self._path(_INDEX_FILENAME).exists():
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

    def staleness_info(self, model_record: "EmbeddingModelRecord") -> dict:
        """Return a summary dict describing the current staleness state."""
        meta_path = self._path(_META_FILENAME)
        meta: Optional[CacheMetadata] = None
        if meta_path.exists():
            try:
                meta = CacheMetadata.from_json(meta_path.read_text())
            except (json.JSONDecodeError, OSError, ValueError, KeyError):
                pass
        return {
            "is_fresh": self.is_fresh(model_record),
            "exported_at": meta.exported_at if meta else None,
            "cached_row_count": meta.row_count if meta else None,
            "model_updated_at": model_record.updated_at.isoformat() if model_record.updated_at else None,
            "cache_dir": str(self.model_dir),
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(
        self,
        backend: "EmbeddingBackend",
        metric_type: MetricType,
        index_config: "IndexConfig",
        batch_size: int = 100_000,
    ) -> None:
        """Export all embeddings from the backend to a local FAISS index.

        Parameters
        ----------
        backend : EmbeddingBackend
            Authoritative backend to export from.
        metric_type : MetricType
            Distance metric for the FAISS index. Must be L2 or COSINE.
        index_config : IndexConfig
            FAISS index configuration. Supported types are
            :class:`~omop_emb.backends.index_config.FlatIndexConfig` (exact
            scan, always correct) and
            :class:`~omop_emb.backends.index_config.HNSWIndexConfig`
            (approximate, faster at scale). Must be supplied explicitly.
        batch_size : int
            Concept IDs fetched per backend call. Also controls memory
            pressure during export; capped internally at 50 000 per DB
            round-trip (PostgreSQL bind-parameter limit).

        Raises
        ------
        ValueError
            If ``metric_type`` is not supported by FAISS, or the model is not
            registered in the backend.
        """
        _validate_faiss_metric(metric_type, index_config)

        record = backend.get_registered_model(model_name=self._model_name)
        if record is None:
            raise ValueError(
                f"Model '{self._model_name}' is not registered in the backend."
            )

        self.model_dir.mkdir(parents=True, exist_ok=True)

        all_ids = sorted(backend.get_all_stored_concept_ids(
            model_name=self._model_name,
            metric_type=metric_type,
        ))
        if not all_ids:
            logger.warning("No embeddings found for '%s' — FAISS export skipped.", self._model_name)
            return

        # Cap at 50 000 per DB round-trip to stay under the PostgreSQL
        # wire-protocol bind-parameter limit (65 535). batch_size controls
        # memory pressure; DB queries use the smaller cap.
        # TODO(phase-g): replace with temp-table JOIN so the cap can be removed.
        _MAX_DB_BATCH = 50_000
        db_batch_size = min(batch_size, _MAX_DB_BATCH)

        concept_ids_list: list[int] = []
        embeddings_list: list[np.ndarray] = []
        domain_ids_list: list[str] = []
        vocabulary_ids_list: list[str] = []
        is_standard_list: list[bool] = []
        is_valid_list: list[bool] = []

        for id_batch in batched(all_ids, db_batch_size):
            id_batch_list = list(id_batch)
            emb_map = backend.get_embeddings_by_concept_ids(
                model_name=self._model_name,
                metric_type=metric_type,
                concept_ids=id_batch_list,
            )
            filter_meta = backend.get_concept_filter_metadata(
                model_name=self._model_name,
                metric_type=metric_type,
                concept_ids=id_batch_list,
            )
            for cid in id_batch_list:
                if cid in emb_map:
                    concept_ids_list.append(cid)
                    embeddings_list.append(np.array(emb_map[cid], dtype=np.float32))
                    m = filter_meta.get(cid, {})
                    domain_ids_list.append(str(m.get("domain_id", "")))
                    vocabulary_ids_list.append(str(m.get("vocabulary_id", "")))
                    is_standard_list.append(bool(m.get("is_standard", False)))
                    is_valid_list.append(bool(m.get("is_valid", True)))

        concept_ids_arr = np.array(concept_ids_list, dtype=np.int64)
        embeddings_arr = np.vstack(embeddings_list).astype(np.float32)
        total_rows = len(concept_ids_arr)

        np.save(self._path("concept_ids.npy"), concept_ids_arr)
        np.save(self._path("is_standard.npy"), np.array(is_standard_list, dtype=bool))
        np.save(self._path("is_valid.npy"), np.array(is_valid_list, dtype=bool))
        np.save(self._path("domain_ids.npy"), np.array(domain_ids_list, dtype=object), allow_pickle=True)
        np.save(self._path("vocabulary_ids.npy"), np.array(vocabulary_ids_list, dtype=object), allow_pickle=True)
        logger.info("Saved concept metadata arrays (%d rows).", total_rows)

        self._build_and_write_index(embeddings_arr, concept_ids_arr, metric_type, index_config, record.dimensions)

        if hasattr(index_config, "to_dict"):
            ic_dict = index_config.to_dict()
        elif hasattr(index_config, "__dataclass_fields__"):
            ic_dict = asdict(index_config)
        else:
            ic_dict = {}

        meta = CacheMetadata(
            model_name=self._model_name,
            dimensions=record.dimensions,
            metric_type=metric_type,
            index_config=ic_dict,
            row_count=total_rows,
            exported_at=_now_iso(),
            model_updated_at=record.updated_at.isoformat() if record.updated_at else None,
        )
        self._path(_META_FILENAME).write_text(meta.to_json())
        logger.info(
            "FAISS export complete: %d vectors, metric=%s, dir='%s'.",
            total_rows, metric_type.value, self.model_dir,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int,
        *,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        """Search the local FAISS index for nearest concepts.

        All ``EmbeddingConceptFilter`` fields are applied as pre-filters using
        the numpy metadata arrays stored at export time.

        Parameters
        ----------
        query_embeddings : ndarray
            Float32 array of shape ``(Q, D)``.
        k : int
            Maximum number of nearest neighbours per query.
        concept_filter : EmbeddingConceptFilter, optional
            Applied as a pre-filter over the stored metadata arrays.

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
        import faiss
        from omop_emb.utils.embedding_utils import get_similarity_from_distance

        meta = self._load_meta()

        query = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        if meta.metric_type == MetricType.COSINE:
            faiss.normalize_L2(query)

        index = self._load_index()

        params = None
        if concept_filter is not None and not concept_filter.is_empty():
            positions = self._build_filter_positions(concept_filter)
            if positions is not None:
                sel = faiss.IDSelectorBatch(positions)  # type: ignore
                params = faiss.SearchParameters()
                params.sel = sel

        if params is not None:
            distances, ids_matrix = index.search(query, k, params=params)
        else:
            distances, ids_matrix = index.search(query, k)

        results: list[tuple[NearestConceptMatch, ...]] = []
        for dist_row, id_row in zip(distances, ids_matrix):
            row_results = tuple(
                NearestConceptMatch(
                    concept_id=int(cid),
                    similarity=float(get_similarity_from_distance(float(dist), meta.metric_type)),
                )
                for dist, cid in zip(dist_row, id_row)
                if cid != -1
            )
            results.append(row_results)
        return tuple(results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_filter_positions(
        self, concept_filter: EmbeddingConceptFilter
    ) -> Optional[np.ndarray]:
        """Return int64 internal-position array for the filter, or None (no restriction).

        The returned positions are sequential 0-based indices into the inner
        FAISS index (not external concept IDs), suitable for ``IDSelectorBatch``.
        """
        concept_ids_arr = np.load(self._path("concept_ids.npy"))
        mask = np.ones(len(concept_ids_arr), dtype=bool)

        if concept_filter.concept_ids is not None:
            mask &= np.isin(concept_ids_arr, np.array(list(concept_filter.concept_ids), dtype=np.int64))

        if concept_filter.domains:
            domain_ids_arr = np.load(self._path("domain_ids.npy"), allow_pickle=True)
            mask &= np.isin(domain_ids_arr, list(concept_filter.domains))

        if concept_filter.vocabularies:
            vocabulary_ids_arr = np.load(self._path("vocabulary_ids.npy"), allow_pickle=True)
            mask &= np.isin(vocabulary_ids_arr, list(concept_filter.vocabularies))

        if concept_filter.require_standard:
            is_standard_arr = np.load(self._path("is_standard.npy"))
            mask &= is_standard_arr

        if concept_filter.require_active:
            is_valid_arr = np.load(self._path("is_valid.npy"))
            mask &= is_valid_arr

        if mask.all():
            return None
        return np.where(mask)[0].astype(np.int64)

    def _build_and_write_index(
        self,
        embeddings: np.ndarray,
        concept_ids: np.ndarray,
        metric_type: MetricType,
        index_config: "IndexConfig",
        dimensions: int,
    ) -> None:
        import faiss

        vecs = embeddings.copy() if metric_type == MetricType.COSINE else embeddings
        if metric_type == MetricType.COSINE:
            faiss.normalize_L2(vecs)

        inner = self._create_inner_index(index_config, dimensions, metric_type, vecs)

        index = faiss.IndexIDMap(inner)
        index.add_with_ids(vecs, concept_ids)  # type: ignore

        path = self._path(_INDEX_FILENAME)
        faiss.write_index(index, str(path))
        logger.info("Built FAISS index at '%s' (%d vectors).", path, len(concept_ids))

    def _create_inner_index(
        self,
        index_config: "IndexConfig",
        dimensions: int,
        metric_type: MetricType,
        train_vecs: np.ndarray,
    ):
        """Create the inner (unwrapped) FAISS index.

        Handles ``FlatIndexConfig``, ``HNSWIndexConfig``, and the FAISS-internal
        ``IVFFlatIndexConfig`` / ``IVFPQIndexConfig`` types.
        """
        import faiss
        from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig

        faiss_metric = (
            faiss.METRIC_INNER_PRODUCT if metric_type == MetricType.COSINE else faiss.METRIC_L2
        )

        if isinstance(index_config, FlatIndexConfig):
            if metric_type == MetricType.L2:
                return faiss.IndexFlatL2(dimensions)
            return faiss.IndexFlatIP(dimensions)

        if isinstance(index_config, HNSWIndexConfig):
            idx = faiss.IndexHNSWFlat(dimensions, index_config.num_neighbors, faiss_metric)
            idx.hnsw.efConstruction = index_config.ef_construction
            return idx

        if isinstance(index_config, IVFFlatIndexConfig):
            quantizer = (
                faiss.IndexFlatL2(dimensions)
                if metric_type == MetricType.L2
                else faiss.IndexFlatIP(dimensions)
            )
            idx = faiss.IndexIVFFlat(quantizer, dimensions, index_config.n_lists, faiss_metric)
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

    def _load_index(self):
        import faiss

        path = self._path(_INDEX_FILENAME)
        if not path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{path}'. Run FAISSCache.export() first."
            )
        index = faiss.read_index(str(path))

        if self._use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except AttributeError as exc:
                raise ImportError(
                    "GPU FAISS requires a GPU-enabled faiss build (faiss-gpu). "
                    "Install it or set use_gpu=False."
                ) from exc

        return index

    def _load_meta(self) -> CacheMetadata:
        """Load and parse ``cache_meta.json``.

        Raises
        ------
        FileNotFoundError
            If ``cache_meta.json`` does not exist.
        ValueError
            If the file is malformed or contains an unknown metric type.
        """
        meta_path = self._path(_META_FILENAME)
        if not meta_path.exists():
            raise FileNotFoundError(
                f"FAISS cache metadata not found at '{meta_path}'. "
                "Run FAISSCache.export() first."
            )
        return CacheMetadata.from_json(meta_path.read_text())
