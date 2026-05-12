"""FAISS sidecar cache providing backend-agnostic read-acceleration layer,
export to local indices and (eventually) GPU support.

``FAISSCache`` is **not** a storage backend. It exports vectors from any
``EmbeddingBackend`` (sqlite-vec or pgvector) into on-disk FAISS indices for
lower-latency approximate search.  ``faiss-cpu`` is the only optional
dependency.

Disk layout
-----------
Each model gets its own sub-directory inside ``cache_dir``::

    <cache_dir>/<safe_model_name>/
        metadata.npz                            : concept_ids, domain_ids, vocabulary_ids, is_standard, is_valid (overwritten every export)
        <index_type>_<metric_type>.faiss        : IndexIDMap wrapping Flat index
        <index_type>_<metric_type>.json         : per-index: exported_at, row_count, index_config
        ...


Multiple indices for the same model share ``metadata.npz``.  The file is
always overwritten on export so it reflects the most recent export.  Staleness
is tracked **per-index** via the ``.json`` sidecar: ``is_fresh()``  compares
``exported_at`` against ``model_record.updated_at``.  A stale index is never
queried, so any row-count mismatch with ``metadata.npz`` cannot produce wrong
results.


FAISS-only index configs
------------------------
:class:`IVFFlatIndexConfig` and :class:`IVFPQIndexConfig` are defined here for
FAISS-specific acceleration.  They subclass the backend
:class:`~omop_emb.backends.index_config.IndexConfig` with ``IndexType.IVFFLAT``
/ ``IndexType.IVFPQ`` and are **NOT** officially supported at the moment.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import batched
from pathlib import Path
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

from omop_emb.config import MetricType, IndexType, ProviderType
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter, NearestConceptMatch
from omop_emb.backends.index_config import IndexConfig
from omop_emb.backends.base_backend import EmbeddingBackend
from omop_emb.model_registry.model_registry_types import EmbeddingModelRecord

logger = logging.getLogger(__name__)

_METADATA_FILENAME = "metadata.npz"

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
# Cache metadata dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CacheMetadata:
    """Typed representation of a per-index ``.json`` sidecar file.

    Written by :meth:`FAISSCache.export` and read back by
    :meth:`FAISSCache.is_fresh`, :meth:`FAISSCache.staleness_info`, and
    :meth:`FAISSCache._load_meta`.

    Parameters
    ----------
    model_name : str
        Canonical model name (e.g. ``'nomic-embed-text:v1.5'``).
    dimensions : int
        Embedding vector dimensionality.
    metric_type : MetricType
        Distance metric used when building the index.
    index_config : dict
        Serialised index configuration.  The index type is already encoded in
        the filename and in this dict; no separate ``index_type`` field is
        needed.
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
    index_config: IndexConfig
    row_count: int
    exported_at: str
    model_updated_at: Optional[str]

    def to_json(self) -> str:
        return json.dumps(
            {
                "model_name": self.model_name,
                "dimensions": self.dimensions,
                "metric_type": self.metric_type.value,
                "index_config": self.index_config.to_dict(),
                "row_count": self.row_count,
                "exported_at": self.exported_at,
                "model_updated_at": self.model_updated_at,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "CacheMetadata":
        """Deserialise from a per-index ``.json`` sidecar string.

        Raises
        ------
        ValueError
            If the JSON is malformed or contains an unknown enum value.
        """
        d = json.loads(text)
        return cls(
            model_name=d.get("model_name", ""),
            dimensions=int(d.get("dimensions", 0)),
            metric_type=MetricType(d["metric_type"]),
            index_config=IndexConfig.from_dict(d.get("index_config", {})),
            row_count=int(d.get("row_count", -1)),
            exported_at=d.get("exported_at", ""),
            model_updated_at=d.get("model_updated_at"),
        )


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

    Parameters
    ----------
    model_name : str
        Registered canonical model name.
    cache_dir : Path | str
        Root cache directory. Each model gets its own sub-directory.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: "Path | str",
    ) -> None:
        self._model_name = model_name
        self._cache_dir = Path(cache_dir).expanduser().resolve()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def model_dir(self) -> Path:
        """Sub-directory for this model's cache files."""
        return self._cache_dir / _safe_model_name(self._model_name)

    def _faiss_path(self, metric_type: MetricType, index_config: IndexConfig) -> Path:
        return self.model_dir / f"{_index_key(metric_type, index_config)}.faiss"

    def _json_path(self, metric_type: MetricType, index_config: IndexConfig) -> Path:
        return self.model_dir / f"{_index_key(metric_type, index_config)}.json"

    def _metadata_path(self) -> Path:
        return self.model_dir / _METADATA_FILENAME

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

        1. ``{key}.faiss``, ``{key}.json``, and ``metadata.npz`` all exist.
        2. ``{key}.json`` is parseable with a non-negative ``row_count``.
        3. ``exported_at`` is strictly newer than ``model_record.updated_at``
           (when available).
        """
        if not self._faiss_path(metric_type, index_config).exists():
            return False
        if not self._metadata_path().exists():
            return False

        json_path = self._json_path(metric_type, index_config)
        if not json_path.exists():
            return False

        try:
            meta = CacheMetadata.from_json(json_path.read_text())
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
        meta: Optional[CacheMetadata] = None
        json_path = self._json_path(metric_type, index_config)
        if json_path.exists():
            try:
                meta = CacheMetadata.from_json(json_path.read_text())
            except (json.JSONDecodeError, OSError, ValueError, KeyError):
                pass
        return {
            "is_fresh": self.is_fresh(model_record, metric_type, index_config),
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
        backend: EmbeddingBackend,
        metric_type: MetricType,
        index_config: IndexConfig,
        batch_size: int = 100_000,
    ) -> None:
        """Export all embeddings from the backend to one FAISS index on disk.

        Writes (or overwrites):

        * ``metadata.npz``: shared concept metadata arrays.
        * ``<index_type>_<metric_type>.faiss`` the FAISS index for this metric+index combo.
        * ``<index_type>_<metric_type>.json``: per-index staleness metadata.

        Parameters
        ----------
        backend : EmbeddingBackend
            Authoritative backend to export from.
        metric_type : MetricType
            Distance metric. Must be L2 or COSINE.
        index_config : IndexConfig
            FAISS index configuration. Supported: ``FlatIndexConfig`` (exact)
            and ``HNSWIndexConfig`` (approximate).
        batch_size : int
            Concept IDs fetched per backend call.  Capped internally at 50 000
            per DB round-trip to stay below the PostgreSQL bind-parameter limit.

        Raises
        ------
        ValueError
            If ``metric_type`` is not supported by FAISS, or the model is not
            registered in the backend.
        """
        _validate_faiss_metric(metric_type)

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
            logger.warning("No embeddings found for '%s'. FAISS export skipped.", self._model_name)
            return

        concept_ids_list: list[int] = []
        embeddings_list: list[np.ndarray] = []
        domain_ids_list: list[str] = []
        vocabulary_ids_list: list[str] = []
        is_standard_list: list[bool] = []
        is_valid_list: list[bool] = []

        for id_batch in tqdm(
            batched(all_ids, batch_size),
            total=(len(all_ids) + batch_size - 1) // batch_size,
            desc="Batched export to FAISS"
        ):
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

        # Shared metadata. Overwrite on every export so it always reflects
        # the current set of concepts.  Stale per-index files are gated by
        # is_fresh() before any positional lookup into this array.
        logger.info("Saving concept metadata arrays (%d rows) to '%s'.", total_rows, self._metadata_path())
        np.savez(
            self.model_dir / "metadata",  # NumPy appends .npz
            concept_ids=concept_ids_arr,
            domain_ids=np.array(domain_ids_list, dtype=object),
            vocabulary_ids=np.array(vocabulary_ids_list, dtype=object),
            is_standard=np.array(is_standard_list, dtype=bool),
            is_valid=np.array(is_valid_list, dtype=bool),
        )
        logger.info("Saved concept metadata arrays (%d rows).", total_rows)

        self._build_and_write_index(embeddings_arr, concept_ids_arr, record.dimensions, metric_type, index_config)

        meta = CacheMetadata(
            model_name=self._model_name,
            dimensions=record.dimensions,
            metric_type=metric_type,
            index_config=index_config,
            row_count=total_rows,
            exported_at=_now_iso(),
            model_updated_at=record.updated_at.isoformat() if record.updated_at else None,
        )
        self._json_path(metric_type, index_config).write_text(meta.to_json())
        logger.info(
            "FAISS export complete: %d vectors, metric=%s, index=%s, dir='%s'.",
            total_rows, metric_type.value, index_config.index_type.value, self.model_dir,
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
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        """Search a specific FAISS index for nearest concepts.

        All ``EmbeddingConceptFilter`` fields are applied as pre-filters using
        the numpy metadata arrays in ``metadata.npz``.

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
        from omop_emb.utils.embedding_utils import get_similarity_from_distance

        query = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        if metric_type == MetricType.COSINE:
            faiss.normalize_L2(query)

        index = self._load_index(metric_type, index_config)

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
                    similarity=float(get_similarity_from_distance(float(dist), metric_type)),
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

        Positions are sequential 0-based indices into the inner FAISS index
        (not external concept IDs), suitable for ``IDSelectorBatch``.
        The ``metadata.npz`` arrays are parallel to FAISS internal rows:
        position *i* in the array corresponds to position *i* in the index.
        """
        meta_path = self._metadata_path()
        if not meta_path.exists():
            raise FileNotFoundError(
                f"FAISS metadata not found at '{meta_path}'. Run FAISSCache.export() first."
            )
        npz = np.load(meta_path, allow_pickle=True)
        concept_ids_arr = npz["concept_ids"]
        mask = np.ones(len(concept_ids_arr), dtype=bool)

        if concept_filter.concept_ids is not None:
            mask &= np.isin(
                concept_ids_arr,
                np.array(list(concept_filter.concept_ids), dtype=np.int64),
            )

        if concept_filter.domains:
            mask &= np.isin(npz["domain_ids"], list(concept_filter.domains))

        if concept_filter.vocabularies:
            mask &= np.isin(npz["vocabulary_ids"], list(concept_filter.vocabularies))

        if concept_filter.require_standard:
            mask &= npz["is_standard"]

        if concept_filter.require_active:
            mask &= npz["is_valid"]

        if mask.all():
            return None
        return np.where(mask)[0].astype(np.int64)

    def _build_and_write_index(
        self,
        embeddings: np.ndarray,
        concept_ids: np.ndarray,
        dimensions: int,
        metric_type: MetricType,
        index_config: IndexConfig,
    ) -> None:
        vecs = embeddings.copy() if metric_type == MetricType.COSINE else embeddings
        if metric_type == MetricType.COSINE:
            faiss.normalize_L2(vecs)

        inner = self._create_inner_index(dimensions, vecs, metric_type, index_config)

        index = faiss.IndexIDMap(inner)
        index.add_with_ids(vecs, concept_ids)  # type: ignore

        path = self._faiss_path(metric_type, index_config)
        faiss.write_index(index, str(path))
        logger.info("Built FAISS index at '%s' (%d vectors).", path, len(concept_ids))

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

    def _load_index(self, metric_type: MetricType, index_config: IndexConfig):
        path = self._faiss_path(metric_type, index_config)
        if not path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{path}'. Run FAISSCache.export() first."
            )
        index = faiss.read_index(str(path))
        return index

    def _load_meta(
        self, metric_type: MetricType, index_config: IndexConfig
    ) -> CacheMetadata:
        """Load and parse the per-index ``.json`` sidecar.

        Raises
        ------
        FileNotFoundError
            If the ``.json`` file does not exist.
        ValueError
            If the file is malformed or contains an unknown enum value.
        """
        json_path = self._json_path(metric_type, index_config)
        if not json_path.exists():
            raise FileNotFoundError(
                f"FAISS index metadata not found at '{json_path}'. "
                "Run FAISSCache.export() first."
            )
        return CacheMetadata.from_json(json_path.read_text())

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def import_to_backend(
        self,
        backend: "EmbeddingBackend",
        metric_type: MetricType,
        index_config: "IndexConfig",
        provider_type: ProviderType,
        *,
        force: bool = False,
        batch_size: int = 10_000,
    ) -> int:
        """Import embeddings from a FAISS index back into a backend.

        Reconstructs raw vectors from the on-disk FAISS index and upserts
        them into *backend* together with the concept metadata stored in
        ``metadata.npz``.

        Only ``FlatIndexConfig`` and ``HNSWIndexConfig`` support exact
        reconstruction.  IVF and PQ indices are lossy and will raise at the
        reconstruction step.

        Parameters
        ----------
        backend : EmbeddingBackend
            Target backend to write into.
        metric_type : MetricType
            Metric of the on-disk index to reconstruct from.
        index_config : IndexConfig
            Index structure of the on-disk index to reconstruct from.
        provider_type : ProviderType
            Provider type recorded in the registry if the model is not yet
            registered.
        force : bool
            When ``False`` (default), refuse if the model already has
            embeddings in the backend.  Set to ``True`` to overwrite.
        batch_size : int
            Number of vectors upserted per backend call.

        Returns
        -------
        int
            Number of concepts imported.

        Raises
        ------
        FileNotFoundError
            If the ``.faiss`` or ``metadata.npz`` files do not exist.
        RuntimeError
            If the FAISS index does not support reconstruction (e.g. IVF-PQ),
            or if the backend already has embeddings and ``force`` is ``False``.
        ValueError
            If the row count in ``metadata.npz`` does not match the FAISS
            index's ``ntotal``.
        """
        from omop_emb.utils.embedding_utils import ConceptEmbeddingRecord

        meta = self._load_meta(metric_type, index_config)

        meta_path = self._metadata_path()
        if not meta_path.exists():
            raise FileNotFoundError(
                f"FAISS metadata not found at '{meta_path}'. "
                "Run FAISSCache.export() first."
            )
        npz = np.load(meta_path, allow_pickle=True)
        concept_ids_arr = npz["concept_ids"]
        domain_ids_arr = npz["domain_ids"]
        vocabulary_ids_arr = npz["vocabulary_ids"]
        is_standard_arr = npz["is_standard"]
        is_valid_arr = npz["is_valid"]
        n = len(concept_ids_arr)

        faiss_path = self._faiss_path(metric_type, index_config)
        if not faiss_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{faiss_path}'. "
                "Run FAISSCache.export() first."
            )
        index = faiss.read_index(str(faiss_path))

        if index.ntotal != n:
            raise ValueError(
                f"Row count mismatch: metadata.npz has {n} concepts but "
                f"FAISS index '{faiss_path.name}' has {index.ntotal} vectors. "
                "Re-export to fix."
            )

        # Refuse to silently overwrite existing embeddings.
        if not force and backend.is_model_registered(model_name=self._model_name):
            existing = backend.get_embedding_count(
                model_name=self._model_name,
                metric_type=metric_type,
            )
            if existing > 0:
                raise RuntimeError(
                    f"Backend already has {existing} embeddings for "
                    f"'{self._model_name}'. Pass force=True to overwrite."
                )

        # Register if not already present.
        if not backend.is_model_registered(model_name=self._model_name):
            backend.register_model(
                model_name=self._model_name,
                provider_type=provider_type,
                dimensions=meta.dimensions,
            )

        # Reconstruct all vectors from the inner index in one call.
        vecs = np.empty((n, meta.dimensions), dtype=np.float32)
        try:
            index.index.reconstruct_n(0, n, vecs)
        except RuntimeError as exc:
            raise RuntimeError(
                f"FAISS index at '{faiss_path}' does not support exact "
                "reconstruction. Only FLAT and HNSW indices can be imported "
                f"back into a backend. Error: {exc}"
            ) from exc

        def _batches():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                records = [
                    ConceptEmbeddingRecord(
                        concept_id=int(concept_ids_arr[i]),
                        domain_id=str(domain_ids_arr[i]),
                        vocabulary_id=str(vocabulary_ids_arr[i]),
                        is_standard=bool(is_standard_arr[i]),
                        is_valid=bool(is_valid_arr[i]),
                    )
                    for i in range(start, end)
                ]
                yield records, vecs[start:end]

        backend.bulk_upsert_embeddings(
            model_name=self._model_name,
            metric_type=metric_type,
            batches=_batches(),
        )
        logger.info(
            "Imported %d vectors for '%s' (metric=%s) into backend.",
            n, self._model_name, metric_type.value,
        )
        return n
