"""FAISS sidecar cache — backend-agnostic read-acceleration layer.

``FAISSCache`` is **not** a storage backend.  It exports vectors from any
``EmbeddingBackend`` (sqlite-vec or pgvector) into on-disk FAISS indices for
lower-latency search.  ``faiss-cpu`` and ``h5py`` are optional dependencies.

Metadata contract
-----------------
Staleness metadata is stored in the embedding registry's ``details`` JSON
under the key :const:`~omop_emb.backends.index_config.FAISS_CACHE_METADATA_KEY`::

    {
        "faiss_cache": {
            "exported_at":  "<ISO-8601 timestamp, UTC>",
            "row_count":    <int>,
            "cache_dir":    "<absolute path>",
        }
    }
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from itertools import batched
from pathlib import Path
from typing import Optional, Sequence, Tuple

from omop_emb.backends.base_backend import EmbeddingBackend, EmbeddingModelRecord
from omop_emb.backends.index_config import (
    FAISS_CACHE_METADATA_KEY,
    FlatIndexConfig,
    HNSWIndexConfig,
    IndexConfig,
)
from omop_emb.config import MetricType, ProviderType
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter, NearestConceptMatch

logger = logging.getLogger(__name__)


def _require_faiss():
    try:
        import faiss  # noqa: F401
        import h5py   # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "FAISSCache requires the 'faiss' optional dependency. "
            "Install it with: pip install omop-emb[faiss]"
        ) from exc


_HDF5_FILENAME = "embeddings.h5"
_EMBEDDINGS_DATASET = "embeddings"
_CONCEPT_IDS_DATASET = "concept_ids"


class FAISSCache:
    """In-memory FAISS acceleration cache backed by any ``EmbeddingBackend``.

    Exports embeddings from the backend into FAISS indices on disk.  Subsequent
    searches use FAISS directly (no SQL round-trip) while the backend remains
    the authoritative store.  CDM enrichment is the caller's responsibility.

    Parameters
    ----------
    backend : EmbeddingBackend
        The backend that owns the embeddings (sqlite-vec or pgvector).
    model_name : str
        Registered canonical model name.
    provider_type : ProviderType
        Provider that serves the model.
    metric_type : MetricType
        Distance metric of the embedding table to cache.
    cache_dir : Path | str
        Local directory where FAISS index files and HDF5 export are stored.
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        model_name: str,
        provider_type: ProviderType,
        metric_type: MetricType,
        cache_dir: "Path | str",
    ) -> None:
        _require_faiss()
        self._backend = backend
        self._model_name = model_name
        self._provider_type = provider_type
        self._metric_type = metric_type
        self._cache_dir = Path(cache_dir).expanduser().resolve()
        self._model_record: Optional[EmbeddingModelRecord] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_record(self) -> EmbeddingModelRecord:
        """Lazily resolved registry record for this cache's model.

        Raises
        ------
        ValueError
            If the model is not registered in the backend.
        """
        if self._model_record is None:
            record = self._backend.get_registered_model(
                model_name=self._model_name,
                provider_type=self._provider_type,
            )
            if record is None:
                raise ValueError(
                    f"Model '{self._model_name}' (provider={self._provider_type.value}) "
                    "is not registered in the backend."
                )
            self._model_record = record
        return self._model_record

    @property
    def cache_metadata(self) -> Optional[dict]:
        """Return the ``"faiss_cache"`` sub-dict from the registry, or ``None``."""
        details = self.model_record.metadata
        return details.get(FAISS_CACHE_METADATA_KEY) if details else None

    # ------------------------------------------------------------------
    # Staleness
    # ------------------------------------------------------------------

    def is_stale(self) -> bool:
        """Return ``True`` if the FAISS cache is absent or out-of-date."""
        meta = self.cache_metadata
        if meta is None:
            return True
        if not Path(meta.get("cache_dir", "")).exists():
            return True
        current_count = self._backend.get_embedding_count(
            model_name=self._model_name,
            provider_type=self._provider_type,
            metric_type=self._metric_type,
        )
        return current_count != meta.get("row_count", -1)

    def staleness_info(self) -> dict:
        """Return a summary dict describing the current staleness state."""
        meta = self.cache_metadata or {}
        return {
            "is_stale": self.is_stale(),
            "exported_at": meta.get("exported_at"),
            "cached_row_count": meta.get("row_count"),
            "current_row_count": self._backend.get_embedding_count(
                model_name=self._model_name,
                provider_type=self._provider_type,
                metric_type=self._metric_type,
            ),
            "cache_dir": meta.get("cache_dir"),
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, batch_size: int = 100_000) -> None:
        """Export all embeddings from the backend to a local FAISS index.

        Parameters
        ----------
        batch_size : int
            Number of concept IDs to fetch per backend call. Default 100 000.

        Notes
        -----
        Streams all concept IDs and embeddings via the backend's public API,
        writes them to an HDF5 snapshot under ``cache_dir``, builds a FAISS
        index for ``metric_type``, and updates the registry with staleness
        metadata. Invalidates the cached ``model_record`` so the next access
        picks up fresh metadata.
        """
        import numpy as np
        import h5py

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        record = self.model_record

        # Collect all stored concept IDs
        all_ids = sorted(self._backend.get_all_stored_concept_ids(
            model_name=self._model_name,
            provider_type=self._provider_type,
            metric_type=self._metric_type,
        ))
        if not all_ids:
            logger.warning("No embeddings found — FAISS export skipped.")
            return

        # Batch-fetch embeddings via the public backend API
        all_concept_ids: list[int] = []
        all_embeddings: list = []
        for id_batch in batched(all_ids, batch_size):
            emb_map = self._backend.get_embeddings_by_concept_ids(
                model_name=self._model_name,
                provider_type=self._provider_type,
                metric_type=self._metric_type,
                concept_ids=list(id_batch),
            )
            for cid in id_batch:
                if cid in emb_map:
                    all_concept_ids.append(cid)
                    all_embeddings.append(np.array(emb_map[cid], dtype=np.float32))

        concept_ids_arr = np.array(all_concept_ids, dtype=np.int64)
        embeddings_arr = np.vstack(all_embeddings).astype(np.float32)
        total_rows = len(concept_ids_arr)

        h5_path = self._cache_dir / _HDF5_FILENAME
        with h5py.File(h5_path, "w") as f:
            f.create_dataset(_EMBEDDINGS_DATASET, data=embeddings_arr, compression="gzip")
            f.create_dataset(_CONCEPT_IDS_DATASET, data=concept_ids_arr)
        logger.info(f"Wrote {total_rows:,} vectors to '{h5_path}'.")

        # Build FAISS index for this table's metric
        self._build_index(embeddings_arr, concept_ids_arr, self._metric_type, record.index_config)

        # Persist staleness metadata via the backend's public API
        self._backend.patch_model_metadata(
            model_name=self._model_name,
            provider_type=self._provider_type,
            key=FAISS_CACHE_METADATA_KEY,
            value={
                "exported_at": datetime.now(tz=timezone.utc).isoformat(),
                "row_count": total_rows,
                "cache_dir": str(self._cache_dir),
            },
        )
        # Invalidate cached model record so next access picks up fresh metadata
        self._model_record = None

        logger.info(
            f"FAISS export complete: {total_rows:,} vectors, "
            f"metric={self._metric_type.value}."
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embeddings,
        *,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        k: Optional[int] = None,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        """Search the local FAISS index for nearest concepts.

        Parameters
        ----------
        query_embeddings : ndarray
            Float32 array of shape ``(Q, D)``.
        concept_filter : EmbeddingConceptFilter, optional
            Only ``concept_ids`` is applied as a post-filter. Domain,
            vocabulary, and standard filters are not supported by FAISS; use
            the primary backend for those.
        k : int, optional
            Maximum number of nearest neighbours per query.

        Returns
        -------
        tuple[tuple[NearestConceptMatch, ...], ...]
            Shape ``(Q, <=k)``. CDM enrichment is the caller's responsibility.
        """
        import faiss
        import numpy as np
        from omop_emb.utils.embedding_utils import get_similarity_from_distance

        record = self.model_record
        effective_k = k or (concept_filter.limit if concept_filter else None) or EmbeddingBackend.DEFAULT_K_NEAREST

        query = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        if self._metric_type == MetricType.COSINE:
            faiss.normalize_L2(query)

        index = self._load_index(self._metric_type, record.index_config)

        if concept_filter is not None and concept_filter.concept_ids is not None:
            import h5py

            with h5py.File(self._cache_dir / _HDF5_FILENAME, "r") as f:
                all_stored_ids = f[_CONCEPT_IDS_DATASET][:]
            candidate_set = set(concept_filter.concept_ids)
            subset_positions = np.where(np.isin(all_stored_ids, list(candidate_set)))[0].astype(np.int64)
            sel = faiss.IDSelectorBatch(subset_positions)
            params = faiss.SearchParameters()
            params.sel = sel
            distances, ids_matrix = index.search(query, effective_k, params=params)
        else:
            distances, ids_matrix = index.search(query, effective_k)

        results: list[tuple[NearestConceptMatch, ...]] = []
        for dist_row, id_row in zip(distances, ids_matrix):
            row_results = tuple(
                NearestConceptMatch(
                    concept_id=int(cid),
                    similarity=get_similarity_from_distance(float(dist), self._metric_type),
                )
                for dist, cid in zip(dist_row, id_row)
                if cid != -1
            )
            results.append(row_results)
        return tuple(results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_path(self, metric_type: MetricType) -> Path:
        return self._cache_dir / f"{metric_type.value}_index.faiss"

    def _build_index(
        self,
        embeddings,
        concept_ids,
        metric_type: MetricType,
        index_config: IndexConfig,
    ) -> None:
        import faiss

        index = self._create_empty_index(metric_type, index_config)

        vecs = embeddings.copy() if metric_type == MetricType.COSINE else embeddings
        if metric_type == MetricType.COSINE:
            faiss.normalize_L2(vecs)

        index.add_with_ids(vecs, concept_ids)
        index_path = self._index_path(metric_type)
        faiss.write_index(index, str(index_path))
        logger.info(f"Built FAISS index for metric={metric_type.value}: '{index_path}'.")

    def _create_empty_index(self, metric_type: MetricType, index_config: IndexConfig):
        import faiss

        dim = self.model_record.dimensions
        if isinstance(index_config, FlatIndexConfig):
            if metric_type == MetricType.L2:
                return faiss.IndexFlatL2(dim)
            return faiss.IndexFlatIP(dim)

        if isinstance(index_config, HNSWIndexConfig):
            faiss_metric = (
                faiss.METRIC_INNER_PRODUCT if metric_type == MetricType.COSINE else faiss.METRIC_L2
            )
            idx = faiss.IndexHNSWFlat(dim, index_config.num_neighbors, faiss_metric)
            idx.hnsw.efConstruction = index_config.ef_construction
            return idx

        raise ValueError(f"No FAISS index factory for {type(index_config).__name__}.")

    def _load_index(self, metric_type: MetricType, index_config: IndexConfig):
        import faiss

        path = self._index_path(metric_type)
        if not path.exists():
            raise FileNotFoundError(
                f"FAISS index for metric='{metric_type.value}' not found at '{path}'. "
                "Run FAISSCache.export() first."
            )
        return faiss.read_index(str(path))
