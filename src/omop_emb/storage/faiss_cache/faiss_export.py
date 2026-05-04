"""Low-level FAISS export and search helpers.

This module handles the mechanics of streaming embeddings from a
``PGVectorEmbeddingBackend`` into local HDF5 storage and building FAISS
indices.  It is an internal detail of :class:`~omop_emb.storage.faiss_cache.FAISSCache`
and should not be imported directly by user code.

FAISS and h5py are optional dependencies; this module raises ``ImportError``
with a helpful message if they are not installed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np

from omop_emb.config import MetricType
from omop_emb.storage.index_config import FlatIndexConfig, HNSWIndexConfig, IndexConfig

if TYPE_CHECKING:
    from omop_emb.storage.postgres.pg_backend import PGVectorEmbeddingBackend
    from omop_emb.storage.postgres.pg_registry import EmbeddingModelRecord

logger = logging.getLogger(__name__)

_HDF5_FILENAME = "embeddings.h5"
_EMBEDDINGS_DATASET = "embeddings"
_CONCEPT_IDS_DATASET = "concept_ids"


class FaissExporter:
    """Manages the HDF5 file and FAISS index files for one model's cache.

    Parameters
    ----------
    cache_dir : Path
        Directory holding ``embeddings.h5`` and ``.faiss`` index files.
    dimensions : int
        Vector dimensionality.
    index_config : IndexConfig
        Index type and parameters (used when building the FAISS index).
    """

    def __init__(
        self,
        cache_dir: Path,
        dimensions: int,
        index_config: IndexConfig,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._dimensions = dimensions
        self._index_config = index_config

    # ------------------------------------------------------------------
    # Export path
    # ------------------------------------------------------------------

    def export_from_backend(
        self,
        backend: "PGVectorEmbeddingBackend",
        model_record: "EmbeddingModelRecord",
        metric_types: List[MetricType],
        batch_size: int = 100_000,
    ) -> Tuple[np.ndarray, int]:
        """Stream embeddings from pgvector, write HDF5, build FAISS indices.

        Returns
        -------
        (concept_ids, total_rows)
            All exported concept IDs as a 1-D array, plus the total row count.
        """
        import h5py

        h5_path = self._h5_path()

        all_concept_ids: list[int] = []
        all_embeddings: list[np.ndarray] = []

        # Stream from pgvector in batches via direct SQL
        table = backend.get_embedding_table(
            model_name=model_record.model_name,
            index_type=model_record.index_type,
            provider_type=model_record.provider_type,
        )
        from sqlalchemy import select

        with backend.emb_session_factory() as session:
            stmt = select(table.concept_id, table.embedding).execution_options(
                yield_per=batch_size
            )
            for row in session.execute(stmt):
                all_concept_ids.append(int(row.concept_id))
                all_embeddings.append(np.array(row.embedding, dtype=np.float32))

        if not all_concept_ids:
            logger.warning("No embeddings found — FAISS export skipped.")
            return np.array([], dtype=np.int64), 0

        concept_ids_arr = np.array(all_concept_ids, dtype=np.int64)
        embeddings_arr = np.vstack(all_embeddings).astype(np.float32)
        total_rows = len(concept_ids_arr)

        # Write HDF5
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset(_EMBEDDINGS_DATASET, data=embeddings_arr, compression="gzip")
            f.create_dataset(_CONCEPT_IDS_DATASET, data=concept_ids_arr)

        logger.info(f"Wrote {total_rows:,} vectors to '{h5_path}'.")

        # Build FAISS indices
        for metric in metric_types:
            self._build_index(
                embeddings=embeddings_arr,
                concept_ids=concept_ids_arr,
                metric_type=metric,
            )

        return concept_ids_arr, total_rows

    # ------------------------------------------------------------------
    # Search path
    # ------------------------------------------------------------------

    def search(
        self,
        query_embeddings: np.ndarray,
        metric_type: MetricType,
        k: int,
        candidate_concept_ids: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search the on-disk FAISS index for nearest neighbors.

        Parameters
        ----------
        query_embeddings : ndarray
            Shape ``(Q, D)``, dtype float32.
        metric_type : MetricType
            Must match an index built during :meth:`export_from_backend`.
        k : int
            Number of nearest neighbors per query.
        candidate_concept_ids : tuple[int, ...], optional
            Restrict search to these concept IDs.

        Returns
        -------
        (distances, concept_ids)
            Both shape ``(Q, k)``.  ``concept_ids`` is -1 where fewer than
            *k* neighbors exist.
        """
        import faiss

        index = self._load_index(metric_type)

        query = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        if metric_type == MetricType.COSINE:
            faiss.normalize_L2(query)

        if candidate_concept_ids is not None:
            import h5py

            with h5py.File(self._h5_path(), "r") as f:
                all_ids = f[_CONCEPT_IDS_DATASET][:]
            candidate_set = set(candidate_concept_ids)
            subset_positions = np.where(
                np.isin(all_ids, list(candidate_set))
            )[0].astype(np.int64)

            sel = faiss.IDSelectorBatch(subset_positions)
            params = faiss.SearchParameters()
            params.sel = sel
            distances, ids = index.search(query, k, params=params)
        else:
            distances, ids = index.search(query, k)

        return distances, ids

    # ------------------------------------------------------------------
    # Internal index helpers
    # ------------------------------------------------------------------

    def _h5_path(self) -> Path:
        return self._cache_dir / _HDF5_FILENAME

    def _index_path(self, metric_type: MetricType) -> Path:
        index_type = self._index_config.index_type
        return self._cache_dir / f"{index_type.value}_{metric_type.value}_index.faiss"

    def _build_index(
        self,
        embeddings: np.ndarray,
        concept_ids: np.ndarray,
        metric_type: MetricType,
    ) -> None:
        import faiss

        index = self._create_empty_index(metric_type)

        if metric_type == MetricType.COSINE:
            vecs = embeddings.copy()
            faiss.normalize_L2(vecs)
        else:
            vecs = embeddings

        index.add_with_ids(vecs, concept_ids)

        index_path = self._index_path(metric_type)
        faiss.write_index(index, str(index_path))
        logger.info(
            f"Built FAISS index for metric={metric_type.value}: '{index_path}'."
        )

    def _create_empty_index(self, metric_type: MetricType):
        import faiss

        dim = self._dimensions

        if isinstance(self._index_config, FlatIndexConfig):
            if metric_type == MetricType.L2:
                return faiss.IndexFlatL2(dim)
            return faiss.IndexFlatIP(dim)  # cosine via inner product after normalisation

        if isinstance(self._index_config, HNSWIndexConfig):
            cfg = self._index_config
            faiss_metric = (
                faiss.METRIC_INNER_PRODUCT
                if metric_type == MetricType.COSINE
                else faiss.METRIC_L2
            )
            idx = faiss.IndexHNSWFlat(dim, cfg.num_neighbors, faiss_metric)
            idx.hnsw.efConstruction = cfg.ef_construction
            return idx

        raise ValueError(
            f"No FAISS index factory for index config {type(self._index_config).__name__}."
        )

    def _load_index(self, metric_type: MetricType):
        import faiss

        path = self._index_path(metric_type)
        if not path.exists():
            raise FileNotFoundError(
                f"FAISS index for metric='{metric_type.value}' not found at '{path}'. "
                "Run FAISSCache.export() first."
            )
        return faiss.read_index(str(path))
