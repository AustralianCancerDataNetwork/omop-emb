"""FAISS index manager.

Each ``FaissBaseIndexManager`` subclass manages one index type (Flat, HNSW, …)
and can hold multiple metric-specific FAISS indices in memory, one per
``MetricType``.  On-disk persistence uses ``.faiss`` files under a per-type
subdirectory of the model's storage directory.
"""

from __future__ import annotations

import abc
import shutil
from pathlib import Path
from typing import Dict, Generator, Generic, Optional, Tuple, TypeVar

import faiss
import numpy as np

from omop_emb.config import IndexType, MetricType
from omop_emb.backends.index_config import IndexConfig, FlatIndexConfig, HNSWIndexConfig
from omop_emb.backends.index_utils import BaseIndexManager

import logging
logger = logging.getLogger(__name__)

C = TypeVar("C", bound=IndexConfig)


def _warn_index_staleness(filepath: Path, index_count: int, expected_count: int) -> None:
    if index_count == expected_count:
        return
    logger.warning(
        f"Index file at '{filepath}' has {index_count} vectors but the HDF5 storage has "
        f"{expected_count}. The index is stale and search results may be incomplete or wrong. "
        "Call rebuild_index_for_metric() to bring it back in sync."
    )


class FaissBaseIndexManager(BaseIndexManager[C], Generic[C]):
    """FAISS-specific index manager base class.

    Extends ``BaseIndexManager`` with the FAISS-only operations: vector
    insertion (``add``), in-process searching (``search``), and on-disk
    persistence (``save`` / ``load``).

    One manager instance handles one ``IndexType`` (e.g. FLAT or HNSW) and
    caches one ``faiss.Index`` per ``MetricType`` it has been asked to serve.
    """

    def __init__(
        self,
        dimension: int,
        base_index_dir: str | Path,
        index_config: C,
    ):
        self.dimension = dimension
        self.base_dir = Path(base_index_dir)

        if index_config.index_type != self.supported_index_type:
            raise ValueError(
                f"index_config has index_type {index_config.index_type!r} but this manager "
                f"supports {self.supported_index_type!r}."
            )
        self._index_config = index_config

        self.index_dir = self.base_dir / f"index_{self.supported_index_type.value}"
        self.index_dir.mkdir(parents=False, exist_ok=True)
        logger.info(
            f"Initializing FAISS index manager dimension={dimension}, "
            f"index_type={self.supported_index_type.value}"
        )
        self._metric_to_index_cache: Dict[MetricType, faiss.Index] = {}

    # ------------------------------------------------------------------
    # BaseIndexManager identity
    # ------------------------------------------------------------------

    @property
    def index_config(self) -> C:
        return self._index_config

    # ------------------------------------------------------------------
    # BaseIndexManager lifecycle (FAISS implementations)
    # ------------------------------------------------------------------

    def has_index(self, metric_type: MetricType) -> bool:
        """Return True if an index file exists on disk for *metric_type*."""
        return self.index_filepath(metric_type).exists()

    def create_index(self, metric_type: MetricType, **_kwargs) -> None:
        """Ensure a FAISS index exists in memory for *metric_type*.

        Loads from disk if a file is present; otherwise builds an empty in-memory
        index.  Does not populate with vectors — call ``load_or_create`` for the
        full load-or-build-from-stream flow.
        """
        if metric_type in self._metric_to_index_cache:
            return

        if self.has_index(metric_type):
            self.load(metric_type)
            return

        logger.info(f"Creating new empty FAISS index for metric '{metric_type.value}'.")
        raw_index = self._create_raw_index(metric_type=metric_type)
        if not isinstance(raw_index, faiss.Index):
            raise TypeError(
                f"_create_raw_index() must return a faiss.Index, got {type(raw_index)}"
            )
        if not isinstance(raw_index, faiss.IndexIVF):
            mapped_index = faiss.IndexIDMap(raw_index)
        else:
            mapped_index = raw_index
        self._metric_to_index_cache[metric_type] = mapped_index

        # Save to file that it exists
        self.save(metric_type)

    def drop_index(self, metric_type: MetricType) -> None:
        """Evict from memory and delete the on-disk file for *metric_type*."""
        self._metric_to_index_cache.pop(metric_type, None)
        filepath = self.index_filepath(metric_type)
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Deleted FAISS index file {filepath}.")

    def load_or_create(
        self,
        metric_type: MetricType,
        *,
        data_stream: Optional[Generator[Tuple[np.ndarray, np.ndarray], None, None]] = None,
        expected_count: Optional[int] = None,
        **_kwargs,
    ) -> None:
        """Load index from disk if present; otherwise populate from *data_stream* and save.

        Parameters
        ----------
        data_stream:
            Generator of ``(concept_ids, embeddings)`` batches used when no
            on-disk index exists yet.  Required when the index file is missing.
        expected_count:
            Expected number of vectors.  When loading from disk, a staleness
            warning is emitted if the file's vector count differs.
        """
        if self.has_index(metric_type):
            filepath = self.index_filepath(metric_type)
            logger.info(f"Index file found at {filepath}, loading from disk.")
            self.load(metric_type)
            if expected_count is not None:
                index = self._metric_to_index_cache[metric_type]
                _warn_index_staleness(filepath, index.ntotal, expected_count)
        else:
            if data_stream is None:
                raise ValueError(
                    f"No index file exists for metric '{metric_type.value}' and no "
                    "data_stream was provided to populate from."
                )
            logger.info(
                f"No index file found for {self.supported_index_type.value}/{metric_type.value}, "
                "populating from HDF5 storage. This may take a while for large datasets..."
            )
            self.create_index(metric_type)
            for concept_ids, embeddings in data_stream:
                self.add(concept_ids, embeddings, metric_type=metric_type)
            self.save(metric_type)

    def rebuild_index(
        self,
        metric_type: MetricType,
        *,
        data_stream: Generator[Tuple[np.ndarray, np.ndarray], None, None],
        expected_count: Optional[int] = None,
    ) -> None:
        """Drop the existing index for *metric_type* and rebuild from *data_stream*."""
        self.drop_index(metric_type)
        self.load_or_create(
            metric_type,
            data_stream=data_stream,
            expected_count=expected_count,
        )

    # ------------------------------------------------------------------
    # FAISS-only operations (not on BaseIndexManager)
    # ------------------------------------------------------------------

    def add(self, concept_ids: np.ndarray, vectors: np.ndarray, metric_type: MetricType) -> None:
        """Append vectors to the in-memory FAISS index."""
        self.validate_concept_ids(concept_ids)
        vecs = self._prepare_vectors(vectors, metric_type)

        index = self._get_cached_index(metric_type)
        if not hasattr(index, "add_with_ids"):
            raise NotImplementedError(
                f"Index type {type(index).__name__} does not support add_with_ids."
            )
        index.add_with_ids(vecs, concept_ids)  # type: ignore[call-arg]

    def search(
        self,
        query_vector: np.ndarray,
        metric_type: MetricType,
        k: int = 5,
        subset_concept_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for the k nearest neighbors of *query_vector*.

        Returns
        -------
        distances : np.ndarray
            Shape (Q, k) — distances to nearest neighbors.
        concept_ids : np.ndarray
            Shape (Q, k) — concept IDs of nearest neighbors.
        """
        query = self._prepare_vectors(query_vector, metric_type=metric_type)
        params = self._create_search_parameters(concept_id_subset=subset_concept_ids)

        index = self._get_cached_index(metric_type)
        if index.ntotal == 0:
            raise RuntimeError(
                f"FAISS index for metric '{metric_type.value}' contains no vectors. "
                "Populate it first via load_or_create() before searching."
            )
        distances, concept_ids = index.search(query, k=k, params=params)  # type: ignore

        if metric_type == MetricType.L2:
            distances = np.sqrt(distances)
        elif metric_type == MetricType.COSINE:
            distances = 1.0 - distances

        return distances, concept_ids

    def save(self, metric_type: MetricType) -> None:
        """Write the in-memory index for *metric_type* to disk."""
        if metric_type not in self._metric_to_index_cache:
            raise RuntimeError(
                f"No in-memory index for metric '{metric_type.value}'. "
                "Call load_or_create() or add() before saving."
            )
        faiss.write_index(
            self._metric_to_index_cache[metric_type],
            str(self.index_filepath(metric_type)),
        )

    def load(self, metric_type: MetricType) -> None:
        """Load an index from disk into the in-memory cache."""
        if metric_type in self._metric_to_index_cache:
            logger.debug(
                f"Index for metric {metric_type.value} already in memory, skipping load."
            )
            return
        self._metric_to_index_cache[metric_type] = faiss.read_index(
            str(self.index_filepath(metric_type))
        )

    def cleanup(self) -> None:
        """Delete all indices (memory + disk) managed by this instance."""
        metrics_to_clean: set[MetricType] = set(self._metric_to_index_cache.keys())
        for metric_type in MetricType:
            if self.has_index(metric_type):
                metrics_to_clean.add(metric_type)
        for metric_type in metrics_to_clean:
            self.drop_index(metric_type)
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)

    def cleanup_metric(self, metric_type: MetricType) -> None:
        """Alias for ``drop_index`` — kept for backward compatibility."""
        self.drop_index(metric_type)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def index_filepath(self, metric_type: MetricType) -> Path:
        return self.index_dir / f"{self.supported_index_type.value}_{metric_type.value}_index.faiss"

    # Backward-compat alias used by storage_manager and tests
    def has_index_on_disk_for_metric(self, metric_type: MetricType) -> bool:
        return self.has_index(metric_type)

    def create_index_filepath_for_metric(self, metric_type: MetricType) -> Path:
        return self.index_filepath(metric_type)

    def _get_cached_index(self, metric_type: MetricType) -> faiss.Index:
        """Return the cached index, creating it if necessary."""
        if metric_type not in self._metric_to_index_cache:
            self.create_index(metric_type)
        index = self._metric_to_index_cache.get(metric_type)
        if index is None:
            raise RuntimeError(
                f"Index for metric '{metric_type.value}' not found after create_index()."
            )
        return index

    @staticmethod
    def metric_requires_norm(metric_type: MetricType) -> bool:
        return metric_type == MetricType.COSINE

    def _prepare_vectors(self, vectors: np.ndarray, metric_type: MetricType) -> np.ndarray:
        self.validate_embedding_vector(vectors)
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.metric_requires_norm(metric_type):
            # faiss.normalize_L2 is in-place; copy first so we never mutate the caller's buffer.
            # np.ascontiguousarray returns the same object when the input is already C-contiguous
            # float32, so without this copy the caller's array would be silently modified.
            vecs = vecs.copy()
            faiss.normalize_L2(vecs)
        return vecs

    def validate_embedding_vector(self, vector: np.ndarray) -> None:
        if not isinstance(vector, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(vector)}")
        if vector.ndim != 2:
            raise ValueError(f"Expected 2D array, got {vector.ndim}D")
        if vector.shape[1] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, got {vector.shape[1]}"
            )

    def validate_concept_ids(self, concept_ids: np.ndarray) -> None:
        if not isinstance(concept_ids, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(concept_ids)}")
        if concept_ids.ndim != 1:
            raise ValueError(f"Expected 1D array, got {concept_ids.ndim}D")

    # ------------------------------------------------------------------
    # Abstract hooks for concrete index types
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _create_raw_index(self, metric_type: MetricType) -> faiss.Index:
        """Instantiate and return the raw FAISS index for *metric_type*."""

    @abc.abstractmethod
    def _create_search_parameters(
        self, concept_id_subset: Optional[np.ndarray] = None
    ) -> faiss.SearchParameters:
        """Return search parameters appropriate for this index type."""


# ---------------------------------------------------------------------------
# Concrete FAISS index managers
# ---------------------------------------------------------------------------

class FaissFlatIndexManager(FaissBaseIndexManager[FlatIndexConfig]):

    @property
    def supported_index_type(self) -> IndexType:
        return IndexType.FLAT

    def _create_raw_index(self, metric_type: MetricType) -> faiss.Index:
        if metric_type == MetricType.L2:
            return faiss.IndexFlatL2(self.dimension)
        elif metric_type == MetricType.COSINE:
            return faiss.IndexFlatIP(self.dimension)
        raise ValueError(f"Unsupported metric {metric_type!r} for Flat index.")

    def _create_search_parameters(
        self, concept_id_subset: Optional[np.ndarray] = None
    ) -> faiss.SearchParameters:
        if concept_id_subset is not None:
            self.validate_concept_ids(concept_id_subset)
            sel = faiss.IDSelectorBatch(concept_id_subset)  # type: ignore
            return faiss.SearchParameters(sel=sel)  # type: ignore
        return faiss.SearchParameters()


class FaissHNSWIndexManager(FaissBaseIndexManager[HNSWIndexConfig]):

    @property
    def supported_index_type(self) -> IndexType:
        return IndexType.HNSW

    def _create_raw_index(self, metric_type: MetricType) -> faiss.Index:
        if metric_type == MetricType.L2:
            index = faiss.IndexHNSWFlat(self.dimension, self.index_config.num_neighbors)
        elif metric_type == MetricType.COSINE:
            index = faiss.IndexHNSWFlat(
                self.dimension,
                self.index_config.num_neighbors,
                faiss.METRIC_INNER_PRODUCT,
            )
        else:
            raise ValueError(f"Unsupported metric {metric_type!r} for HNSW index.")
        index.hnsw.efConstruction = self.index_config.ef_construction
        return index

    def _create_search_parameters(
        self, concept_id_subset: Optional[np.ndarray] = None
    ) -> faiss.SearchParameters:
        params = faiss.SearchParametersHNSW()
        params.efSearch = self.index_config.ef_search
        if concept_id_subset is not None:
            self.validate_concept_ids(concept_id_subset)
            params.sel = faiss.IDSelectorBatch(concept_id_subset)  # type: ignore
        return params
