"""Index manager for FAISS backend. Responsible for creating, loading, and managing FAISS indices.
This module is closely tied to the storage manager, as it needs to know how to read the raw embeddings and concept_ids from the HDF5 file in order to build the indices."""

import faiss
import numpy as np
import abc
import shutil
from pathlib import Path
from typing import Generic, Optional, Generator, Tuple, TypeVar, Dict

from omop_emb.config import IndexType, MetricType
from omop_emb.backends.index_config import (
    IndexConfig,
    FlatIndexConfig,
    HNSWIndexConfig,
)
import logging
logger = logging.getLogger(__name__)

C = TypeVar('C', bound=IndexConfig)

def _warn_index_staleness(filepath: Path, index_count: int, expected_count: int) -> None:
    if index_count == expected_count:
        return
    logger.warning(
        f"Index file at '{filepath}' has {index_count} vectors but the HDF5 storage has "
        f"{expected_count}. The index is stale and search results may be incomplete or wrong. "
        "Call rebuild_index_for_metric() to bring it back in sync."
    )

class BaseIndexManager(abc.ABC, Generic[C]):
    """Abstract Base class for managing FAISS indices. 
    Each IndexManager is responsible for a specific index type (e.g. Flat, HNSW) and can hold various
    metrics per IndexType (e.g. L2, Cosine). 
    The manager handles creation, loading, saving, and searching of the index, 
    as well as ensuring that the correct parameters are used for each metric and index type.
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
            raise ValueError(f"Provided index_config has index_type {index_config.index_type} which does not match the supported index type {self.supported_index_type} for this manager.")
        self._index_config = index_config  # Required prior as suppported_index_type is depending on the index_config

        self.index_dir = self.base_dir / ("index_" + self.supported_index_type.value)
        self.index_dir.mkdir(parents=False, exist_ok=True)
        logger.info(f"Initializing index manager with dimension={dimension}, index_type={self.supported_index_type.value}")
        self._metric_to_index_cache: Dict[MetricType, faiss.Index] = {}

    @property
    def index_config(self) -> C:
        return self._index_config
    
    def create_index(self, metric_type: MetricType) -> faiss.Index:
        """Lazily loads the FAISS index, creating it if it doesn't exist.
        Wraps the raw index in an IDMap if it's not already an IVF index, to ensure we can use explicit concept_ids as IDs regardless of the underlying index type.
        See https://github.com/facebookresearch/faiss/wiki/Pre--and-post-processing#faiss-id-mapping for details on why this is necessary.
    """
        if metric_type in self._metric_to_index_cache:
            return self._metric_to_index_cache[metric_type]
        
        # Try loading from disk if available
        if self.has_index_on_disk_for_metric(metric_type):
            self.load(metric_type)
            loaded_index = self._metric_to_index_cache.get(metric_type)
            if loaded_index is None:
                raise RuntimeError(f"Index was loaded from disk but not found in cache for metric {metric_type}.")
            return loaded_index
        
        logger.info("FAISS index not loaded, creating new one.")
        raw_index = self._create_index_for_metric(metric_type=metric_type)
        if not isinstance(raw_index, faiss.Index):
            raise TypeError(f"_create_index() should return a faiss.Index instance, got {type(raw_index)}")
        if not isinstance(raw_index, faiss.IndexIVF):
            raw_index = faiss.IndexIDMap(raw_index)
        self._metric_to_index_cache[metric_type] = raw_index
        return raw_index
    
    @staticmethod
    def metric_requires_norm(metric_type: MetricType) -> bool:
        """Determines if the given metric type requires L2 normalization of vectors."""
        return metric_type == MetricType.COSINE
    
    def create_index_filepath_for_metric(self, metric_type: MetricType) -> Path:
        return self.index_dir / f"{self.supported_index_type.value}_{metric_type.value}_index.faiss"
    
    def has_index_on_disk_for_metric(self, metric_type) -> bool:
        return self.create_index_filepath_for_metric(metric_type).exists()

    @abc.abstractmethod
    def _create_index_for_metric(self, metric_type: MetricType) -> faiss.Index:
        """Creates a new FAISS index based on the specified metric and dimension."""
        pass
    
    @property
    @abc.abstractmethod
    def supported_index_type(self) -> IndexType:
        """Returns the IndexType supported by this manager."""
        pass

    @abc.abstractmethod
    def _create_search_parameters(self, concept_id_subset: Optional[np.ndarray] = None) -> dict:
        """Creates a dictionary of search parameters specific to the index type and metric."""
        pass

    def _prepare_vectors(self, vectors: np.ndarray, metric_type: MetricType) -> np.ndarray:
        """Forces correct datatype and handles Cosine normalization automatically."""
        self.validate_embedding_vector(vectors)
        # FAISS only accepts float32
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.metric_requires_norm(metric_type):
            faiss.normalize_L2(vecs)
            
        return vecs

    def add(self, concept_ids: np.ndarray, vectors: np.ndarray, metric_type: MetricType):
        """Appends vectors to the index."""
        self.validate_concept_ids(concept_ids)
        vecs = self._prepare_vectors(vectors, metric_type)

        index = self.create_index(metric_type)
        if not hasattr(index, "add_with_ids"):
            raise NotImplementedError(f"Index type {type(index).__name__} does not support adding with explicit IDs. ")
        index.add_with_ids(vecs, concept_ids)  # type: ignore[call-arg]

    def search(
        self, 
        query_vector: np.ndarray, 
        metric_type: MetricType,
        k: int = 5,
        subset_concept_ids: Optional[np.ndarray] = None
    ):
        """Finds the k-nearest neighbors.
        
        Parameters
        ----------
        query_vector : np.ndarray
            The vector to search with. Size q x d, where q is the number of query vectors and d is the dimension
            of the embeddings.
        metric_type : MetricType
            The metric type to use for the search. This determines which index is used and how the distances are computed.
        k : int, optional
            The number of nearest neighbors to return, by default 5. 
        subset_concept_ids : Optional[np.ndarray], optional
            If provided, only search within this subset of concept_ids. This is used to implement the concept filtering logic for nearest neighbor search. The array should be 1D and contain the concept_ids that are allowed to be returned in the search results.
        
        Returns
        -------
        distances : np.ndarray
            The distances of the nearest neighbors. Size q x k.
        concept_ids : np.ndarray
            The concept_ids of the nearest neighbors. Size q x k.
        """
        query = self._prepare_vectors(query_vector, metric_type=metric_type)
        params = self._create_search_parameters(concept_id_subset=subset_concept_ids)

        index = self.create_index(metric_type)
        if index.ntotal == 0:
            raise RuntimeError(
                f"FAISS index for metric '{metric_type.value}' contains no vectors. "
                "Populate it first via load_or_populate() before searching."
            )
        distances, concept_ids = index.search(query, k=k, params=params)  # type: ignore

        if metric_type == MetricType.L2:
            # https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#metric_l2
            distances = np.sqrt(distances)
        elif metric_type == MetricType.COSINE:
            # https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-index-vectors-for-cosine-similarity
            # Cosine returns similarity, not distance
            distances = 1.0 - distances

        return distances, concept_ids

    def save(self, metric_type: MetricType):
        """Saves the in-memory index to disk. Raises if the index hasn't been loaded or populated yet."""
        if metric_type not in self._metric_to_index_cache:
            raise RuntimeError(
                f"No in-memory index for metric '{metric_type.value}'. "
                "Call load_or_populate() or add() before saving."
            )
        faiss.write_index(
            self._metric_to_index_cache[metric_type],
            str(self.create_index_filepath_for_metric(metric_type)),
        )

    def load(self, metric_type: MetricType):
        """Loads an index from disk."""
        if metric_type in self._metric_to_index_cache:
            logger.info(f"Index for metric {metric_type.value} already loaded in memory, skipping load from disk.")
            return

        index_filepath = self.create_index_filepath_for_metric(metric_type)
        self._metric_to_index_cache[metric_type] = faiss.read_index(str(index_filepath))

    def cleanup(self):
        """Deletes all indices and associated files managed by this IndexManager. Use with caution."""
        # Union of what's in memory and what exists on disk — covers both
        # indices added this session and files left from a previous process.
        metrics_to_clean: set[MetricType] = set(self._metric_to_index_cache.keys())
        for metric_type in MetricType:
            if self.has_index_on_disk_for_metric(metric_type):
                metrics_to_clean.add(metric_type)

        for metric_type in metrics_to_clean:
            self.cleanup_metric(metric_type)

        if self.index_dir.exists() and self.index_dir.is_dir():
            shutil.rmtree(self.index_dir)
            

    def cleanup_metric(self, metric_type: MetricType):
        """Deletes the index and associated file for a specific metric."""
        if self.has_index_on_disk_for_metric(metric_type):
            index_filepath = self.create_index_filepath_for_metric(metric_type)
            index_filepath.unlink()
            logger.info(f"Deleted index file at {index_filepath} for metric {metric_type.value}")
        else:
            logger.debug(f"No index file found for metric {metric_type.value} at expected location {self.create_index_filepath_for_metric(metric_type)}, skipping cleanup.")
        
        # Also remove from in-memory cache if present
        if metric_type in self._metric_to_index_cache:
            del self._metric_to_index_cache[metric_type]
            logger.debug(f"Removed index for metric {metric_type.value} from in-memory cache.")

    def load_or_populate(
        self,
        data_stream: Generator[Tuple[np.ndarray, np.ndarray], None, None],
        metric_type: MetricType,
        expected_count: Optional[int] = None,
    ):
        """Loads the index from disk if it exists, otherwise builds it from the data stream and saves.

        Parameters
        ----------
        expected_count : int, optional
            Number of vectors in the HDF5 store. When provided and an on-disk index
            is loaded, the index ``ntotal`` is compared against this value and a
            staleness warning is emitted on mismatch.
        """
        if self.has_index_on_disk_for_metric(metric_type):
            filepath = self.create_index_filepath_for_metric(metric_type)
            logger.info(f"Index file found at {filepath}, loading from disk.")
            self.load(metric_type)
            if expected_count is not None:
                index = self._metric_to_index_cache[metric_type]
                _warn_index_staleness(filepath, index.ntotal, expected_count)
        else:
            logger.info(
                f"No index file found for {self.supported_index_type.value}/{metric_type.value}, "
                "populating from HDF5 storage. This may take a while for large datasets..."
            )
            self.create_index(metric_type)
            for concept_ids, embeddings in data_stream:
                self.add(concept_ids, embeddings, metric_type=metric_type)
            self.save(metric_type)

    def validate_embedding_vector(self, vector: np.ndarray):
        if not isinstance(vector, np.ndarray):
            raise TypeError(f"Expected query_vector to be a numpy array, got {type(vector)}")
        if vector.ndim != 2:
            raise ValueError(f"Expected query_vector to be 2D, got {vector.ndim}D")
        if vector.shape[1] != self.dimension:
            raise ValueError(f"Expected query_vector to have dimension {self.dimension}, got {vector.shape[1]}")

    def validate_concept_ids(self, concept_ids: np.ndarray):
        if not isinstance(concept_ids, np.ndarray):
            raise TypeError(f"Expected concept_ids to be a numpy array, got {type(concept_ids)}")
        if concept_ids.ndim != 1:
            raise ValueError(f"Expected concept_ids to be 1D, got {concept_ids.ndim}D")

    def rebuild_index_for_metric(
        self,
        metric_type: MetricType,
        data_stream: Generator[Tuple[np.ndarray, np.ndarray], None, None],
        expected_count: Optional[int] = None,
    ):
        """Deletes the existing index for the given metric and rebuilds it from the provided data stream."""
        self.cleanup_metric(metric_type)
        self.load_or_populate(data_stream, metric_type, expected_count=expected_count)

class FlatIndexManager(BaseIndexManager[FlatIndexConfig]):
    @property
    def supported_index_type(self) -> IndexType:
        return IndexType.FLAT
    
    def _create_index_for_metric(self, metric_type: MetricType) -> faiss.Index:

        if metric_type == MetricType.L2:
            return faiss.IndexFlatL2(self.dimension)
        elif metric_type == MetricType.COSINE:
            return faiss.IndexFlatIP(self.dimension)  # Inner Product for Cosine similarity after normalization
        else:
            raise ValueError(f"Unsupported metric {metric_type} for Flat index.")
    
    def _create_search_parameters(self, concept_id_subset: np.ndarray | None = None) -> faiss.SearchParameters:
        if concept_id_subset is not None:
            self.validate_concept_ids(concept_id_subset)
            id_selector = faiss.IDSelectorBatch(n=len(concept_id_subset), indices=concept_id_subset)
            params = faiss.SearchParameters(sel=id_selector) # type: ignore
            return params
        else:
            return faiss.SearchParameters()  # Default parameters with no filtering
    
class HNSWIndexManager(BaseIndexManager[HNSWIndexConfig]):
    @property
    def supported_index_type(self) -> IndexType:
        return IndexType.HNSW

    def _create_index_for_metric(
        self, 
        metric_type: MetricType,
    ) -> faiss.Index:

        if metric_type == MetricType.L2:
            index = faiss.IndexHNSWFlat(self.dimension, self.index_config.num_neighbors)
        elif metric_type == MetricType.COSINE:
            # Vectors are L2-normalised in _prepare_vectors; inner-product on
            # unit vectors equals cosine similarity.
            index = faiss.IndexHNSWFlat(self.dimension, self.index_config.num_neighbors, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unsupported metric {metric_type} for HNSW index.")
        index.hnsw.efConstruction = self.index_config.ef_construction
        return index

    def _create_search_parameters(self, concept_id_subset: np.ndarray | None = None) -> faiss.SearchParameters:
        params = faiss.SearchParametersHNSW()
        params.efSearch = self.index_config.ef_search
        if concept_id_subset is not None:
            self.validate_concept_ids(concept_id_subset)
            params.sel = faiss.IDSelectorBatch(concept_id_subset)  # type: ignore
        return params