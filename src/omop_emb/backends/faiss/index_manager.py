"""Index manager for FAISS backend. Responsible for creating, loading, and managing FAISS indices.
This module is closely tied to the storage manager, as it needs to know how to read the raw embeddings and concept_ids from the HDF5 file in order to build the indices."""

import faiss
import numpy as np
import abc
from pathlib import Path
from typing import Optional, Generator, Tuple

from omop_emb.backends.config import IndexType, MetricType
import logging
logger = logging.getLogger(__name__)

class BaseIndexManager(abc.ABC):
    def __init__(
        self, 
        dimension: int, 
        metric: MetricType,
        index_dir: str | Path
    ):
        self.dimension = dimension
        self.metric = metric
        self.index_dir = Path(index_dir)

        logger.info(f"Initializing index manager with dimension={dimension}, metric={metric}, index_dir={index_dir}")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.requires_norm = (metric == MetricType.COSINE)
        self._index = None
    
    @property
    def index(self) -> faiss.IndexIDMap:
        if self._index is None:
            logger.info("FAISS index not loaded, creating new one.")
            self._index = faiss.IndexIDMap(self._create_index())
        return self._index
    
    @property
    def filepath(self) -> Path:
        return self.index_dir / f"{self.supported_index_type.value}_{self.metric.value}_index.faiss"
    
    def has_index_on_disk(self) -> bool:
        return self.filepath.exists()

    @abc.abstractmethod
    def _create_index(self) -> faiss.Index:
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

    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Forces correct datatype and handles Cosine normalization automatically."""
        self.validate_embedding_vector(vectors)
        # FAISS only accepts float32
        vecs = vectors.astype('float32')
        if self.requires_norm:
            faiss.normalize_L2(vecs)
            
        return vecs

    def add(self, concept_ids: np.ndarray, vectors: np.ndarray):
        """Appends vectors to the index."""
        self.validate_concept_ids(concept_ids)

        vecs = self._prepare_vectors(vectors)
        self.index.add_with_ids(vecs, concept_ids)  # type: ignore

    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 5,
        subset_concept_ids: Optional[np.ndarray] = None
    ):
        """Finds the k-nearest neighbors.
        
        Parameters
        ----------
        query_vector : np.ndarray
            The vector to search with. Size q x d, where q is the number of query vectors and d is the dimension
            of the embeddings.
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
        query = self._prepare_vectors(query_vector)
        params = self._create_search_parameters(subset_concept_ids)
        distances, concept_ids = self.index.search(query, k=k, params=params)  # type: ignore
        return distances, concept_ids

    def save(self):
        """Saves the index to disk."""
        faiss.write_index(self.index, str(self.filepath))

    def load(self):
        """Loads an index from disk."""
        self._index = faiss.read_index(str(self.filepath))

    def populate_from_storage(
        self, 
        data_stream: Generator[Tuple[np.ndarray, np.ndarray], None, None]
    ):
        """Loads embeddings and concept_ids from storage and populates the index."""
        if self.has_index_on_disk():
            # TODO: Support partial loading and streaming/extending if the 
            # local embedding storage was populated from a different indexmanager
            # NOTE: Could also delete the existing file and re-populate
            logger.warning(
                "Current implementation does not support populating existing indices from "
                "storage for existing partially populated indices.\nIf you want to "
                f"re-populate from storage, please delete the existing index file at `{self.filepath}` first.\n\n"
                "YOU MAY NOW HAVE INCOMPLETE OR INCONSISTENT DATA IN YOUR INDEX IF YOU PROCEED WITHOUT "
                "DELETING THE FILE FIRST, AS WE WILL NOT CHECK FOR CONSISTENCY BETWEEN THE INDEX AND THE STORAGE."
            )
        else:
            logger.info(f"Populating {self.supported_index_type.value} from storage with data stream. This may take a while for large datasets...")
            for concept_ids, embeddings in data_stream:
                self.add(concept_ids, embeddings)
            self.save()

    def validate_embedding_vector(self, vector: np.ndarray):
        assert isinstance(vector, np.ndarray), f"Expected query_vector to be a numpy array, got {type(vector)}"
        assert vector.ndim == 2, f"Expected query_vector to be 2D, got {vector.ndim}D"
        assert vector.shape[1] == self.dimension, f"Expected query_vector to have dimension {self.dimension}, got {vector.shape[1]}"

    def validate_concept_ids(self, concept_ids: np.ndarray):
        assert isinstance(concept_ids, np.ndarray), f"Expected concept_ids to be a numpy array, got {type(concept_ids)}"
        assert concept_ids.ndim == 1, f"Expected concept_ids to be 1D, got {concept_ids.ndim}D"
        

class FlatIndexManager(BaseIndexManager):
    def __init__(
        self, 
        dimension: int, 
        metric: MetricType,
        index_dir: str | Path
    ):
        super().__init__(dimension, metric, index_dir)
        
    def _create_index(self) -> faiss.Index:
        if self.metric == MetricType.L2:
            return faiss.IndexFlatL2(self.dimension)
        elif self.metric == MetricType.COSINE:
            return faiss.IndexFlatIP(self.dimension)  # Inner Product for Cosine similarity after normalization
        else:
            raise ValueError(f"Unsupported metric {self.metric} for Flat index.")
        
    def supported_index_type(self) -> IndexType:
        return IndexType.FLAT
    
    def _create_search_parameters(self, concept_id_subset: np.ndarray | None = None) -> faiss.SearchParameters:
        if concept_id_subset is not None:
            self.validate_concept_ids(concept_id_subset)
            id_selector = faiss.IDSelectorBatch(concept_id_subset)  # type: ignore
            params = faiss.SearchParameters(sel=id_selector)  # type: ignore
            return params
        else:
            return faiss.SearchParameters()  # Default parameters with no filtering
    
# NOTE: The following classes are placeholders for future implementation
# These require training and more complex logic fod adding vectors
# Also require training etc.

class HNSWIndexManager(BaseIndexManager):
    def __init__(
        self, 
        dimension: int, 
        metric: MetricType,
        index_dir: str | Path,
        num_neighbors: int = 32
    ):
        super().__init__(dimension, metric, index_dir)
        self.num_neighbors = num_neighbors
        
    #def _create_index(self) -> faiss.Index:
    #    if self.metric == MetricType.L2:
    #        return faiss.IndexHNSWFlat(self.dimension, self.num_neighbors)
    #    elif self.metric == MetricType.COSINE:
    #        # Inner Product for Cosine similarity after normalization
    #        return faiss.IndexHNSWFlat(self.dimension, self.num_neighbors, faiss.METRIC_INNER_PRODUCT)  
    #    else:
    #        raise ValueError(f"Unsupported metric {self.metric} for HNSW index.")
        
    def supported_index_type(self) -> IndexType:
        return IndexType.HNSW
    
class IVFIndexManager(BaseIndexManager):
    def __init__(
        self, 
        dimension: int, 
        metric: MetricType,
        index_dir: str | Path,
        num_clusters: int = 100
    ):
        super().__init__(dimension, metric, index_dir)
        self.num_clusters = num_clusters
        
    #def _create_index(self) -> faiss.Index:
    #    quantizer = faiss.IndexFlatL2(self.dimension)  # Use L2 for quantization
    #    if self.metric == MetricType.L2:
    #        return faiss.IndexIVFFlat(quantizer, self.dimension, self.num_clusters, faiss.METRIC_L2)
    #    elif self.metric == MetricType.COSINE:
    #        return faiss.IndexIVFFlat(quantizer, self.dimension, self.num_clusters, faiss.METRIC_INNER_PRODUCT)
    #    else:
    #        raise ValueError(f"Unsupported metric {self.metric} for IVF index.")
        
    def supported_index_type(self) -> IndexType:
        return IndexType.IVF