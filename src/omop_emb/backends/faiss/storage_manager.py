"""Embedding storage manager for FAISS backend using HDF5 files.
Instead of directly converting the embeddings into an index,
we store the raw embeddings and their corresponding concept_ids in an HDF5 file on disk.
This allows us to generate indices on the fly and also easily update the embeddings without needing to rebuild the entire index."""

import h5py
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from typing import cast, Generator, Tuple, Optional, Mapping, Sequence, Dict
import logging
logger = logging.getLogger(__name__)

from omop_emb.backends.config import BackendType, IndexType, MetricType
from .index_manager import FlatIndexManager, BaseIndexManager

class EmbeddingStorageManager:
    """Storage manager for raw embeddings and concept_ids using HDF5 files for FAISS backend. 
    This additional supports the management of index files on disk, which can be calculated on demand and updated
    as this manager stores the raw embeddings and concept_ids in a way that allows appending new data without needing to load everything into RAM.

    Notes
    -----
    - Currently only supports the FLAT index type for FAISS, but the structure allows for easy extension to other index types in the future.
    - Currently only supports one IndexMAnager per metric

    """
    DATASET_NAME_EMBEDDINGS = "embeddings"
    DATASET_NAME_CONCEPT_IDS = "concept_ids"
    INDEX_DIR = "indices"
    EMBEDDINGS_FILE = "embeddings.h5"
    def __init__(
        self, 
        file_dir: str | Path, 
        dimensions: int,
        backend_type: BackendType,
        # NOTE: Add neighbours/clusters here as param if differently support index types
    ):
        # Sanity check
        if backend_type != BackendType.FAISS:
            raise ValueError(f"EmbeddingStorageManager is designed for FAISS backend. Got {backend_type}.")

        self.embeddings_file = Path(file_dir) / self.EMBEDDINGS_FILE
        self.dimensions = dimensions
        self._init_embedding_storage_if_missing()
        self._index_managers: Dict[IndexType, BaseIndexManager] = {}

    @property
    def index_dir(self) -> Path:
        return self.embeddings_file.parent / self.INDEX_DIR
    
    def get_index_manager(
        self, 
        index_type: IndexType,
        metric_type: MetricType
    ) -> BaseIndexManager:
        if index_type not in self._index_managers:
            logger.info(f"Index manager for {index_type} not found in memory. Creating new one.")
            self.create_index_manager(
                index_type=index_type,
                metric_type=metric_type
            )
        index_manager = self._index_managers[index_type]
        assert index_manager.metric == metric_type, f"Requested metric type {metric_type} does not match existing index manager's metric {index_manager.metric}. Currently only supporting one metric type per index type. Please create a new index manager for this metric type if you want to use it."
        return index_manager
    
    def create_index_manager(
        self,
        index_type: IndexType,
        metric_type: MetricType,
        batch_size: int = 100_000
    ):
        if index_type == IndexType.FLAT:
            index_manager_cls = FlatIndexManager
        else:
            raise ValueError(f"Unsupported index type {index_type} for FAISS backend. Only {IndexType.FLAT} is supported currently.")
        
        self._index_managers[index_type] = index_manager_cls(
            dimension=self.dimensions,
            metric=metric_type,
            index_dir=self.index_dir
        )

        # Now we also attempt to populate the index from storage
        self._index_managers[index_type].populate_from_storage(
            self.stream_concept_ids_and_embeddings(batch_size=batch_size)
        )

    def _init_embedding_storage_if_missing(self):
        """Creates the .h5 file with resizable datasets if it doesn't exist."""
        if not self.embeddings_file.exists():
            logger.info(f"Creating new HDF5 storage at {self.embeddings_file}")
            with h5py.File(self.embeddings_file, 'w') as f:
                # maxshape=(None, dim) is the magic that allows appending!
                f.create_dataset(
                    self.DATASET_NAME_EMBEDDINGS, 
                    shape=(0, self.dimensions), 
                    maxshape=(None, self.dimensions), 
                    dtype='float32',
                    chunks=True # Required for resizing
                )
                f.create_dataset(
                    self.DATASET_NAME_CONCEPT_IDS, 
                    shape=(0,), 
                    maxshape=(None,), 
                    dtype='int64',
                    chunks=True
                )

    @contextmanager
    def open_embeddings(self, mode: str) -> Generator[h5py.Dataset, None, None]:
        """
        Context manager that yields the embeddings dataset with static typing.
        Guarantees the file is closed after use.
        """
        with h5py.File(self.embeddings_file, mode) as f:
            embeddings: h5py.Dataset = cast(h5py.Dataset, f[self.DATASET_NAME_EMBEDDINGS])
            yield embeddings

    @contextmanager
    def open_concept_ids(self, mode: str) -> Generator[h5py.Dataset, None, None]:
        """Context manager for the concept_id dataset."""
        with h5py.File(self.embeddings_file, mode) as f:
            concept_ids: h5py.Dataset = cast(h5py.Dataset, f[self.DATASET_NAME_CONCEPT_IDS])
            yield concept_ids

    @contextmanager
    def open_all(self, mode: str) -> Generator[tuple[h5py.Dataset, h5py.Dataset], None, None]:
        """Context manager that yields both datasets together."""
        with h5py.File(self.embeddings_file, mode) as f:
            embeddings: h5py.Dataset = cast(h5py.Dataset, f[self.DATASET_NAME_EMBEDDINGS])
            concept_ids: h5py.Dataset = cast(h5py.Dataset, f[self.DATASET_NAME_CONCEPT_IDS])
            yield embeddings, concept_ids


    def append(
        self, 
        concept_ids: np.ndarray, 
        embeddings: np.ndarray,
        index_type: Optional[IndexType] = None,
        metric_type: Optional[MetricType] = None
    ):
        """Appends new vectors directly to the disk without loading old ones into RAM."""
        
        assert embeddings.shape[1] == self.dimensions, f"Embeddings dimension {embeddings.shape[1]} does not match expected {self.dimensions}"
        assert embeddings.shape[0] == concept_ids.shape[0], "Number of embeddings must match number of concept_ids"
        assert embeddings.ndim == 2, "Embeddings must be a 2D array"

        unique = np.unique(concept_ids)
        if len(unique) != len(concept_ids):
            raise ValueError("Duplicate concept_ids found in the input. Please ensure all concept_ids are unique to prevent data corruption.")

        existing_concept_ids = self.get_concept_ids()
        if np.intersect1d(existing_concept_ids, concept_ids).size > 0:
            raise ValueError("Some concept_ids already exist in storage. Appending duplicate concept_ids would corrupt the data. Please ensure all concept_ids are unique and do not already exist in storage.")

        with self.open_all(mode="a") as (emb_ds, cid_ds):
            # 1. Calculate new sizes
            old_size = emb_ds.shape[0]
            new_size = old_size + embeddings.shape[0]
            
            # 2. Resize the datasets on disk
            emb_ds.resize((new_size, self.dimensions))
            cid_ds.resize((new_size,))
            
            # 3. Write the new data into the newly created empty space at the end
            emb_ds[old_size:new_size] = embeddings.astype('float32')
            cid_ds[old_size:new_size] = concept_ids.astype('int64')

        if index_type is not None and metric_type is not None:
            index_manager = self.get_index_manager(index_type=index_type, metric_type=metric_type)
            index_manager.add(concept_ids, embeddings)
            index_manager.save()

    def search(
        self, 
        query_vector: np.ndarray, 
        metric_type: MetricType,
        index_type: IndexType,
        k: int = 5,
        subset_concept_ids: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interface to the index manager's search function, which handles the actual FAISS index searching logic.
        
        Parameters
        ----------
        query_vector : np.ndarray
            The vector to search with. Size q x d, where q is the number of query vectors and d is the dimension
            of the embeddings.
        metric_type : MetricType
            The distance metric to use for the search, which determines how the index manager calculates distances and
            thus affects the search results.
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
        index_manager = self.get_index_manager(index_type=index_type, metric_type=metric_type)
        assert index_manager is not None, "Index manager should have been created by now."
        return index_manager.search(
            query_vector=query_vector,
            k=k,
            subset_concept_ids=subset_concept_ids
        )
    
    def get_embeddings_by_concept_ids(self, concept_ids: np.ndarray) -> Mapping[int, Sequence[float]]:
        self.validate_concept_ids(concept_ids)

        with self.open_all(mode="r") as (emb_ds, cid_ds):
            # Load into RAM - cheap for low numbers
            all_ids = cid_ds[:]

            # Sanity check if requested concept_ids actually exist in storage
            difference = np.setdiff1d(concept_ids, all_ids)
            if len(difference) > 0:
                raise ValueError(f"The following concept_ids were requested but do not exist in storage: {difference}. Please check your input or storage contents.")
            
            mask = np.isin(all_ids, concept_ids)
            row_indices = np.where(mask)[0]
            
            if len(row_indices) == 0:
                return {}

            # Sorting row_indices is highly recommended for faster HDF5 disk reads
            row_indices = np.sort(row_indices)
            fetched_embeddings = emb_ds[row_indices]
            fetched_ids = all_ids[row_indices]

            return {
                int(cid): emb.tolist()
                for cid, emb in zip(fetched_ids, fetched_embeddings)
            }


    def get_embeddings(self) -> np.ndarray:
        """Loads all embeddings into RAM. Use with care for large datasets."""
        with self.open_embeddings(mode="r") as emb_ds:
            return emb_ds[:]

    def get_concept_ids(self) -> np.ndarray:
        """Loads all concept_ids into RAM. Use with care for large datasets."""
        with self.open_concept_ids(mode="r") as cid_ds:
            return cid_ds[:] 

    def get_all(self) -> tuple[np.ndarray, np.ndarray]:
        """Loads everything into RAM. Use with care for large datasets."""
        with self.open_all(mode="r") as (emb_ds, cid_ds):
            return cid_ds[:], emb_ds[:]

    def get_count(self) -> int:
        """Returns how many vectors are stored instantly, without loading them."""
        with self.open_concept_ids(mode="r") as f:
            return f.shape[0]
        
    def stream_concept_ids_and_embeddings(
        self,
        batch_size: int = 100_000
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Sequentially reads embeddings and IDs from HDF5 in chunks.
        """

        with self.open_all(mode="r") as (emb_ds, cid_ds):
            total_records = emb_ds.shape[0]
            for start in range(0, total_records, batch_size):
                end = min(start + batch_size, total_records)
                
                # Slicing triggers the actual disk read for this batch only
                batch_embeddings = emb_ds[start:end]
                batch_ids = cid_ds[start:end]
                
                yield batch_ids, batch_embeddings

    def validate_concept_ids(self, concept_ids: np.ndarray):
        assert isinstance(concept_ids, np.ndarray), f"Expected concept_ids to be a numpy array, got {type(concept_ids)}"
        assert concept_ids.ndim == 1, f"Expected concept_ids to be 1D, got {concept_ids.ndim}D"