"""Embedding storage manager for FAISS backend using HDF5 files.
Instead of directly converting the embeddings into an index,
we store the raw embeddings and their corresponding concept_ids in an HDF5 file on disk.
This allows us to generate indices on the fly and also easily update the embeddings without needing to rebuild the entire index."""

import h5py
import numpy as np
import shutil
from pathlib import Path
from contextlib import contextmanager
from typing import Callable, cast, Generator, Iterable, Tuple, Optional, Mapping, Sequence, Dict
import logging
logger = logging.getLogger(__name__)

from omop_emb.config import BackendType, IndexType, MetricType
from ..index_config import IndexConfig, FlatIndexConfig, HNSWIndexConfig
from .faiss_index_manager import FaissFlatIndexManager, FaissHNSWIndexManager, FaissBaseIndexManager

_INDEX_MANAGER_FOR_CONFIG: dict[type[IndexConfig], Callable[..., FaissBaseIndexManager]] = {
    FlatIndexConfig: FaissFlatIndexManager,
    HNSWIndexConfig: FaissHNSWIndexManager,
}


class EmbeddingStorageManager:
    """Storage manager for raw embeddings and concept_ids using HDF5 files.

    Also manages the creation of IndexManager instances for different
    index types.

    Notes
    -----
    - The HDF5 file contains two datasets: 'embeddings' (float32, shape NxD) and 'concept_ids' (int64, shape N).

    """
    HDF5_DATASET_NAME_EMBEDDINGS = "embeddings"
    HDF5_DATASET_NAME_CONCEPT_IDS = "concept_ids"
    HDF5_EMBEDDINGS_FILE = "embeddings.h5"

    def __init__(
        self,
        file_dir: str | Path,
        dimensions: int,
        backend_type: BackendType,
    ):
        if backend_type != BackendType.FAISS:
            raise ValueError(f"EmbeddingStorageManager is designed for FAISS backend. Got {backend_type}.")

        self.base_dir = Path(file_dir)
        self.dimensions = dimensions
        self._init_embedding_storage_if_missing()
        self._index_managers: Dict[IndexType, FaissBaseIndexManager] = {}

    @property
    def embeddings_filepath(self) -> Path:
        return self.base_dir / self.HDF5_EMBEDDINGS_FILE

    
    def rebuild_index_for_metric(
        self,
        metric_type: MetricType,
        index_config: IndexConfig,
        batch_size: int = 100_000,
    ) -> None:
        """Delete the on-disk FAISS index for the given metric and rebuild from HDF5.

        Forces a full rebuild even when an index file already exists on disk.
        Accepts ``index_config`` so it can create the manager from scratch on a cold
        start where no manager is in memory yet.
        """
        logger.info(f"Rebuilding {index_config.index_type.value}/{metric_type.value} from HDF5 storage.")
        index_manager = self._get_or_create_index_manager(index_config)
        index_manager.rebuild_index(
            metric_type=metric_type,
            data_stream=self.stream_concept_ids_and_embeddings(batch_size=batch_size),
            expected_count=self.get_count(),
        )
        logger.info(f"Finished rebuilding {index_config.index_type.value}/{metric_type.value}.")

    def create_index_for_metric(
        self,
        metric_type: MetricType,
        index_config: IndexConfig,
        batch_size: int = 100_000,
    ) -> None:
        """Load or build the FAISS index for the given metric from HDF5 storage.

        If an index file already exists on disk it is loaded; otherwise it is
        populated from HDF5 and saved.  Note: if an index file for a *different*
        config of the same index type already exists on disk, it will be loaded as-is
        (the existing file is not deleted).  Use ``rebuild_index_for_metric`` to force
        a clean rebuild with the new config.
        """
        logger.info(f"Initialising {index_config.index_type.value}/{metric_type.value} index.")
        index_manager = self._get_or_create_index_manager(index_config)
        index_manager.load_or_create(
            data_stream=self.stream_concept_ids_and_embeddings(batch_size=batch_size),
            metric_type=metric_type,
            expected_count=self.get_count(),
        )
        logger.info(f"Finished initialising {index_config.index_type.value}/{metric_type.value} index.")

    def _get_or_create_index_manager(self, index_config: IndexConfig) -> FaissBaseIndexManager:
        """Return the cached manager for this index type, creating it from *index_config* if absent."""
        existing = self._index_managers.get(index_config.index_type)
        if existing is not None:
            return existing

        manager_cls = _INDEX_MANAGER_FOR_CONFIG.get(type(index_config))
        if manager_cls is None:
            raise ValueError(f"Unsupported index config type: {type(index_config)}")
        manager: FaissBaseIndexManager = manager_cls(
            dimension=self.dimensions,
            base_index_dir=self.base_dir,
            index_config=index_config,
        )
        self._index_managers[index_config.index_type] = manager
        return manager

    def get_index_manager(self, index_type: IndexType) -> FaissBaseIndexManager:
        """Return the cached manager for *index_type*, or raise if none has been created yet."""
        manager = self._index_managers.get(index_type)
        if manager is None:
            raise ValueError(
                f"No index manager for index type '{index_type.value}'. "
                "Call create_index_for_metric() first."
            )
        return manager


    def _init_embedding_storage_if_missing(self):
        """Creates the .h5 file with resizable datasets if it doesn't exist."""
        if not self.embeddings_filepath.exists():
            logger.info(f"Creating new HDF5 storage at {self.embeddings_filepath}")
            with h5py.File(self.embeddings_filepath, 'w') as f:
                # maxshape=(None, dim) is the magic that allows appending!
                f.create_dataset(
                    self.HDF5_DATASET_NAME_EMBEDDINGS, 
                    shape=(0, self.dimensions), 
                    maxshape=(None, self.dimensions), 
                    dtype='float32',
                    chunks=True # Required for resizing
                )
                f.create_dataset(
                    self.HDF5_DATASET_NAME_CONCEPT_IDS, 
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
        with h5py.File(self.embeddings_filepath, mode) as f:
            embeddings: h5py.Dataset = cast(h5py.Dataset, f[self.HDF5_DATASET_NAME_EMBEDDINGS])
            yield embeddings

    @contextmanager
    def open_concept_ids(self, mode: str) -> Generator[h5py.Dataset, None, None]:
        """Context manager for the concept_id dataset."""
        with h5py.File(self.embeddings_filepath, mode) as f:
            concept_ids: h5py.Dataset = cast(h5py.Dataset, f[self.HDF5_DATASET_NAME_CONCEPT_IDS])
            yield concept_ids

    @contextmanager
    def open_all(self, mode: str) -> Generator[tuple[h5py.Dataset, h5py.Dataset], None, None]:
        """Context manager that yields both datasets together."""
        with h5py.File(self.embeddings_filepath, mode) as f:
            embeddings: h5py.Dataset = cast(h5py.Dataset, f[self.HDF5_DATASET_NAME_EMBEDDINGS])
            concept_ids: h5py.Dataset = cast(h5py.Dataset, f[self.HDF5_DATASET_NAME_CONCEPT_IDS])
            yield embeddings, concept_ids

    def _bulk_write(
        self,
        batches: Iterable[Tuple[np.ndarray, np.ndarray]],
    ) -> list[int]:
        """Write ``(concept_ids, embeddings)`` batches to HDF5 in a single file open.

        Parameters
        ----------
        batches : Iterable[Tuple[np.ndarray, np.ndarray]]
            Lazy iterable of ``(concept_ids, embeddings)`` pairs. ``concept_ids``
            must be 1-D int64; ``embeddings`` float32 of shape ``(batch_size, D)``.
            Wrap with ``tqdm`` for a progress bar.

        Returns
        -------
        list[int]
            All concept IDs written, in order. Pass to the SQL registry insert
            to keep HDF5 and database in sync.

        Notes
        -----
        No FAISS operations — caller's responsibility.
        Call ``rebuild_index_for_metric`` separately after this returns.
        Raises ``ValueError`` if any concept_id appears more than once across
        all batches (cross-batch deduplication).
        """
        all_ids: list[int] = []
        seen_ids: set[int] = set()
        with self.open_all(mode="a") as (emb_ds, cid_ds):
            for concept_ids, embeddings in batches:
                if embeddings.shape[1] != self.dimensions:
                    raise ValueError(
                        f"Embeddings dimension {embeddings.shape[1]} does not match expected {self.dimensions}"
                    )
                batch_ids = concept_ids.tolist()
                duplicates = seen_ids.intersection(batch_ids)
                if duplicates:
                    raise ValueError(
                        f"Duplicate concept_ids detected across batches: {duplicates}. "
                        "Each concept_id may appear at most once per bulk upsert."
                    )
                seen_ids.update(batch_ids)
                old_size = emb_ds.shape[0]
                new_size = old_size + len(concept_ids)
                emb_ds.resize((new_size, self.dimensions))
                cid_ds.resize((new_size,))
                emb_ds[old_size:new_size] = embeddings.astype("float32")
                cid_ds[old_size:new_size] = concept_ids.astype("int64")
                all_ids.extend(batch_ids)
            emb_ds.file.flush()
        return all_ids

    def _truncate_to(self, count: int) -> None:
        """Shrink HDF5 storage back to *count* vectors.

        Used to roll back a ``_bulk_write`` when the subsequent SQL commit fails,
        keeping HDF5 and the SQL registry in sync.
        """
        with self.open_all(mode="a") as (emb_ds, cid_ds):
            emb_ds.resize((count, self.dimensions))
            cid_ds.resize((count,))
            emb_ds.file.flush()
        logger.warning(
            f"HDF5 storage truncated to {count} vectors (rollback after failed SQL commit)."
        )

    def append(
        self,
        concept_ids: np.ndarray,
        embeddings: np.ndarray,
        index_config: Optional[IndexConfig] = None,
        metric_type: Optional[MetricType] = None,
    ) -> None:
        """Append new vectors to HDF5 and optionally update a FAISS index.

        Notes
        -----
        - This method does not check for duplicates across different calls. It is the caller's responsibility to ensure that the same concept_id is not appended multiple times across different calls, as this would lead to data corruption and incorrect search results. 

        Parameters
        ----------
        index_config : IndexConfig, optional
            When provided together with *metric_type*, the corresponding FAISS index
            is created on demand (if it doesn't exist yet) and updated with the new
            vectors. Both must be supplied together or both omitted.
        metric_type : MetricType, optional
            See *index_config*.
        """
        if embeddings.shape[1] != self.dimensions:
            raise ValueError(f"Embeddings dimension {embeddings.shape[1]} does not match expected {self.dimensions}")
        if embeddings.shape[0] != concept_ids.shape[0]:
            raise ValueError("Number of embeddings must match number of concept_ids")
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if (index_config is None) != (metric_type is None):
            raise ValueError(
                "index_config and metric_type must both be provided together, or both omitted. "
                f"Got index_config={index_config!r}, metric_type={metric_type!r}."
            )

        unique = np.unique(concept_ids)
        if len(unique) != len(concept_ids):
            raise ValueError("Duplicate concept_ids found in the input. Please ensure all concept_ids are unique to prevent data corruption.")

        with self.open_all(mode="a") as (emb_ds, cid_ds):
            old_size = emb_ds.shape[0]
            new_size = old_size + embeddings.shape[0]
            emb_ds.resize((new_size, self.dimensions))
            cid_ds.resize((new_size,))
            emb_ds[old_size:new_size] = embeddings.astype('float32')
            cid_ds[old_size:new_size] = concept_ids.astype('int64')
            emb_ds.file.flush()

        if index_config is not None and metric_type is not None:
            index_manager = self._get_or_create_index_manager(index_config)
            index_manager.add(concept_ids, embeddings, metric_type=metric_type)
            index_manager.save(metric_type=metric_type)

    def search(
        self,
        query_vector: np.ndarray,
        metric_type: MetricType,
        index_type: IndexType,
        k: int = 5,
        subset_concept_ids: Optional[np.ndarray] = None,
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
        index_type : IndexType
            The type of index to search against, which determines which index manager is used and how the search is performed.
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
        index_manager = self.get_index_manager(index_type=index_type)
        return index_manager.search(
            query_vector=query_vector,
            k=k,
            metric_type=metric_type,
            subset_concept_ids=subset_concept_ids,
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
        if not isinstance(concept_ids, np.ndarray):
            raise TypeError(f"Expected concept_ids to be a numpy array, got {type(concept_ids)}")
        if concept_ids.ndim != 1:
            raise ValueError(f"Expected concept_ids to be 1D, got {concept_ids.ndim}D")
        
    def cleanup(self, index_type: IndexType) -> None:
        """Delete all FAISS index files for *index_type* and evict from memory.

        Works even when no manager has been created in this process yet — in that
        case the index directory is removed directly from disk.
        """
        manager = self._index_managers.pop(index_type, None)
        if manager is not None:
            manager.cleanup()
        else:
            # Manager was never loaded into memory this session; remove files directly.
            index_dir = self.base_dir / f"index_{index_type.value}"
            if index_dir.exists():
                shutil.rmtree(index_dir)
                logger.info(f"Deleted index directory {index_dir} (no in-memory manager).")

    def cleanup_metric(self, metric_type: MetricType, index_type: IndexType) -> None:
        """Delete the FAISS index file for *metric_type* only, keeping HDF5 and other metrics intact."""
        manager = self._index_managers.get(index_type)
        if manager is not None:
            manager.cleanup_metric(metric_type=metric_type)
        else:
            # Remove the file directly if the manager was never loaded.
            index_file = self.base_dir / f"index_{index_type.value}" / f"{index_type.value}_{metric_type.value}_index.faiss"
            if index_file.exists():
                index_file.unlink()
                logger.info(f"Deleted index file {index_file} (no in-memory manager).")