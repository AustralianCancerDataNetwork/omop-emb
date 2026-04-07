from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional, Sequence, Dict, Tuple 

import numpy as np
from numpy import ndarray
from omop_emb.config import BackendType
from sqlalchemy import Engine
from sqlalchemy.orm import Session
import logging

from omop_emb.model_registry import EmbeddingModelRecord
from omop_emb.config import IndexType, MetricType, ENV_OMOP_EMB_FAISS_INDEX_DIR
from omop_emb.backends.base_backend import EmbeddingBackendBase, require_registered_model
from omop_emb.utils.embedding_utils import (
    EmbeddingConceptFilter,
    NearestConceptMatch,
    get_similarity_from_distance
)
from .storage_manager import EmbeddingStorageManager, EmbeddingStorageError

logger = logging.getLogger(__name__)

class FaissEmbeddingBackend(EmbeddingBackendBase):
    """
    File-based FAISS embedding backend.

    Notes
    -----
    This backend is intentionally conservative and optimized for clarity over
    extreme scale. In particular, updates currently rewrite the per-model numpy
    arrays and rebuild the FAISS index, which is acceptable for a first pass
    but not ideal for very large incremental workloads.
    """

    DEFAULT_FAISS_DIR = ".omop_emb/faiss"
    WARNING_CONCEPT_FILTERS = "Concept filters only applicable to the DatabaseEmbeddingBackend. The FAISS backend without DB support does not know about any OMOP concept metadata, so filters will be ignored in the nearest neighbor search"

    def __init__(self, base_dir: Optional[str | os.PathLike[str]] = None):
        super().__init__()
        self.base_dir = Path(
            base_dir
            or os.getenv(ENV_OMOP_EMB_FAISS_INDEX_DIR)
            or FaissEmbeddingBackend.DEFAULT_FAISS_DIR
        ).expanduser()
        logger.info(f"FAISS backend initialized with base directory: {self.base_dir}")
        self.embedding_storage_managers: Dict[str, EmbeddingStorageManager] = {}

    @property
    def backend_type(self) -> BackendType:
        return BackendType.FAISS

    def initialise_store(self, engine) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        return super().initialise_store(engine)
    
    @require_registered_model
    def has_any_embeddings(
        self, 
        session: Session, 
        model_name: str,
        index_type: IndexType,
        model_record: EmbeddingModelRecord,
    ) -> bool:
        storage_manager = self.get_storage_manager(
            model_name=model_name,
            dimensions=model_record.dimensions,
            index_type=model_record.index_type,
        )
        return storage_manager.get_count() > 0

    
    def get_safe_model_dir(self, model_name: str) -> Path:
        return self.base_dir / self.model_registry.safe_model_name(model_name)
    
    def get_storage_manager(
        self, 
        model_name: str,
        dimensions: int,
        index_type: IndexType,
    ) -> EmbeddingStorageManager:
        self.register_storage_manager(
            model_name=model_name,
            dimensions=dimensions,
            index_type=index_type,
        )
        return self.embedding_storage_managers[model_name]
    
    def register_storage_manager(
        self,
        model_name: str,
        dimensions: int,
        index_type: IndexType,
    ) -> None:
        """Registers a storage manager for the given model if not already registered. This ensures that the necessary on-disk structures are in place for the model's embeddings and index."""

        if model_name not in self.embedding_storage_managers:
            logger.info(f"Registering new storage manager for model '{model_name}' with dimensions={dimensions}, index_type={index_type}")
            self.embedding_storage_managers[model_name] = EmbeddingStorageManager(
                    file_dir=self.get_safe_model_dir(model_name), 
                    dimensions=dimensions,
                    backend_type=self.backend_type,
                )

    def register_model(
        self,
        engine: Engine,
        model_name: str,
        dimensions: int,
        *,
        index_type: IndexType,
        metadata: Mapping[str, object] = {},
    ) -> EmbeddingModelRecord:

        # Create the storage for the model on disk
        safe_name = self.model_registry.safe_model_name(model_name)
        model_dir = self.base_dir / safe_name
        model_dir.mkdir(parents=False, exist_ok=True)
        logger.info(f"Created model directory: {model_dir}")
        return super().register_model(
            engine=engine,
            model_name=model_name,
            dimensions=dimensions,
            index_type=index_type,
            metadata=metadata
        )

    @require_registered_model
    def upsert_embeddings(
        self,
        session: Session,
        model_name: str,
        index_type: IndexType,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
        embeddings: ndarray,
        metric_type: Optional[MetricType] = None,
    ) -> None:
        """
        Insert or update vector embeddings for a collection of OMOP concept IDs.

        This method presents an interface for persisting generated embeddings. 
        The FAISS backend implementation stages the provided embeddings in an HDF5 file and updates the corresponding FAISS index on disk. 
        Additionally, it ensures that the mapping between concept IDs and their positions in the FAISS index is maintained in a SQLAlchemy-managed registry table.

        Parameters
        ----------
        session : sqlalchemy.orm.Session
            The active database session used for transactional persistence and 
            model metadata updates.
        model_name : str
            The unique identifier or name of the embedding model (e.g., 
            'text-embedding-3-small').
        model_record : EmbeddingModelRecord
            A record object containing metadata, dimensions, and configuration 
            specific to the embedding model being processed.
        concept_ids : Sequence[int]
            A sequence of OMOP standard concept IDs corresponding to the 
            ordered rows in the embeddings array.
        embeddings : numpy.ndarray
            A 2D array of shape (n_concepts, n_dimensions) containing the 
            generated vector representations.
        metric_type : Optional[MetricType]
            The similarity metric type (e.g., COSINE, EUCLIDEAN) that should be 
            associated with the stored embeddings for accurate nearest neighbor 
            search behavior. If None, the index will not be built and the raw
            embeddings are only stored in the HDF5 file for later use.

        Returns
        -------
        None

        Notes
        -----
        - Indices in FAISS are directly tied to a metric for optimisation. Providing
        no metric_type means we won't be able to create the index and will only store the raw embeddings.
        The index will be created upon :meth:`~FaissEmbeddingBackend.get_nearest_concepts`, where a `metric_type` is required.
        """

        concept_id_tuple = tuple(concept_ids)
        self.validate_embeddings_and_concept_ids(
            concept_ids=concept_id_tuple,
            embeddings=embeddings,
            dimensions=model_record.dimensions,
        )        
        storage_manager = self.get_storage_manager(
            model_name=model_name,
            dimensions=model_record.dimensions,
            index_type=model_record.index_type,
        )
        storage_manager.append(
            concept_ids=np.array(concept_id_tuple, dtype=np.int64),
            embeddings=embeddings,
            index_type=model_record.index_type,
            metric_type=metric_type
        )

    @require_registered_model
    def get_nearest_concepts(
        self,
        session: Session,
        model_name: str,
        model_record: EmbeddingModelRecord,
        query_embeddings: np.ndarray,
        metric_type: MetricType,
        *,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        k: int = 10,
    ) -> Tuple[Tuple[int, ...], ...]:
        
        if concept_filter is not None:
            logger.warning(self.WARNING_CONCEPT_FILTERS)
        
        storage_manager = self.get_storage_manager(
            model_name=model_name,
            dimensions=model_record.dimensions,
            index_type=model_record.index_type,
        )
        if storage_manager.get_count() == 0:
            return ()

        self.validate_embeddings(embeddings=query_embeddings, dimensions=model_record.dimensions)
        
        distances, concept_ids = storage_manager.search(
            query_vector=query_embeddings,
            metric_type=metric_type,
            index_type=model_record.index_type,
            k=k,
            subset_concept_ids=None
        )

        matches = []
        for concept_id_per_query, distance_per_query in zip(concept_ids, distances):
            matches_per_query = []
            for concept_id, distance in zip(concept_id_per_query, distance_per_query):
                if concept_id == -1:  # Skip empty for k>total_concepts 
                    continue
                matches_per_query.append(NearestConceptMatch(
                    concept_id=int(concept_id),
                    concept_name=None,
                    similarity=get_similarity_from_distance(distance, metric_type),
                    is_standard=None,
                    is_active=None,
                ))
            matches.append(tuple(matches_per_query))
        return tuple(matches)
    
    @require_registered_model
    def get_embeddings_by_concept_ids(
        self, 
        session: Session, 
        model_name: str, 
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int]
    ) -> Mapping[int, Sequence[float]]:
        concept_id_tuple = tuple(concept_ids)
        if not concept_id_tuple:
            return {}
        
        concept_ids_np = np.array(concept_id_tuple, dtype=np.int64)

        storage_manager = self.get_storage_manager(
            model_name=model_name,
            dimensions=model_record.dimensions,
            index_type=model_record.index_type,
        )

        return storage_manager.get_embeddings_by_concept_ids(concept_ids=concept_ids_np)
