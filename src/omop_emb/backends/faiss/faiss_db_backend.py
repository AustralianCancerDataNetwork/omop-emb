from __future__ import annotations

from typing import Optional, Sequence, Type, Tuple 

from sqlalchemy import  Engine, Result, select, Select, case, literal, column, func, values, Integer
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from orm_loader.helpers import Base
from omop_alchemy.cdm.model.vocabulary import Concept

import logging
import numpy as np

from omop_emb.model_registry import ModelRegistry, EmbeddingModelRecord
from omop_emb.config import IndexType, MetricType, BackendType
from omop_emb.backends.base_backend import  require_registered_model
from omop_emb.backends.database_backend import DatabaseEmbeddingBackend, ConceptIDEmbeddingBase
from omop_emb.utils.embedding_utils import (
    EmbeddingConceptFilter,
    NearestConceptMatch,
    get_similarity_from_distance
)

from .faiss_backend import FaissEmbeddingBackend
from .storage_manager import EmbeddingStorageError

logger = logging.getLogger(__name__)


class FaissConceptIDEmbeddingTable(ConceptIDEmbeddingBase, Base):
    """Registry table to track which concept_ids are present in the FAISS/H5 storage for a specific model."""
    __abstract__ = True

class FaissEmbeddingBackendWithDBSupport(
    FaissEmbeddingBackend,
    DatabaseEmbeddingBackend[FaissConceptIDEmbeddingTable]
):
    """
    File-based FAISS embedding backend with a SQLAlchemy-managed registry for concept ID tracking.
    """

    @property
    def backend_type(self) -> BackendType:
        return BackendType.FAISS_DB

    def _create_storage_table(self, engine: Engine, model_registry_entry: ModelRegistry) -> Type[FaissConceptIDEmbeddingTable]:
        tablename = model_registry_entry.storage_identifier
        mapping_table = type(
            tablename,
            (FaissConceptIDEmbeddingTable, ),
            {
                "__tablename__": tablename,
                "__table_args__": {"extend_existing": True},
            },
        )
        Base.metadata.create_all(engine, tables=[mapping_table.__table__])  # type: ignore[arg-type]
        logger.debug(
            f"Initialized {FaissConceptIDEmbeddingTable.__name__} table for model '{model_registry_entry.model_name}'",
        )
        return mapping_table

    @require_registered_model
    def upsert_embeddings(
        self,
        session: Session,
        model_name: str,
        index_type: IndexType,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
        embeddings: np.ndarray,
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

        super().upsert_embeddings(
            session=session,
            model_name=model_name,
            index_type=index_type,
            model_record=model_record,
            concept_ids=concept_ids,
            embeddings=embeddings,
            metric_type=metric_type
        )

        concept_id_tuple = tuple(concept_ids)
        self.validate_embeddings_and_concept_ids(
            concept_ids=concept_id_tuple,
            embeddings=embeddings,
            dimensions=model_record.dimensions,
        )  
        try:
            assert session.bind is not None, "Session must be bound to an engine"
            assert session.bind.dialect.name == "postgresql", "This function is only implemented for PostgreSQL databases"
            registered_table = self._get_embedding_table(
                session=session,
                model_name=model_name,
            )
            stmt = insert(registered_table).values(list({"concept_id": cid} for cid in concept_ids))
            session.execute(stmt)
            session.commit()
        except Exception as e:
            session.rollback()
            raise ValueError(
                f"Failed to add concept IDs to FAISS registry for model '{model_name}'. This may be due to a mismatch between the provided concept_ids and the existing entries in the database, or a database constraint violation. Original error: {str(e)}"
            ) from e

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
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        
        storage_manager = self.get_storage_manager(
            model_name=model_name,
            dimensions=model_record.dimensions,
            index_type=model_record.index_type,
        )
        existing_concept_ids = storage_manager.get_concept_ids()

        if storage_manager.get_count() == 0:
            return ()

        self.validate_embeddings(embeddings=query_embeddings, dimensions=model_record.dimensions)
        q_permitted_concept_ids = self.q_concept_ids_with_embeddings(
            existing_concept_ids=existing_concept_ids,
            concept_filter=concept_filter,
            limit=None
        )

        permitted_concept_ids_storage = {
            row.concept_id: row for row in session.execute(q_permitted_concept_ids)
        }

        if concept_filter is None:
            # Easier to not do any filter if all are allowed
            permitted_concept_ids = None
        else:
            permitted_concept_ids = np.array(list(permitted_concept_ids_storage.keys()), dtype=np.int64)

        distances, concept_ids = storage_manager.search(
            query_vector=query_embeddings,
            metric_type=metric_type,
            index_type=model_record.index_type,
            k=k,
            subset_concept_ids=permitted_concept_ids
        )

        matches = []
        for concept_id_per_query, distance_per_query in zip(concept_ids, distances):
            matches_per_query = []
            for concept_id, distance in zip(concept_id_per_query, distance_per_query):
                if concept_id == -1:  # Skip empty for k>total_concepts 
                    continue
                row = permitted_concept_ids_storage.get(concept_id)
                if row is None:
                    logger.warning(f"Concept ID {concept_id} returned by FAISS search but not found in permitted concept IDs. This indicates a mismatch between the FAISS index and the database registry. Skipping this result.")
                    continue
                matches_per_query.append(NearestConceptMatch(
                    concept_id=int(concept_id),
                    concept_name=row.concept_name,
                    similarity=get_similarity_from_distance(distance, metric_type),
                    is_standard=bool(row.is_standard),
                    is_active=bool(row.is_active),
                ))
            matches.append(tuple(matches_per_query))
        return tuple(matches)
    
    @require_registered_model
    def get_concepts_without_embedding(
        self,
        session: Session,
        model_name: str,
        index_type: IndexType,
        model_record: EmbeddingModelRecord,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: Optional[int] = None,
        metric_type: Optional[MetricType] = None,
    ) -> Result:
        """Return concept IDs and names for concepts that do not have embeddings."""
        storage_manager = self.get_storage_manager(
            model_name=model_name,
            dimensions=model_record.dimensions,
            index_type=model_record.index_type,
        )
        try:
            existing_concept_ids = storage_manager.get_concept_ids()
        except EmbeddingStorageError as e:
            logger.error(f"Error retrieving concept IDs from storage manager for model '{model_name}': {e}")
            if metric_type is not None:
                try:
                    existing_concept_ids_np = storage_manager.get_concept_ids_from_index(index_type=model_record.index_type, metric_type=metric_type)
                    existing_concept_ids = tuple(existing_concept_ids_np.tolist())
                except EmbeddingStorageError as e_idx:
                    logger.error(f"Error retrieving concept IDs from index for model '{model_name}': {e_idx}")
                    raise ValueError(f"Failed to retrieve existing concept IDs from both storage and index for model '{model_name}'. Cannot proceed with counting concepts without embeddings.") from e_idx

        query = self.q_concepts_without_embeddings(
            existing_concept_ids=existing_concept_ids,
            concept_filter=concept_filter,
            limit=limit
        )
        return session.execute(
            query, 
            execution_options={"stream_results": True}
        )
    
    
    @require_registered_model
    def get_concepts_without_embedding_count(
        self,
        session: Session,
        model_name: str,
        index_type: IndexType,
        model_record: EmbeddingModelRecord,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        metric_type: Optional[MetricType] = None,
    ) -> int:

        storage_manager = self.get_storage_manager(
            model_name=model_name,
            dimensions=model_record.dimensions,
            index_type=model_record.index_type,
        )
        try:
            existing_concept_ids = storage_manager.get_concept_ids()
        except EmbeddingStorageError as e:
            logger.error(f"Error retrieving concept IDs from storage manager for model '{model_name}': {e}")
            if metric_type is not None:
                try:
                    existing_concept_ids_np = storage_manager.get_concept_ids_from_index(index_type=model_record.index_type, metric_type=metric_type)
                    existing_concept_ids = tuple(existing_concept_ids_np.tolist())
                except EmbeddingStorageError as e_idx:
                    logger.error(f"Error retrieving concept IDs from index for model '{model_name}': {e_idx}")
                    raise ValueError(f"Failed to retrieve existing concept IDs from both storage and index for model '{model_name}'. Cannot proceed with counting concepts without embeddings.") from e_idx

        query = self.q_count_concepts_without_embeddings(
            existing_concept_ids=existing_concept_ids,
            concept_filter=concept_filter
        )
        result = session.execute(query).scalar_one()
        return int(result)
    
    @staticmethod
    def q_concept_ids_with_embeddings(
        existing_concept_ids: Tuple[int, ...],
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: Optional[int] = None,
    ) -> Select:
        
        v_table = values(
            column("id", Integer),
            name="faiss_ids"
        ).data([(i,) for i in existing_concept_ids])

        # 2. Build the main statement
        stmt = (
            select(
                Concept.concept_id,
                Concept.concept_name,
                case(
                    (Concept.standard_concept.in_(["S", "C"]), literal(True)),
                    else_=literal(False),
                ).label("is_standard"),
                case(
                    (Concept.invalid_reason.in_(["D", "U"]), literal(False)),
                    else_=literal(True),
                ).label("is_active"),
            )
            # 3. Replace the Join with a semi-join (EXISTS) for speed
            .where(
                select(1).where(v_table.c.id == Concept.concept_id).exists()
            )
        )

        if concept_filter is not None:
            stmt = concept_filter.apply(stmt)
            
        return stmt.limit(limit)
    
    @staticmethod
    def q_concepts_without_embeddings(
        existing_concept_ids: Tuple[int, ...],
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: Optional[int] = None,
    ) -> Select:
        
        if not existing_concept_ids:
            query = select(Concept.concept_id, Concept.concept_name)
        else:
            v_table = values(
                column("id", Integer),
                name="exclude_list"
            ).data([(i,) for i in existing_concept_ids])

            query = select(Concept.concept_id, Concept.concept_name).where(
                ~select(1).where(v_table.c.id == Concept.concept_id).exists()
            )
        if concept_filter is not None:
            query = concept_filter.apply(query)

        return query.limit(limit)

    @staticmethod
    def q_count_concepts_without_embeddings(
        existing_concept_ids: Tuple[int, ...],
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> Select:
        
        if not existing_concept_ids:
            query = select(func.count()).select_from(Concept)
        else:
            v_table = values(
                column("id", Integer),
                name="exclude_list"
            ).data([(i,) for i in existing_concept_ids])

            query = select(func.count()).where(
                ~select(1).where(v_table.c.id == Concept.concept_id).exists()
            )
        if concept_filter is not None:
            query = concept_filter.apply(query)

        return query