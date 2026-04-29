from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Optional, Sequence, Union, Type, TypeVar, Generic, Any, Callable, Tuple
from pathlib import Path
from functools import wraps
import os
import logging

from numpy import ndarray
from sqlalchemy import Engine, select, Integer, ForeignKey, func, Select
from sqlalchemy.orm import Session, mapped_column

from omop_alchemy.cdm.model.vocabulary import Concept

from ..model_registry import EmbeddingModelRecord, ModelRegistryManager
from ..config import BackendType, IndexType, MetricType, ENV_BASE_STORAGE_DIR, ProviderType
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter, NearestConceptMatch
from .index_config import IndexConfig, INDEX_CONFIG_METADATA_KEY

logger = logging.getLogger(__name__)


def require_registered_model(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(
        self,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        **kwargs
    ) -> Any:
        record = self.get_registered_model(model_name=model_name, index_type=index_type, provider_type=provider_type)
        if record is None:
            raise ValueError(
                f"Embedding model '{model_name}' with index_type='{index_type.value}' is not registered in backend '{self.backend_type.value}'."
            )
        kwargs["_model_record"] = record
        return func(self, model_name=model_name, provider_type=provider_type, index_type=index_type, **kwargs)
    return wrapper


class ConceptIDEmbeddingBase:
    """Abstract mixin to ensure consistent concept_id handling across backends."""
    __tablename__: str

    concept_id = mapped_column(
        Integer,
        ForeignKey(Concept.concept_id, ondelete="CASCADE"),
        primary_key=True,
    )


T = TypeVar("T", bound=ConceptIDEmbeddingBase)


class EmbeddingBackend(ABC, Generic[T]):
    """Abstract interface for swappable embedding storage and retrieval backends.

    Notes
    -----
    - The `engine` and `session` parameters refer to the OMOP CDM database connection; the model_registry on disk has its own engine
        and is managed separately by the ModelRegistryManager.
    """

    DEFAULT_BASE_STORAGE_DIR = Path.home() / ".omop_emb"
    DEFAULT_K_NEAREST = 10

    def __init__(
        self,
        storage_base_dir: Optional[str | Path] = None,
        registry_db_name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        storage_base_dir : str | Path, optional
            Local base directory used for backend metadata and file-based assets.
            Resolution order:
            1. explicit ``storage_base_dir`` argument
            2. ``OMOP_EMB_BASE_STORAGE_DIR`` environment variable
            3. ``DEFAULT_BASE_STORAGE_DIR``
            The resolved value must be an absolute path.
        registry_db_name : str, optional
            Optional registry database filename. If omitted, backend defaults are used.
        """
        super().__init__()

        storage_base_dir = (
            storage_base_dir
            or os.getenv(ENV_BASE_STORAGE_DIR)
            or self.DEFAULT_BASE_STORAGE_DIR
        )
        resolved_storage_base_dir = Path(storage_base_dir).expanduser()
        if not resolved_storage_base_dir.is_absolute():
            raise ValueError(
                "storage_base_dir must be an absolute path. "
                f"Got '{storage_base_dir}'."
            )

        self._storage_base_dir = resolved_storage_base_dir
        self._storage_base_dir.mkdir(parents=True, exist_ok=True)

        self._embedding_table_cache: dict[Tuple[str, ProviderType, BackendType, IndexType], Type[T]] = {}
        self._embedding_model_registry = ModelRegistryManager(
            base_dir=self.storage_base_dir,
            db_file=registry_db_name,
        )

    # ------------------------------------------------------------------
    # Backend identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """General category of backend, used for index compatibility checks."""

    @property
    def backend_name(self) -> str:
        """Stable identifier for this backend implementation."""
        return self.backend_type.value

    @property
    def storage_base_dir(self) -> Path:
        """Base directory for any local storage needs of the backend."""
        return self._storage_base_dir
    
    @property
    def embedding_model_registry(self) -> ModelRegistryManager:
        """Access the model registry manager for this backend."""
        return self._embedding_model_registry

    # ------------------------------------------------------------------
    # Store lifecycle
    # ------------------------------------------------------------------

    def initialise_store(self, engine: Engine) -> None:
        """Initialise the model registry and populate the embedding table cache.

        Calls ``pre_initialise_store`` first so backends can run setup steps
        (e.g. creating Postgres extensions or staging directories) before the
        registry is queried.
        """
        self.pre_initialise_store(engine)
        registered_models = self._embedding_model_registry.get_registered_models_from_db(
            backend_type=self.backend_type
        )
        for model_record in registered_models:
            self._cache_model_record(engine=engine, model_record=model_record)

    def pre_initialise_store(self, engine: Engine) -> None:
        """Hook for backend-specific setup that must run before store initialisation."""
        pass

    # ------------------------------------------------------------------
    # Model registration / deletion / configuration
    # ------------------------------------------------------------------

    def register_model(
        self,
        *,
        engine: Engine,
        model_name: str,
        dimensions: int,
        provider_type: ProviderType,
        index_config: IndexConfig,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> EmbeddingModelRecord:
        """Shared template method for model registration.

        Parameters
        ----------
        engine : Engine
            SQLAlchemy engine for OMOP CDM database storage.
        model_name : str
            Canonical name of the embedding model.
        provider_type : ProviderType
            Provider type of the model_name.
        dimensions : int
            Embedding dimensionality $D$ for this model.
        index_config : IndexConfig
            Index configuration (type + parameters) for this model.
        metadata : Mapping[str, object], optional
            Additional metadata persisted with the model registration.
            Must not contain reserved keys (e.g. ``"index_config"``).
        """
        metadata = metadata or {}
        persisted_metadata: dict[str, object] = {
            INDEX_CONFIG_METADATA_KEY: index_config.to_dict(),
            **(self._validate_external_metadata(metadata)),
        }
        model_record = self._embedding_model_registry.register_model(
            model_name=model_name,
            provider_type=provider_type,
            dimensions=dimensions,
            backend_type=self.backend_type,
            index_type=index_config.index_type,
            metadata=persisted_metadata,
        )
        self._cache_model_record(engine=engine, model_record=model_record)
        return model_record

    def delete_model(
        self,
        engine: Engine,
        model_name: str,
        *,
        provider_type: ProviderType,
        index_type: IndexType,
    ) -> None:
        """Delete a registered model and all associated embeddings.

        This is irreversible. Removes the registry row, drops the dynamic
        embedding table, and (for file-backed backends such as FAISS) deletes
        any associated index files from disk.
        """
        record = self.get_registered_model(
            model_name=model_name,
            provider_type=provider_type,
            index_type=index_type,
        )
        if record is None:
            raise ValueError(
                f"Model '{model_name}' with provider='{provider_type.value}' and "
                f"index_type='{index_type.value}' is not registered."
            )

        cache_key = self._get_embedding_table_cache_key(model_name, provider_type, index_type)
        embedding_table = self._embedding_table_cache.pop(cache_key, None)
        if embedding_table is None:
            logger.warning(
                f"Embedding table for model '{model_name}' with provider='{provider_type.value}' and "
                f"index_type='{index_type.value}' not found in cache during deletion. "
                "Proceeding with registry deletion and storage cleanup, but this may indicate that the storage was not properly initialised or that the model was not registered correctly."
            )
        self._delete_storage_table(engine=engine, model_record=record)

        self._embedding_model_registry.delete_model(
            model_name=model_name,
            provider_type=provider_type,
            backend_type=self.backend_type,
            index_type=index_type,
        )
        logger.info(f"Deleted model '{model_name}' from registry.")

    def update_model_index_configuration(
        self,
        model_name: str,
        *,
        provider_type: ProviderType,
        index_type: IndexType,
        index_config: IndexConfig,
    ) -> EmbeddingModelRecord:
        """Persist updated index configuration parameters for an existing model.

        Only metadata (e.g. HNSW ``num_neighbors``, ``ef_search``) is changed.
        Callers should follow up with :meth:`rebuild_model_indexes` to apply the
        new parameters to the on-disk index files.
        """
        if index_config.index_type != index_type:
            raise ValueError(
                f"index_config.index_type ({index_config.index_type!r}) must match "
                f"the registered index_type ({index_type!r}). "
                "Use delete_model + register_model to change the index type."
            )
        new_metadata = {INDEX_CONFIG_METADATA_KEY: index_config.to_dict()}
        return self._embedding_model_registry.update_model_metadata(
            model_name=model_name,
            provider_type=provider_type,
            backend_type=self.backend_type,
            index_type=index_type,
            metadata=new_metadata,
        )

    @abstractmethod
    @require_registered_model
    def rebuild_model_indexes(
        self,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        *,
        engine: Engine,
        metric_types: Sequence[MetricType],
        batch_size: int = 100_000,
    ) -> None:
        """Rebuild backend-specific indexes for *metric_types*.

        Parameters
        ----------
        model_name : str
            Registered canonical name of the embedding model.
        provider_type : ProviderType
            Provider type the model was registered with.
        index_type : IndexType
            Index type the model was registered with.
        engine : Engine
            SQLAlchemy engine for the OMOP CDM database. May be needed by
            backends that store index state in the database (e.g. pgvector).
        metric_types : Sequence[MetricType]
            Metrics for which indexes should be rebuilt.
        batch_size : int, optional
            Chunk size when streaming vectors from storage. Meaningful for
            file-backed backends (FAISS); ignored by SQL-native backends.
        """
        pass

    # ------------------------------------------------------------------
    # Model registry queries
    # ------------------------------------------------------------------

    def get_registered_model(
        self,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
    ) -> Optional[EmbeddingModelRecord]:
        """Return metadata for one registered model, or ``None`` if not found."""
        registered_model = self._embedding_model_registry.get_registered_models_from_db(
            backend_type=self.backend_type,
            model_name=model_name,
            index_type=index_type,
            provider_type=provider_type,
        )
        if not registered_model:
            return None
        if len(registered_model) != 1:
            raise RuntimeError(
                f"Expected exactly one registered model for name '{model_name}', "
                f"provider '{provider_type.value}', and index type '{index_type.value}', "
                f"but found {len(registered_model)}. This indicates a data integrity issue "
                f"in the model registry database."
            )
        return registered_model[0]

    def is_model_registered(
        self,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
    ) -> bool:
        """Convenience wrapper over ``get_registered_model``."""
        return self.get_registered_model(
            model_name=model_name,
            index_type=index_type,
            provider_type=provider_type,
        ) is not None

    def get_registered_models(
        self,
        model_name: Optional[str] = None,
        provider_type: Optional[ProviderType] = None,
        index_type: Optional[IndexType] = None,
    ) -> tuple[EmbeddingModelRecord, ...]:
        """Return registered models, optionally filtered by name, index_type, and/or provider_type."""
        return self._embedding_model_registry.get_registered_models_from_db(
            backend_type=self.backend_type,
            model_name=model_name,
            index_type=index_type,
            provider_type=provider_type,
        )

    # ------------------------------------------------------------------
    # Embedding table cache (internal)
    # ------------------------------------------------------------------

    def get_embedding_table(
        self,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
    ) -> Type[T]:
        """Return the dynamically created ORM class for the model's embedding table.

        Requires ``initialise_store`` to have been called first. Raises
        ``ValueError`` on a cache miss — this indicates a programming error
        (e.g. store not initialised, model not registered) rather than bad input.
        """
        storage_key = self._get_embedding_table_cache_key(
            model_name=model_name,
            provider_type=provider_type,
            index_type=index_type,
        )
        embedding_table = self._embedding_table_cache.get(storage_key)
        if embedding_table is not None:
            return embedding_table
        raise ValueError(
            f"Embedding table for model '{model_name}' with index type '{index_type.value}' "
            f"and backend '{self.backend_type.value}' not found in cache. "
            "Ensure that the store is initialized and the model is registered "
            "before attempting to access the embedding table."
        )

    def _cache_model_record(self, engine: Engine, model_record: EmbeddingModelRecord) -> None:
        dynamic_table = self._create_storage_table(engine=engine, model_record=model_record)
        storage_key = self._get_embedding_table_cache_key(
            model_name=model_record.model_name,
            provider_type=model_record.provider_type,
            index_type=model_record.index_type,
        )
        self._embedding_table_cache[storage_key] = dynamic_table

    def _get_embedding_table_cache_key(
        self,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
    ) -> Tuple[str, ProviderType, BackendType, IndexType]:
        return (model_name, provider_type, self.backend_type, index_type)

    @abstractmethod
    def _create_storage_table(self, engine: Engine, model_record: EmbeddingModelRecord) -> Type[T]:
        """Backend-specific logic to create the dynamic SQLAlchemy ORM class."""

    @abstractmethod
    def _delete_storage_table(self, engine: Engine, model_record: EmbeddingModelRecord) -> None:
        """Backend-specific logic to drop the dynamic table and any associated resources."""

    # ------------------------------------------------------------------
    # Core read/write operations
    # ------------------------------------------------------------------

    def _get_existing_concept_ids(
        self,
        session: Session,
        concept_ids: tuple[int, ...],
        table: Type[T],
    ) -> set[int]:
        """Return the subset of *concept_ids* already present in *table*.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session bound to the OMOP CDM database.
        concept_ids : tuple[int, ...]
            Candidate concept IDs to check.
        table : Type[T]
            ORM class for the backend's concept-ID registry table (subclass of
            ``ConceptIDEmbeddingBase``).

        Returns
        -------
        set[int]
            Already-stored concept IDs; empty set when none exist.

        Notes
        -----
        Uses an indexed PK lookup — O(batch_size). Call before any INSERT to
        get a clear ``ValueError`` instead of a transaction-aborting
        ``IntegrityError`` that rolls back the entire batch.
        """
        if not concept_ids:
            return set()
        result = session.execute(
            select(table.concept_id).where(table.concept_id.in_(concept_ids))
        )
        return {row[0] for row in result}

    @abstractmethod
    @require_registered_model
    def upsert_embeddings(
        self,
        *,
        session: Session,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        concept_ids: Sequence[int],
        embeddings: ndarray,
        _model_record: EmbeddingModelRecord,
        metric_type: Optional[MetricType] = None,
    ) -> None:
        """Insert or update vector embeddings for a collection of OMOP concept IDs.

        Parameters
        ----------
        session : Session
            SQLAlchemy session bound to the OMOP CDM database.
        model_name : str
            Registered name of the embedding model.
        provider_type : ProviderType
            Provider type for the embedding model.
        index_type : IndexType
            Storage index type used for this model's embeddings.
        concept_ids : Sequence[int]
            Concept IDs aligned with the rows of ``embeddings``.
        embeddings : ndarray
            Embedding matrix of shape ``(n_concepts, D)``.
        _model_record : EmbeddingModelRecord
            Injected by ``@require_registered_model``.
        metric_type : MetricType, optional
            When provided the backend should also update its nearest-neighbor
            index for this metric (FAISS only; ignored by pgvector).
        """

    def bulk_upsert_embeddings(
        self,
        *,
        session: Session,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        batches: Iterable[Tuple[Sequence[int], ndarray]],
        metric_type: Optional[MetricType] = None,
    ) -> None:
        """Upsert embeddings from a lazy iterable of ``(concept_ids, embeddings)`` batches.

        Parameters
        ----------
        session : Session
            Active SQLAlchemy session bound to the OMOP CDM database.
        model_name : str
            Registered canonical name of the embedding model.
        provider_type : ProviderType
            Provider that produced the embeddings.
        index_type : IndexType
            Index type the model was registered with.
        batches : Iterable[Tuple[Sequence[int], ndarray]]
            Lazy iterable of ``(concept_ids, embeddings)`` pairs. ``embeddings``
            must be float32 of shape ``(batch_size, D)``, rows aligned to
            ``concept_ids``. Wrap with ``tqdm`` for a progress bar.
        metric_type : MetricType, optional
            When provided the backend updates its nearest-neighbour index for
            this metric. Ignored by pgvector.

        Notes
        -----
        Default implementation calls ``upsert_embeddings`` once per batch.
        Backends may override for an optimised bulk path.
        """
        for concept_ids, embeddings in batches:
            self.upsert_embeddings(
                session=session,
                model_name=model_name,
                provider_type=provider_type,
                index_type=index_type,
                concept_ids=concept_ids,
                embeddings=embeddings,
                metric_type=metric_type,
            )
        

    @abstractmethod
    @require_registered_model
    def get_embeddings_by_concept_ids(
        self,
        *,
        session: Session,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        concept_ids: Sequence[int],
        _model_record: EmbeddingModelRecord,
    ) -> Mapping[int, Sequence[float]]:
        """Fetch stored embedding vectors keyed by concept ID.

        Returns
        -------
        Mapping[int, Sequence[float]]
            Dictionary mapping concept IDs to their corresponding embedding vectors.
        """

    @abstractmethod
    @require_registered_model
    def get_nearest_concepts(
        self,
        *,
        session: Session,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        query_embeddings: ndarray,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        _model_record: EmbeddingModelRecord,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        """Return nearest stored concepts for each query embedding.

        Parameters
        ----------
        query_embeddings : ndarray
            Query matrix of shape ``(Q, D)``.
        metric_type : MetricType
            Similarity or distance metric for nearest-neighbor search.
        concept_filter : EmbeddingConceptFilter, optional
            Restricts which concepts are considered. The ``limit`` field controls
            how many neighbors are returned per query; defaults to ``DEFAULT_K_NEAREST``.

        Returns
        -------
        Tuple[Tuple[NearestConceptMatch, ...], ...]
            Shape ``(Q, k)`` — one inner tuple of matches per query vector.
        """

    # ------------------------------------------------------------------
    # Concept queries
    # ------------------------------------------------------------------

    def has_any_embeddings(
        self,
        session: Session,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
    ) -> bool:
        embedding_table = self.get_embedding_table(
            model_name=model_name,
            index_type=index_type,
            provider_type=provider_type,
        )
        return session.query(embedding_table.concept_id).limit(1).first() is not None

    def get_concepts_without_embedding(
        self,
        session: Session,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: Optional[int] = None,
    ) -> Mapping[int, str]:
        """Return concept IDs and names for concepts that do not yet have embeddings."""
        query = self.q_get_concepts_without_embedding(
            model_name=model_name,
            provider_type=provider_type,
            index_type=index_type,
            concept_filter=concept_filter,
            limit=limit,
        )
        return {row.concept_id: row.concept_name for row in session.execute(query)}

    def q_get_concepts_without_embedding(
        self,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: Optional[int] = None,
    ) -> Select:
        embedding_table = self.get_embedding_table(
            model_name=model_name, index_type=index_type, provider_type=provider_type
        )
        subq = select(1).where(embedding_table.concept_id == Concept.concept_id)
        query = select(Concept.concept_id, Concept.concept_name).where(~subq.exists())
        if concept_filter is not None:
            query = concept_filter.apply(query)
        return query.limit(limit)

    def get_concepts_without_embedding_count(
        self,
        session: Session,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> int:
        embedding_table = self.get_embedding_table(
            model_name=model_name, index_type=index_type, provider_type=provider_type
        )
        subq = select(1).where(embedding_table.concept_id == Concept.concept_id)
        query = select(func.count()).select_from(Concept).where(~subq.exists())
        if concept_filter is not None:
            query = concept_filter.apply(query)
        return session.scalar(query)  # type: ignore

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_nearest_concepts_output(
        self,
        nearest_concepts: Tuple[Tuple[NearestConceptMatch, ...], ...],
        k: int,
        query_embeddings: ndarray,
    ) -> None:
        if not all(len(d) <= k for d in nearest_concepts):
            max_neighbors = max(len(d) for d in nearest_concepts)
            raise RuntimeError(
                f"Expected at most {k} nearest neighbors per query embedding, "
                f"but found {max_neighbors} entries."
            )
        if len(nearest_concepts) != query_embeddings.shape[0]:
            raise RuntimeError(
                f"Expected nearest concepts for {query_embeddings.shape[0]} query embeddings, "
                f"but got {len(nearest_concepts)}."
            )

    @staticmethod
    def validate_embeddings(embeddings: ndarray, dimensions: int) -> None:
        """Validate shape and dimensionality of an embeddings array."""
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D array of embeddings, got ndim={embeddings.ndim}.")
        if embeddings.shape[1] != dimensions:
            raise ValueError(
                f"Embedding dimensionality ({embeddings.shape[1]}) does not match "
                f"model configuration ({dimensions})."
            )

    @staticmethod
    def validate_embeddings_and_concept_ids(
        embeddings: ndarray,
        concept_ids: Union[Sequence[int], ndarray],
        dimensions: int,
    ) -> None:
        """Validate that embeddings and concept IDs match in count and dimensions."""
        EmbeddingBackend.validate_embeddings(embeddings, dimensions=dimensions)
        if len(concept_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Number of concept IDs ({len(concept_ids)}) does not match "
                f"number of embeddings ({embeddings.shape[0]})."
            )

    @staticmethod
    def _validate_external_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Expected metadata to be a mapping type, got {type(metadata)}")
        reserved_keys = {INDEX_CONFIG_METADATA_KEY}
        if any(key in reserved_keys for key in metadata):
            raise ValueError(f"Metadata contains reserved keys: {reserved_keys & metadata.keys()}")
        return metadata
