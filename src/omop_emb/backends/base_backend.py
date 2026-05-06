from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
import logging
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

from numpy import ndarray
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from omop_emb.config import (
    BackendType,
    MetricType,
    ProviderType,
    IndexType,
    is_supported_index_metric_combination_for_backend,
)
from omop_emb.backends.index_config import IndexConfig, FlatIndexConfig
from omop_emb.model_registry import EmbeddingModelRecord, RegistryManager
from omop_emb.utils.embedding_utils import (
    ConceptEmbeddingRecord,
    EmbeddingConceptFilter,
    NearestConceptMatch,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def require_registered_model(func: Callable) -> Callable:
    """Resolve and validate a registry record before the wrapped method runs.

    Looks up the model by ``(model_name, provider_type)`` using the backend's
    own ``backend_type``, then validates the caller-supplied ``metric_type``
    against the registry:

    - FLAT index (registry ``metric_type`` is ``None``): any metric supported
      by the backend is accepted.
    - HNSW index (registry ``metric_type`` is set): the caller must supply
      exactly that metric.

    The resolved record is injected as ``_model_record`` into the wrapped
    function's keyword arguments.

    Parameters
    ----------
    func : Callable
        Backend method that accepts ``model_name``, ``provider_type``,
        ``metric_type``, and ``_model_record`` as keyword arguments.

    Returns
    -------
    Callable
        Wrapped function with registry lookup and metric validation.

    Raises
    ------
    ValueError
        If the model is not registered, the caller-supplied metric is
        incompatible with the registered index, or the metric is not
        supported by the backend.
    """
    @wraps(func)
    def wrapper(
        self: "EmbeddingBackend",
        *,
        model_name: str,
        provider_type: ProviderType,
        metric_type: MetricType,
        **kwargs: Any,
    ) -> Any:
        record = self.get_registered_model(
            model_name=model_name, 
            provider_type=provider_type
        )
        if record is None:
            raise ValueError(
                f"Embedding model '{model_name}' (provider='{provider_type.value}') "
                f"is not registered in backend '{self.backend_type.value}'."
            )

        registry_metric = record.metric_type
        registry_index = record.index_type
        if registry_metric is not None:
            # Non-FLAT index: caller must supply the exact locked metric.
            if record.index_type == IndexType.FLAT:
                raise ValueError(
                    f"Model '{model_name}' is registered with a metric "
                    f"('{registry_metric.value}'), which indicates a non-FLAT index. This model is registered with {registry_index.value}, indicating a faulty registry state."
                )
            if metric_type != registry_metric:
                raise ValueError(
                    f"Model '{model_name}' is indexed with metric "
                    f"'{registry_metric.value}' but caller requested "
                    f"'{metric_type.value}'. Rebuild the index with the desired "
                    "metric or query with the registered one."
                )
        else:
            # FLAT index: any backend-supported metric is valid.
            if record.index_type != IndexType.FLAT:
                raise ValueError(
                    f"Model '{model_name}' is registered without any metric, which indicates a FLAT index. This model is registered with {registry_index.value}, indicating a faulty registry state."
                )
            if not is_supported_index_metric_combination_for_backend(
                backend=self.backend_type,
                index=record.index_type,
                metric=metric_type,
            ):
                raise ValueError(
                    f"Metric '{metric_type.value}' is not supported by backend "
                    f"'{self.backend_type.value}' with index type "
                    f"'{record.index_type.value}'."
                )

        kwargs["_model_record"] = record
        return func(
            self,
            model_name=model_name,
            provider_type=provider_type,
            metric_type=metric_type,
            **kwargs,
        )

    return wrapper


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------

class EmbeddingBackend(ABC):
    """Abstract base class for embedding storage and retrieval backends.

    Parameters
    ----------
    emb_engine : Engine
        SQLAlchemy engine pointing at the embedding store. For sqlite-vec this
        is the same .db file as the registry; for pgvector it is the same
        Postgres database.

    Properties
    ----------
    backend_type : BackendType
        Backend type identifier (e.g. ``BackendType.PGVECTOR``).
    backend_name : str
        String value of ``backend_type`` (e.g. ``"pgvector"``).
    emb_engine : Engine
        SQLAlchemy engine for the embedding store. Obtained through the registry manager.
    emb_session_factory : sessionmaker
        Session factory bound to ``emb_engine``. Obtained through the registry manager.

    Notes
    -----
    The SQLAlchemy engine is obtained from the RegistryManager to prevent 
    duplication in code but also to have a single source of truth for the database connection.  
    """

    DEFAULT_K_NEAREST = 10

    def __init__(self, emb_engine: Engine) -> None:
        super().__init__()
        self._registry = RegistryManager(emb_engine)
        self._table_cache: dict[str, Any] = {}
        self._initialise_store()

    # ------------------------------------------------------------------
    # Backend identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def backend_type(self) -> BackendType: ...

    @property
    def backend_name(self) -> str:
        """String value of ``backend_type``."""
        return self.backend_type.value

    @property
    def emb_engine(self) -> Engine:
        """SQLAlchemy engine for the embedding store."""
        return self._registry.embedding_engine

    @property
    def emb_session_factory(self) -> sessionmaker:
        """Session factory bound to ``emb_engine``."""
        return self._registry.emb_session_factory

    # ------------------------------------------------------------------
    # Store lifecycle
    # ------------------------------------------------------------------

    def _initialise_store(self) -> None:
        self.pre_initialise_store()
        for record in self._registry.get_registered_models(backend_type=self.backend_type):
            self._ensure_storage_table(record)

    def pre_initialise_store(self) -> None:
        """Hook for backend-specific setup before the registry is queried.

        Override to run DDL such as ``CREATE EXTENSION`` before tables are
        created. The default implementation is a no-op.
        """

    def _ensure_storage_table(self, model_record: EmbeddingModelRecord) -> Any:
        if model_record.storage_identifier not in self._table_cache:
            table = self._create_storage_table(model_record)
            self._table_cache[model_record.storage_identifier] = table
        return self._table_cache[model_record.storage_identifier]

    # ------------------------------------------------------------------
    # Model registration / deletion / index management
    # ------------------------------------------------------------------

    def register_model(
        self,
        *,
        model_name: str,
        dimensions: int,
        provider_type: ProviderType,
        index_config: Optional[IndexConfig] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> EmbeddingModelRecord:
        """Register a model and create its physical storage table.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        dimensions : int
            Embedding vector dimensionality.
        provider_type : ProviderType
            Provider that serves the model.
        index_config : IndexConfig, optional
            Index configuration. Defaults to ``FlatIndexConfig()`` when not
            provided.
        metadata : Mapping[str, object], optional
            Free-form operational metadata. Must not contain reserved keys
            (``"index_config"``, ``"faiss_cache"``).

        Returns
        -------
        EmbeddingModelRecord
            The new or existing registry record.

        Raises
        ------
        ModelRegistrationConflictError
            If the model is already registered with a different dimensionality.
        ValueError
            If ``metadata`` contains a reserved key.
        """

        if index_config is None:
            index_config = FlatIndexConfig()

        if index_config != FlatIndexConfig():
            # non-flat index is super expensive to continuously ingest so we don't 
            # allow it at the moment
            raise ValueError(
                "Only FLAT index is allowed at registration as it is expensive to continously inject it.\n"
                "To use a non-FLAT index, register the model first, ingest the data and then build the index.\n"
                "See the CLI documentation for details."
            )

        record = self._registry.register_model(
            model_name=model_name,
            provider_type=provider_type,
            backend_type=self.backend_type,
            dimensions=dimensions,
            index_config=index_config,
            metadata=metadata,
        )
        self._ensure_storage_table(record)
        # Disable for now as we prevent non-FLAT index registration
        #self._rebuild_index_impl(model_record=record, index_config=index_config)
        logger.info(
            f"Registered model '{model_name}' (provider='{provider_type.value}') in backend '{self.backend_type.value}'."
        )
        return record

    def delete_model(
        self,
        *,
        model_name: str,
        provider_type: ProviderType,
    ) -> None:
        """Delete the registry row and drop the physical embedding table.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        provider_type : ProviderType
            Provider that serves the model.
        Raises
        ------
        ValueError
            If the model is not registered.

        Notes
        -----
        One row per model means deletion always drops the physical table.
        There is no shared-table check.
        """
        record = self.get_registered_model(model_name, provider_type)
        if record is None:
            raise ValueError(
                f"Model '{model_name}' (provider='{provider_type.value}') "
                f"is not registered in backend '{self.backend_type.value}'."
            )
        self._registry.delete_model(
            model_name=model_name,
            provider_type=provider_type,
            backend_type=self.backend_type,
        )
        self._table_cache.pop(record.storage_identifier, None)
        self._delete_storage_table(model_record=record)
        logger.info(
            f"Deleted model '{model_name}' (provider='{provider_type.value}') from backend '{self.backend_type.value}' and dropped storage table."
        )
    
    def rebuild_index(
        self,
        *,
        model_name: str,
        provider_type: ProviderType,
        index_config: IndexConfig,
    ) -> EmbeddingModelRecord:
        """Build or rebuild the index on an existing embedding table.

        This is the unified entry point for both rebuilding an existing index
        and switching index type (e.g. FLAT → HNSW or HNSW → FLAT). Pass the
        desired target ``index_config`` regardless of the current state.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        provider_type : ProviderType
            Provider that serves the model.
        index_config : IndexConfig
            Target index configuration. Pass ``HNSWIndexConfig`` (with
            ``metric_type`` set) to build an HNSW index, or ``FlatIndexConfig()``
            to revert to an exact scan.

        Returns
        -------
        EmbeddingModelRecord
            Updated registry record reflecting the new index type and metric.

        Raises
        ------
        ValueError
            If the model is not registered.

        Notes
        -----
        The registry ``metric_type`` is updated to match the new ``index_config``.
        Any in-flight queries that passed the old metric validation may fail on
        their next call — this is an administrative operation and the error
        message will be clear.

        ``require_registered_model`` is not applied here because this method
        intentionally modifies the state that the decorator validates against.
        """
        record = self.get_registered_model(model_name, provider_type)
        if record is None:
            raise ValueError(
                f"Model '{model_name}' (provider='{provider_type.value}') "
                f"is not registered in backend '{self.backend_type.value}'."
            )
        self._rebuild_index_impl(model_record=record, index_config=index_config)
        return self._registry.update_index_config(
            model_name=model_name,
            provider_type=provider_type,
            backend_type=self.backend_type,
            index_config=index_config,
        )

    @abstractmethod
    def _rebuild_index_impl(
        self, *, model_record: EmbeddingModelRecord, index_config: IndexConfig
    ) -> None: ...
    
    # ------------------------------------------------------------------
    # Registry queries
    # ------------------------------------------------------------------

    def get_registered_model(
        self,
        model_name: str,
        provider_type: ProviderType,
    ) -> Optional[EmbeddingModelRecord]:
        """Convenience method to return a single registry record for the given model and provider.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        provider_type : ProviderType
            Provider that serves the model.
        Returns
        -------
        EmbeddingModelRecord or None
        """
        registered_models = self.get_registered_models(
            model_name=model_name,
            provider_type=provider_type,
        )
        if len(registered_models) > 1:
            raise ValueError(
                f"Multiple records found for model '{model_name}' (provider='{provider_type.value}') "
                f"in backend '{self.backend_type.value}'. This should not happen as the registry is designed to have one record per model-provider-backend combination. Found {len(registered_models)} records."
            )
        return registered_models[0] if registered_models else None
    
    def get_registered_models(
        self,
        *,
        model_name: Optional[str] = None,
        provider_type: Optional[ProviderType] = None,
    ) -> tuple[EmbeddingModelRecord, ...]:
        """Return all registry entries for this backend, with optional filters.

        Parameters
        ----------
        model_name : str, optional
            Canonical model name to filter by.
        provider_type : ProviderType, optional
            Provider that serves the model.

        Returns
        -------
        tuple[EmbeddingModelRecord, ...]
        """
        return self._registry.get_registered_models(
            backend_type=self.backend_type,
            model_name=model_name,
            provider_type=provider_type,
        )

    def is_model_registered(
        self,
        *,
        model_name: str,
        provider_type: ProviderType,
    ) -> bool:
        """Return ``True`` if the model is present in the registry.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        provider_type : ProviderType
            Provider that serves the model.
        Returns
        -------
        bool
        """
        return self.get_registered_model(model_name, provider_type) is not None

    def patch_model_metadata(
        self,
        *,
        model_name: str,
        provider_type: ProviderType,
        key: str,
        value: object,
    ) -> None:
        """Merge a single key-value pair into the registry metadata.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        provider_type : ProviderType
            Provider that serves the model.
        key : str
            Metadata key to set or overwrite. Must not be a reserved key.
        value : object
            JSON-serialisable value.

        Raises
        ------
        ValueError
            If the model is not registered or ``key`` is a reserved metadata
            key.
        """
        record = self.get_registered_model(model_name, provider_type)
        if record is None:
            raise ValueError(
                f"Model '{model_name}' (provider='{provider_type.value}') "
                f"is not registered in backend '{self.backend_type.value}'."
            )
        updated = {**record.metadata, key: value}
        self._registry.update_metadata(
            model_name=model_name,
            provider_type=provider_type,
            backend_type=self.backend_type,
            metadata=updated,
        )

    # ------------------------------------------------------------------
    # Storage table management (backend-specific)
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_storage_table(self, model_record: EmbeddingModelRecord) -> Any: ...

    @abstractmethod
    def _delete_storage_table(self, model_record: EmbeddingModelRecord) -> None: ...

    # ------------------------------------------------------------------
    # Core write operations
    # ------------------------------------------------------------------

    @require_registered_model
    def upsert_embeddings(
        self,
        *,
        model_name: str,
        provider_type: ProviderType,
        metric_type: MetricType,
        records: Sequence[ConceptEmbeddingRecord],
        embeddings: ndarray,
        _model_record: EmbeddingModelRecord,
    ) -> None:
        """Insert or update embeddings for a set of concepts.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        provider_type : ProviderType
            Provider that serves the model.
        metric_type : MetricType
            Metric to validate against the registry. For FLAT any
            backend-supported metric is accepted; for HNSW it must match the
            registered metric.
        records : Sequence[ConceptEmbeddingRecord]
            Concept metadata rows aligned with ``embeddings``.
        embeddings : ndarray
            Float32 array of shape ``(N, D)`` where ``N = len(records)`` and
            ``D`` is the registered dimensionality.
        """
        return self._upsert_embeddings_impl(
            model_record=_model_record,
            records=records,
            embeddings=embeddings,
        )

    @abstractmethod
    def _upsert_embeddings_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        records: Sequence[ConceptEmbeddingRecord],
        embeddings: ndarray,
    ) -> None: ...

    def bulk_upsert_embeddings(
        self,
        *,
        model_name: str,
        provider_type: ProviderType,
        metric_type: MetricType,
        batches: Iterable[Tuple[Sequence[ConceptEmbeddingRecord], ndarray]],
    ) -> None:
        """Upsert embeddings in multiple batches, delegating to ``upsert_embeddings``.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        provider_type : ProviderType
            Provider that serves the model.
        metric_type : MetricType
            Validated once per batch via ``upsert_embeddings``.
        batches : Iterable[tuple[Sequence[ConceptEmbeddingRecord], ndarray]]
            Iterable of ``(records, embeddings)`` pairs.
        """
        for records, embeddings in batches:
            self.upsert_embeddings(
                model_name=model_name,
                provider_type=provider_type,
                metric_type=metric_type,
                records=records,
                embeddings=embeddings,
            )

    # ------------------------------------------------------------------
    # Core read operations
    # ------------------------------------------------------------------

    @require_registered_model
    def get_embeddings_by_concept_ids(
        self,
        *,
        model_name: str,
        provider_type: ProviderType,
        metric_type: MetricType,
        concept_ids: Sequence[int],
        _model_record: EmbeddingModelRecord,
    ) -> Mapping[int, Sequence[float]]:
        """Retrieve stored embeddings for the given concept IDs.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        provider_type : ProviderType
            Provider that serves the model.
        metric_type : MetricType
            Validated against the registry.
        concept_ids : Sequence[int]
            OMOP concept IDs to look up.

        Returns
        -------
        Mapping[int, Sequence[float]]
            Mapping from concept ID to embedding vector.

        Raises
        ------
        ValueError
            If any requested concept ID is not found in the table.
        """
        return self._get_embeddings_by_concept_ids_impl(
            model_record=_model_record,
            concept_ids=concept_ids,
        )

    @abstractmethod
    def _get_embeddings_by_concept_ids_impl(
        self,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
    ) -> Mapping[int, Sequence[float]]: ...

    @require_registered_model
    def get_nearest_concepts(
        self,
        *,
        model_name: str,
        provider_type: ProviderType,
        metric_type: MetricType,
        query_embeddings: ndarray,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        k: int = DEFAULT_K_NEAREST,
        _model_record: EmbeddingModelRecord,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        """Find the nearest stored concepts for one or more query vectors.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        provider_type : ProviderType
            Provider that serves the model.
        metric_type : MetricType
            Distance metric for the KNN query. Validated against the registry:
            must match the HNSW metric exactly, or be any backend-supported
            metric for FLAT.
        query_embeddings : ndarray
            Float32 array of shape ``(Q, D)`` where ``Q`` is the number of
            queries and ``D`` is the registered dimensionality.
        concept_filter : EmbeddingConceptFilter, optional
            Constraints applied during retrieval (domain, vocabulary, standard
            flag, concept ID allowlist, result limit).
        k : int
            Maximum number of results per query. Default ``10``.

        Returns
        -------
        tuple[tuple[NearestConceptMatch, ...], ...]
            Shape ``(Q, <=K)``. Outer tuple is one entry per query vector;
            inner tuple contains up to ``k`` matches ordered by similarity
            descending.
        """
        return self._get_nearest_concepts_impl(
            model_record=_model_record,
            metric_type=metric_type,
            query_embeddings=query_embeddings,
            concept_filter=concept_filter,
            k=k,
        )

    @abstractmethod
    def _get_nearest_concepts_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        metric_type: MetricType,
        query_embeddings: ndarray,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        k: int = DEFAULT_K_NEAREST,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]: ...

    # ------------------------------------------------------------------
    # Utility / diagnostic queries
    # ------------------------------------------------------------------

    @require_registered_model
    def has_any_embeddings(
        self,
        *,
        model_name: str,
        provider_type: ProviderType,
        metric_type: MetricType,
        _model_record: EmbeddingModelRecord,
    ) -> bool:
        """Return ``True`` if at least one embedding row exists in the table.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        provider_type : ProviderType
            Provider that serves the model.
        metric_type : MetricType
            Validated against the registry.

        Returns
        -------
        bool
        """
        return self._has_any_embeddings_impl(model_record=_model_record)

    @abstractmethod
    def _has_any_embeddings_impl(self, *, model_record: EmbeddingModelRecord) -> bool: ...

    @require_registered_model
    def get_all_stored_concept_ids(
        self,
        *,
        model_name: str,
        provider_type: ProviderType,
        metric_type: MetricType,
        _model_record: EmbeddingModelRecord,
    ) -> set[int]:
        """Return all concept IDs stored in the embedding table.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        provider_type : ProviderType
            Provider that serves the model.
        metric_type : MetricType
            Validated against the registry.

        Returns
        -------
        set[int]
        """
        return self._get_all_stored_concept_ids_impl(model_record=_model_record)

    @abstractmethod
    def _get_all_stored_concept_ids_impl(self, *, model_record: EmbeddingModelRecord) -> set[int]: ...

    def get_embedding_count(
        self,
        *,
        model_name: str,
        provider_type: ProviderType,
        metric_type: MetricType,
    ) -> int:
        """Return the number of embeddings stored in the table.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        provider_type : ProviderType
            Provider that serves the model.
        metric_type : MetricType

        Returns
        -------
        int
        """
        return len(self.get_all_stored_concept_ids(
            model_name=model_name,
            provider_type=provider_type,
            metric_type=metric_type,
        ))

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def validate_embeddings(embeddings: ndarray, dimensions: int) -> None:
        """Validate that ``embeddings`` has the expected shape.

        Parameters
        ----------
        embeddings : ndarray
            Array to validate.
        dimensions : int
            Expected number of columns (embedding dimensionality).

        Raises
        ------
        ValueError
            If ``embeddings`` is not 2-D or its column count does not match
            ``dimensions``.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D array, got ndim={embeddings.ndim}.")
        if embeddings.shape[1] != dimensions:
            raise ValueError(
                f"Embedding dimensionality ({embeddings.shape[1]}) does not match "
                f"model configuration ({dimensions})."
            )

    @staticmethod
    def validate_embeddings_and_records(
        embeddings: ndarray,
        records: Sequence[ConceptEmbeddingRecord],
        dimensions: int,
    ) -> None:
        """Validate that ``embeddings`` and ``records`` are aligned and well-shaped.

        Parameters
        ----------
        embeddings : ndarray
            Array of shape ``(N, D)``.
        records : Sequence[ConceptEmbeddingRecord]
            Concept metadata rows. Must have ``len(records) == N``.
        dimensions : int
            Expected embedding dimensionality ``D``.

        Raises
        ------
        ValueError
            If ``embeddings`` fails shape validation or its row count does not
            match ``len(records)``.
        """
        EmbeddingBackend.validate_embeddings(embeddings, dimensions=dimensions)
        if len(records) != embeddings.shape[0]:
            raise ValueError(
                f"Number of records ({len(records)}) does not match "
                f"number of embeddings ({embeddings.shape[0]})."
            )
