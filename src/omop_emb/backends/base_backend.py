from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import (
    Any,
    Concatenate,
    Protocol,
    Self,
)

from numpy import ndarray
from oa_configurator import (
    Dialect,
    ResolvedDatabase,
    ResolvedVectorStore,
    physical_schema_of,
    qualified,
)
from sqlalchemy import Engine, inspect, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from omop_emb.backends.embedding_table import ConceptEmbeddingRecord
from omop_emb.backends.index_config import FlatIndexConfig, IndexConfig
from omop_emb.config import (
    BackendType,
    IndexType,
    MetricType,
    get_supported_index_types_for_backend,
    is_index_type_supported_for_backend,
    is_supported_index_metric_combination_for_backend,
    parse_backend_type,
)
from omop_emb.model_registry import (
    EmbeddingModelRecord,
    RegistryManager,
    registry_reader_engine,
    registry_writer_engine,
)
from omop_emb.utils.cdm import streamed
from omop_emb.utils.embedding_utils import (
    EmbeddingArray,
    EmbeddingConceptFilter,
    NearestConceptMatch,
    as_embedding_array,
)
from omop_emb.utils.errors import (
    EmbeddingBackendConfigurationError,
    EmbeddingBackendDependencyError,
    MissingStorageTableError,
    ReadOnlyStoreError,
    UnknownEmbeddingBackendError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Write marker
# ---------------------------------------------------------------------------


def writes[B: EmbeddingBackend, **P, R](method: Callable[Concatenate[B, P], R]) -> Callable[Concatenate[B, P], R]:
    """Mark an EmbeddingBackend method as a write: raises ReadOnlyStoreError on a reader."""

    @wraps(method)
    def wrapper(self: B, *args: P.args, **kwargs: P.kwargs) -> R:
        if not self._writable:
            raise ReadOnlyStoreError(
                f"This '{self.backend_name}' store was opened with open_vector_store_reader(); "
                "use open_vector_store_writer() to write."
            )
        return method(self, *args, **kwargs)

    wrapper.__omop_emb_writes__ = True  # ty: ignore[unresolved-attribute]
    return wrapper


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompactionResult:
    """Storage size around a compaction, in bytes.

    Attributes
    ----------
    size_before : int
        Size before compaction: the database file on sqlite-vec, the model
        table with its indexes on pgvector.
    size_after : int
        The same measure after compaction.
    """

    size_before: int
    size_after: int

    @property
    def reclaimed(self) -> int:
        return max(self.size_before - self.size_after, 0)


class EmbeddingStoreReader(Protocol):
    """Read-only view of an embedding store, as returned by open_vector_store_reader()."""

    @property
    def backend_type(self) -> BackendType: ...

    @property
    def initialized(self) -> bool: ...

    def get_registered_model(self, *, model_name: str) -> EmbeddingModelRecord | None: ...

    def get_registered_models(
        self, *, model_name: str | None = None, provider_type: str | None = None
    ) -> tuple[EmbeddingModelRecord, ...]: ...

    def iter_stored_embeddings(
        self, model_name: str, *, batch_size: int = 10_000
    ) -> Iterator[ConceptEmbeddingRecord]: ...

    def has_any_embeddings(self, *, model_name: str) -> bool: ...

    def get_stored_concept_ids(
        self, *, model_name: str, concept_filter: EmbeddingConceptFilter | None = None
    ) -> set[int]: ...

    def get_embeddings_by_concept_ids(
        self, *, model_name: str, concept_ids: Sequence[int]
    ) -> Mapping[int, ndarray]: ...

    def get_concept_filter_metadata(
        self, *, model_name: str, concept_ids: Sequence[int]
    ) -> Mapping[int, ConceptEmbeddingRecord]: ...

    def get_embedding_count(self, *, model_name: str) -> int: ...

    def get_embedding_count_by_vocabulary(self, *, model_name: str) -> Mapping[str, int]: ...

    def get_nearest_concepts(
        self,
        *,
        model_name: str,
        metric_type: MetricType,
        query_embeddings: EmbeddingArray,
        concept_filter: EmbeddingConceptFilter | None = None,
        k: int = ...,
    ) -> tuple[tuple[NearestConceptMatch, ...], ...]: ...

    def physical_indexes(self, model_name: str) -> tuple[str, ...]: ...

    def drop_index_sql(self, model_name: str) -> tuple[str, ...]: ...

    def close(self) -> None: ...

    def __enter__(self) -> EmbeddingStoreReader: ...  # noqa: PYI034

    def __exit__(self, *_exc_info: object) -> None: ...


class EmbeddingBackend[TEmbeddingTable](EmbeddingStoreReader, ABC):
    """Abstract base class for embedding storage and retrieval backends; implements EmbeddingStoreReader.

    Parameters
    ----------
    emb_engine : Engine
        SQLAlchemy engine pointing at the embedding store. For sqlite-vec this
        is the same .db file as the registry; for pgvector it is the same
        Postgres database.
    resolved : ResolvedDatabase, optional
        Enables the schema-provenance guard around storage-table DDL.
    writable : bool, optional
        False makes every ``@writes`` method raise ``ReadOnlyStoreError``.
        Set by ``open_vector_store_reader()``.

    Properties
    ----------
    backend_type : BackendType
        Backend type identifier (e.g. ``BackendType.PGVECTOR``).
    backend_name : str
        String value of ``backend_type`` (e.g. ``"pgvector"``).
    dialect : str
        SQL dialect this backend requires (e.g. ``"postgresql"``, ``"sqlite"``).
        Matched against ``emb_engine.dialect.name`` at construction time.
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
    _DELETE_CHUNK_SIZE = 100_000

    def __init__(
        self,
        emb_engine: Engine,
        *,
        writable: bool = True,
    ) -> None:
        actual_dialect = emb_engine.dialect.name
        if actual_dialect != self.dialect:
            raise ValueError(
                f"{type(self).__name__} requires a '{self.dialect}'-dialect engine, "
                f"got '{actual_dialect}'."
            )
        super().__init__()
        self._writable = writable
        self._registry = RegistryManager(emb_engine)
        self._table_cache: dict[str, TEmbeddingTable] = {}

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
    @abstractmethod
    def dialect(self) -> str:
        """SQL dialect this backend requires, matching ``emb_engine.dialect.name``."""
        ...

    @property
    @abstractmethod
    def default_flat_metric(self) -> MetricType | None:
        """Metric a FLAT model takes when none is given.

        ``None`` for backends that compute distances from the raw vectors, whose
        FLAT models carry no metric. A metric for backends that lock it on the
        table.
        """
        ...

    @property
    def emb_engine(self) -> Engine:
        """SQLAlchemy engine for the embedding store."""
        return self._registry.embedding_engine

    @property
    def emb_session_factory(self) -> sessionmaker:
        """Session factory bound to ``emb_engine``."""
        return self._registry.emb_session_factory

    @property
    def initialized(self) -> bool:
        """Whether the model registry table exists."""
        return self._registry.registry_available

    def close(self) -> None:
        """Dispose the underlying engine."""
        self.emb_engine.dispose()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def iter_stored_embeddings(
        self,
        model_name: str,
        *,
        batch_size: int = 10_000,
    ) -> Iterator[ConceptEmbeddingRecord]:
        """Stream concept metadata; yields nothing if the model is not registered."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        record = self.get_registered_model(model_name=model_name)
        if record is None:
            return
        table = self._storage_table(record)
        columns = inspect(table).columns  # ty: ignore[unresolved-attribute]
        statement = streamed(
            select(
                columns.concept_id,
                columns.domain_id,
                columns.vocabulary_id,
                columns.is_standard,
                columns.is_valid,
            ),
            batch_size,
        )
        with self.emb_engine.connect() as connection:
            rows = connection.execute(statement).mappings()
            for row in rows:
                yield ConceptEmbeddingRecord(
                    concept_id=int(row["concept_id"]),
                    domain_id=str(row["domain_id"]),
                    vocabulary_id=str(row["vocabulary_id"]),
                    is_standard=bool(row["is_standard"]),
                    is_valid=bool(row["is_valid"]),
                )

    @abstractmethod
    def physical_indexes(self, model_name: str) -> tuple[str, ...]:
        """Return existing physical indexes for this model's table, without
        creating or changing them."""
        ...

    def drop_index_sql(self, model_name: str) -> tuple[str, ...]:
        """Return reviewed index-removal statements without executing them."""
        physical_schema = physical_schema_of(self.emb_engine)
        return tuple(
            f"DROP INDEX IF EXISTS {qualified(self.emb_engine, name, physical_schema=physical_schema)};"
            for name in self.physical_indexes(model_name)
        )

    # ------------------------------------------------------------------
    # Store lifecycle
    # ------------------------------------------------------------------

    def _ensure_storage_table(self, model_record: EmbeddingModelRecord) -> TEmbeddingTable:
        """Create-if-new. Used by register_model's write path only."""
        if model_record.storage_identifier not in self._table_cache:
            table = self._create_storage_table(model_record)
            self._table_cache[model_record.storage_identifier] = table
        return self._table_cache[model_record.storage_identifier]

    def _storage_table(self, model_record: EmbeddingModelRecord) -> TEmbeddingTable:
        """Lazy lookup for an already-registered model. Raises
        MissingStorageTableError if genuinely absent -- never recreates it."""
        cached = self._table_cache.get(model_record.storage_identifier)
        if cached is not None:
            return cached
        if not self._storage_table_exists(model_record):
            raise MissingStorageTableError(
                f"Model '{model_record.model_name}' is registered with storage_identifier "
                f"'{model_record.storage_identifier}', but no such table exists in the "
                f"'{self.backend_name}' store."
            )
        table = self._get_storage_table_descriptor(model_record)
        self._table_cache[model_record.storage_identifier] = table
        return table

    # ------------------------------------------------------------------
    # Model registration / deletion / index management
    # ------------------------------------------------------------------

    @writes
    def register_model(
        self,
        *,
        model_name: str,
        dimensions: int,
        provider_type: str,
        index_config: IndexConfig | None = None,
        metadata: Mapping[str, object] | None = None,
        registered_at: datetime | None = None,
    ) -> EmbeddingModelRecord:
        """Register a model and create its physical storage table.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        dimensions : int
            Embedding vector dimensionality.
        provider_type : str
            omop-llm provider key that serves the model.
        index_config : IndexConfig, optional
            Index configuration. Defaults to ``FlatIndexConfig()`` when not
            provided.
        metadata : Mapping[str, object], optional
            Free-form operational metadata. Must not contain reserved keys
            (see ``RESERVED_METADATA_KEYS`` in ``index_config.py``).
        registered_at : datetime, optional
            Backdate ``created_at``/``updated_at`` to this timestamp instead
            of "now". Only applies to a brand-new registration.

        Returns
        -------
        EmbeddingModelRecord
            The new or existing registry record.

        Raises
        ------
        ModelRegistrationConflictError
            If the model is already registered with a different dimensionality.
        ValueError
            If ``metadata`` contains a reserved key, or if ``index_config`` is
            not a FLAT config (non-FLAT indexes may only be built after
            registration, not at registration time), or its metric is
            unsupported by the backend.
        """

        if index_config is None:
            index_config = FlatIndexConfig()

        if index_config.index_type != IndexType.FLAT:
            # non-flat index is super expensive to continuously ingest so we don't
            # allow it at the moment
            raise ValueError(
                "Only FLAT index is allowed at registration as it is expensive to continously inject it.\n"
                "To use a non-FLAT index, register the model first, ingest the data and then build the index.\n"
                "See the CLI documentation for details."
            )
        index_config = self._normalized_flat_metric(index_config)
        self._validate_index_config(index_config)

        preexisting = bool(self._registry.get_registered_models(model_name=model_name))
        record = self._registry.register_model(
            model_name=model_name,
            provider_type=provider_type,
            dimensions=dimensions,
            index_config=index_config,
            metadata=metadata,
            registered_at=registered_at,
        )
        try:
            self._ensure_storage_table(record)
        except Exception:
            if not preexisting:
                self._registry.delete_model(model_name=model_name)
            raise
        # Disable for now as we prevent non-FLAT index registration
        # self._rebuild_index_impl(model_record=record, index_config=index_config)
        logger.info(
            f"Registered model '{model_name}' (provider='{provider_type}') in backend '{self.backend_type.value}'."
        )
        return record

    @writes
    def delete_model(
        self,
        *,
        model_name: str,
    ) -> None:
        """Delete the registry row and drop the physical embedding table.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.

        Raises
        ------
        ValueError
            If the model is not registered.

        Notes
        -----
        One row per model means deletion always drops the physical table.
        There is no shared-table check.

        The physical table is dropped **before** the registry row is removed.
        If the DDL step fails the registry entry remains intact and the call
        is re-runnable. If the DDL succeeds but the registry delete fails the
        only failure mode is a registry entry pointing at a table that no
        longer exists -- the next read or write against that model raises
        ``MissingStorageTableError`` instead of silently recreating it.
        """
        record = self._registered_record(model_name)
        self._delete_storage_table(model_record=record)
        self._table_cache.pop(record.storage_identifier, None)
        self._registry.delete_model(model_name=model_name)
        logger.info(
            f"Deleted model '{model_name}' from backend '{self.backend_type.value}' and dropped storage table."
        )

    @writes
    def rebuild_index(
        self,
        *,
        model_name: str,
        index_config: IndexConfig,
        vacuum: bool = True,
    ) -> EmbeddingModelRecord:
        """Build or rebuild the index on an existing embedding table.

        This is the unified entry point for both rebuilding an existing index
        and switching index type (e.g. FLAT → HNSW or HNSW → FLAT). Pass the
        desired target ``index_config`` regardless of the current state.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        index_config : IndexConfig
            Target index configuration. Pass ``HNSWIndexConfig`` (with
            ``metric_type`` set) to build an HNSW index, or ``FlatIndexConfig()``
            to revert to an exact scan. On sqlite-vec a ``FlatIndexConfig``
            locks the model to its metric (cosine if none is given); pgvector
            FLAT models take the metric per query, so any metric is dropped.
        vacuum : bool
            Run the backend's maintenance step after the rebuild. pgvector
            runs ``VACUUM (ANALYZE)`` on the table to refresh planner
            statistics. sqlite-vec runs ``VACUUM`` on the database file after
            recreating a table for a metric change. Neither returns space
            freed by deleted rows, see ``compact_storage``. Default ``True``.

        Returns
        -------
        EmbeddingModelRecord
            Updated registry record reflecting the new index type and metric.

        Raises
        ------
        ValueError
            If the model is not registered, or the index type or metric is
            unsupported by the backend.

        Notes
        -----
        Afterwards the table carries exactly the index ``index_config`` describes:
        any other physical index on it is dropped. Queries then must use
        ``index_config.metric_type`` when it is set.
        """
        record = self._registered_record(model_name)
        index_config = self._normalized_flat_metric(index_config)
        self._validate_index_config(index_config)
        self._rebuild_index_impl(model_record=record, index_config=index_config, vacuum=vacuum)
        return self._registry.update_index_config(
            model_name=model_name,
            index_config=index_config,
        )

    def _normalized_flat_metric(self, index_config: IndexConfig) -> IndexConfig:
        """Return a FLAT config with the metric its backend calls for.

        Backends with a ``default_flat_metric`` lock the metric on the table,
        so their FLAT models keep the given metric or take that default. The
        others compute distances from the raw vectors, so their FLAT models
        carry no metric. Other index types are returned unchanged.
        """
        if index_config.index_type != IndexType.FLAT:
            return index_config
        if self.default_flat_metric is None:
            metric_type = None
        else:
            metric_type = index_config.metric_type or self.default_flat_metric
        if metric_type == index_config.metric_type:
            return index_config
        return FlatIndexConfig(metric_type=metric_type)

    def _validate_index_config(self, index_config: IndexConfig) -> None:
        """Raise ``ValueError`` if the backend supports neither the index type nor its metric."""
        if not is_index_type_supported_for_backend(
            backend=self.backend_type, index=index_config.index_type
        ):
            supported = get_supported_index_types_for_backend(self.backend_type)
            raise ValueError(
                f"Index type '{index_config.index_type.value}' is not supported by "
                f"backend '{self.backend_name}'. "
                f"Supported: {[idx.value for idx in supported]}."
            )
        if index_config.metric_type is not None and not is_supported_index_metric_combination_for_backend(
            backend=self.backend_type, index=index_config.index_type, metric=index_config.metric_type
        ):
            raise ValueError(
                f"Metric '{index_config.metric_type.value}' is not supported by backend "
                f"'{self.backend_name}' with index type '{index_config.index_type.value}'."
            )

    @abstractmethod
    def _rebuild_index_impl(
        self, *, model_record: EmbeddingModelRecord, index_config: IndexConfig, vacuum: bool
    ) -> None: ...

    # ------------------------------------------------------------------
    # Registry queries
    # ------------------------------------------------------------------

    def get_registered_model(
        self,
        *,
        model_name: str,
    ) -> EmbeddingModelRecord | None:
        """Return the registry record for *model_name*, or ``None`` if not registered.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.

        Returns
        -------
        EmbeddingModelRecord or None
        """
        registered_models = self.get_registered_models(model_name=model_name)
        return registered_models[0] if registered_models else None

    def get_registered_models(
        self,
        *,
        model_name: str | None = None,
        provider_type: str | None = None,
    ) -> tuple[EmbeddingModelRecord, ...]:
        """Return all registry entries for this backend, with optional filters.

        Parameters
        ----------
        model_name : str, optional
            Canonical model name to filter by.
        provider_type : str, optional
            omop-llm provider key that serves the model.

        Returns
        -------
        tuple[EmbeddingModelRecord, ...]
        """
        return self._registry.get_registered_models(
            model_name=model_name,
            provider_type=provider_type,
        )

    def _registered_record(self, model_name: str) -> EmbeddingModelRecord:
        """Return model_name's registry record.

        Raises
        ------
        ValueError
            If the model is not registered.
        """
        record = self.get_registered_model(model_name=model_name)
        if record is None:
            raise ValueError(
                f"Embedding model '{model_name}' is not registered in backend '{self.backend_name}'."
            )
        return record

    def _check_query_metric(self, record: EmbeddingModelRecord, metric_type: MetricType) -> None:
        """Check that a nearest-neighbour query may use metric_type on record's index.

        HNSW accepts only the metric its index was built with; FLAT accepts any
        metric the backend supports for FLAT.

        Raises
        ------
        ValueError
            If the metric is incompatible with the index or unsupported by the backend.
        """
        if record.metric_type is not None:
            if metric_type != record.metric_type:
                raise ValueError(
                    f"Model '{record.model_name}' is indexed with metric "
                    f"'{record.metric_type.value}' but caller requested "
                    f"'{metric_type.value}'. Rebuild the index with the desired "
                    "metric or query with the registered one."
                )
        elif not is_supported_index_metric_combination_for_backend(
            backend=self.backend_type, index=record.index_type, metric=metric_type
        ):
            raise ValueError(
                f"Metric '{metric_type.value}' is not supported by backend "
                f"'{self.backend_name}' with index type '{record.index_type.value}'."
            )

    @writes
    def patch_model_metadata(
        self,
        *,
        model_name: str,
        key: str,
        value: object,
    ) -> None:
        """Merge a single key-value pair into the registry metadata.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
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
        record = self._registered_record(model_name)
        updated = {**record.metadata, key: value}
        self._registry.update_metadata(model_name=model_name, metadata=updated)

    @writes
    def refresh_model_updated_at_timestamp(self, *, model_name: str) -> None:
        """Bump a registry row's ``updated_at`` to now.
        Required for faiss-cache freshness validation after live upserts.
        No-op if the model is not registered.

        Parameters
        ----------
        model_name : str
        """
        self._registry.refresh_model_updated_at_timestamp(model_name=model_name)

    # ------------------------------------------------------------------
    # Storage table management (backend-specific)
    # ------------------------------------------------------------------

    @abstractmethod
    def _storage_table_exists(self, model_record: EmbeddingModelRecord) -> bool:
        """Return ``True`` if the physical embedding table exists in the database."""
        ...

    @abstractmethod
    def _get_storage_table_descriptor(self, model_record: EmbeddingModelRecord) -> TEmbeddingTable:
        """Build the in-process descriptor for a table known to already exist.
        Must not issue any DDL.

        Returns
        -------
        descriptor
            EmbeddingTable describing the physical table. Type depends on the backend.
        """
        ...

    @abstractmethod
    def _create_storage_table(self, model_record: EmbeddingModelRecord) -> TEmbeddingTable:
        """Create the physical table in the database and return its descriptor.

        Returns
        -------
        descriptor
            EmbeddingTable describing the newly created table. Type depends on the backend.
        """
        ...

    @abstractmethod
    def _delete_storage_table(self, model_record: EmbeddingModelRecord) -> None: ...

    # ------------------------------------------------------------------
    # Core write operations
    # ------------------------------------------------------------------

    @writes
    def upsert_embeddings(
        self,
        *,
        model_name: str,
        records: Sequence[ConceptEmbeddingRecord],
        embeddings: EmbeddingArray,
    ) -> None:
        """Insert or update embeddings for a set of concepts.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        records : Sequence[ConceptEmbeddingRecord]
            Concept metadata rows aligned with ``embeddings``.
        embeddings : EmbeddingArray
            Array of shape ``(N, D)`` where ``N = len(records)`` and ``D`` is
            the registered dimensionality. Converted to float32.

        Raises
        ------
        ValueError
            If ``embeddings`` is not 2-D, has the wrong dimensionality or
            contains NaN or infinite values, or its row count differs from
            ``len(records)``.
        """
        record = self._registered_record(model_name)
        array = as_embedding_array(embeddings, record.dimensions)
        if len(records) != array.shape[0]:
            raise ValueError(
                f"Number of records ({len(records)}) does not match "
                f"number of embeddings ({array.shape[0]})."
            )
        return self._upsert_embeddings_impl(model_record=record, records=records, embeddings=array)

    @abstractmethod
    def _upsert_embeddings_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        records: Sequence[ConceptEmbeddingRecord],
        embeddings: EmbeddingArray,
    ) -> None: ...

    @writes
    def delete_embeddings(
        self,
        *,
        model_name: str,
        concept_ids: Sequence[int],
    ) -> int:
        """Delete the stored embeddings of the given concepts.

        Concept IDs without a stored embedding are ignored. The deletion runs
        in chunks of ``_DELETE_CHUNK_SIZE`` IDs, one transaction each.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        concept_ids : Sequence[int]
            OMOP concept IDs whose embeddings are removed.

        Returns
        -------
        int
            Number of embedding rows deleted.
        """
        model_record = self._registered_record(model_name)
        ids = list(concept_ids)
        return sum(
            self._delete_embeddings_impl(
                model_record=model_record,
                concept_ids=ids[start : start + self._DELETE_CHUNK_SIZE],
            )
            for start in range(0, len(ids), self._DELETE_CHUNK_SIZE)
        )

    @abstractmethod
    def _delete_embeddings_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
    ) -> int: ...

    @writes
    def compact_storage(self, *, model_name: str) -> CompactionResult:
        """Return the disk space held by deleted or rewritten rows of a model.

        Deleting rows, and upserting existing concepts, leaves space that the
        backend does not hand back on its own. This is the only operation that
        does. It rewrites the model's storage, so it needs free disk space for
        a second copy and, on pgvector, an exclusive lock on the table.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.

        Returns
        -------
        CompactionResult
            Storage size before and after.
        """
        return self._compact_storage_impl(model_record=self._registered_record(model_name))

    @abstractmethod
    def _compact_storage_impl(self, *, model_record: EmbeddingModelRecord) -> CompactionResult: ...

    @writes
    def bulk_upsert_embeddings(
        self,
        *,
        model_name: str,
        batches: Iterable[tuple[Sequence[ConceptEmbeddingRecord], EmbeddingArray]],
        total_n_batches: int | None = None,
    ) -> None:
        """Upsert embeddings in multiple batches, delegating to ``upsert_embeddings``.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        batches : Iterable[tuple[Sequence[ConceptEmbeddingRecord], EmbeddingArray]]
            Iterable of ``(records, embeddings)`` pairs.
        total_n_batches : Optional[int]
            Total number of batches for the progress bar.
        """
        import tqdm

        pbar = tqdm.tqdm(
            batches,
            desc=f"Upserting embeddings into {model_name}",
            total=total_n_batches,
        )
        for records, embeddings in pbar:
            self.upsert_embeddings(
                model_name=model_name,
                records=records,
                embeddings=embeddings,
            )

    # ------------------------------------------------------------------
    # Core read operations
    # ------------------------------------------------------------------

    def get_embeddings_by_concept_ids(
        self,
        *,
        model_name: str,
        concept_ids: Sequence[int],
    ) -> Mapping[int, ndarray]:
        """Retrieve stored embeddings for the given concept IDs.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        concept_ids : Sequence[int]
            OMOP concept IDs to look up.

        Returns
        -------
        Mapping[int, ndarray]
            Mapping from concept ID to its float32 embedding vector.

        Raises
        ------
        ValueError
            If any requested concept ID is not found in the table.
        """
        return self._get_embeddings_by_concept_ids_impl(
            model_record=self._registered_record(model_name),
            concept_ids=concept_ids,
        )

    @abstractmethod
    def _get_embeddings_by_concept_ids_impl(
        self,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
    ) -> Mapping[int, ndarray]: ...

    def get_nearest_concepts(
        self,
        *,
        model_name: str,
        metric_type: MetricType,
        query_embeddings: EmbeddingArray,
        concept_filter: EmbeddingConceptFilter | None = None,
        k: int = DEFAULT_K_NEAREST,
    ) -> tuple[tuple[NearestConceptMatch, ...], ...]:
        """Find the nearest stored concepts for one or more query vectors.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        metric_type : MetricType
            Distance metric for the KNN query. Must match the HNSW metric
            exactly, or be any backend-supported metric for FLAT.
        query_embeddings : EmbeddingArray
            Array of shape ``(Q, D)`` where ``Q`` is the number of queries and
            ``D`` is the registered dimensionality. Converted to float32.
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

        Raises
        ------
        ValueError
            If ``query_embeddings`` is not 2-D, has the wrong dimensionality or
            contains NaN or infinite values, or the metric is incompatible
            with the model.
        """
        record = self._registered_record(model_name)
        self._check_query_metric(record, metric_type)
        return self._get_nearest_concepts_impl(
            model_record=record,
            metric_type=metric_type,
            query_embeddings=as_embedding_array(query_embeddings, record.dimensions),
            concept_filter=concept_filter,
            k=k,
        )

    @abstractmethod
    def _get_nearest_concepts_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        metric_type: MetricType,
        query_embeddings: EmbeddingArray,
        concept_filter: EmbeddingConceptFilter | None = None,
        k: int = DEFAULT_K_NEAREST,
    ) -> tuple[tuple[NearestConceptMatch, ...], ...]: ...

    # ------------------------------------------------------------------
    # Utility / diagnostic queries
    # ------------------------------------------------------------------

    def has_any_embeddings(
        self,
        *,
        model_name: str,
    ) -> bool:
        """Return ``True`` if at least one embedding row exists in the table.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.

        Returns
        -------
        bool
        """
        return self._has_any_embeddings_impl(model_record=self._registered_record(model_name))

    @abstractmethod
    def _has_any_embeddings_impl(
        self, *, model_record: EmbeddingModelRecord
    ) -> bool: ...

    def get_stored_concept_ids(
        self,
        *,
        model_name: str,
        concept_filter: EmbeddingConceptFilter | None = None,
    ) -> set[int]:
        """Return the stored concept IDs satisfying concept_filter, or all of them without one.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        concept_filter : EmbeddingConceptFilter, optional
            Filter constraints to evaluate.

        Returns
        -------
        set[int]
        """
        record = self._registered_record(model_name)
        if concept_filter is None or concept_filter.is_empty():
            return self._get_all_stored_concept_ids_impl(model_record=record)
        return self._get_concept_ids_matching_filter_impl(
            model_record=record,
            concept_filter=concept_filter,
        )

    @abstractmethod
    def _get_all_stored_concept_ids_impl(
        self, *, model_record: EmbeddingModelRecord
    ) -> set[int]: ...

    @abstractmethod
    def _get_concept_ids_matching_filter_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        concept_filter: EmbeddingConceptFilter,
    ) -> set[int]: ...

    def get_concept_filter_metadata(
        self,
        *,
        model_name: str,
        concept_ids: Sequence[int],
    ) -> Mapping[int, ConceptEmbeddingRecord]:
        """Fetch the stored concept metadata for the given concept IDs.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        concept_ids : Sequence[int]
            OMOP concept IDs to look up. IDs that are not stored are omitted.

        Returns
        -------
        Mapping[int, ConceptEmbeddingRecord]
        """
        return self._get_concept_filter_metadata_impl(
            model_record=self._registered_record(model_name),
            concept_ids=concept_ids,
        )

    @abstractmethod
    def _get_concept_filter_metadata_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
    ) -> Mapping[int, ConceptEmbeddingRecord]: ...

    def get_embedding_count(
        self,
        *,
        model_name: str,
    ) -> int:
        """Return the number of embeddings stored in the table.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.

        Returns
        -------
        int
        """
        return len(self._get_all_stored_concept_ids_impl(model_record=self._registered_record(model_name)))

    def get_embedding_count_by_vocabulary(
        self,
        *,
        model_name: str,
    ) -> Mapping[str, int]:
        """Return the number of stored embeddings, grouped by vocabulary_id.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.

        Returns
        -------
        Mapping[str, int]
            ``vocabulary_id`` to embedding count, for every vocabulary with at
            least one stored embedding.
        """
        return self._get_embedding_count_by_vocabulary_impl(model_record=self._registered_record(model_name))

    @abstractmethod
    def _get_embedding_count_by_vocabulary_impl(
        self, *, model_record: EmbeddingModelRecord
    ) -> Mapping[str, int]: ...


def _extensions_for(backend_type: BackendType, *, writable: bool) -> Sequence[Callable[[Any, Any], None]]:
    """Connect-event callables for backend_type's engine.

    The writer path may create the pgvector extension. The reader path runs no
    DDL and makes the database itself reject writes on every connection.
    """
    if backend_type == BackendType.SQLITEVEC:
        from omop_emb.backends.sqlitevec.sqlitevec_backend import (
            _load_sqlite_vec,
            _set_query_only,
        )

        return [_load_sqlite_vec] if writable else [_load_sqlite_vec, _set_query_only]
    elif backend_type == BackendType.PGVECTOR:
        from omop_emb.backends.pgvector.pg_backend import (
            _create_vector_extension,
            _set_read_only,
        )

        return [_create_vector_extension] if writable else [_set_read_only]
    elif backend_type == BackendType.DUCKDBVSS:
        from omop_emb.backends.duckdbvss.duckdbvss_backend import _install_vss

        return [_install_vss]
    else:
        raise UnknownEmbeddingBackendError(f"Unknown backend type {backend_type!r}.")


def _connect_args_for(backend_type: BackendType, *, writable: bool) -> dict[str, Any]:
    """Connect-time kwargs for backend_type's engine, forwarded to
    ``sqlalchemy.create_engine(connect_args=...)``.

    Injects DuckDB's ``read_only=True`` flag for the reader path since 
    DuckDB does not have a connect-event hook to enforce read-only mode.
    """
    if backend_type == BackendType.DUCKDBVSS and not writable:
        return {"read_only": True}
    return {}


def _parse_backend_type(backend_type: str | BackendType) -> BackendType:
    """parse_backend_type(), case-insensitive for strings.

    Raises
    ------
    UnknownEmbeddingBackendError
        If backend_type names no known backend.
    """
    try:
        return parse_backend_type(backend_type.lower() if isinstance(backend_type, str) else backend_type)
    except ValueError as exc:
        raise UnknownEmbeddingBackendError(str(exc)) from exc


def _backend_class_for(backend_type: BackendType, *, dialect: str) -> type[EmbeddingBackend]:
    """Validate the dialect for backend_type and return its concrete class.

    Raises
    ------
    EmbeddingBackendConfigurationError
        If the database dialect does not match the backend.
    EmbeddingBackendDependencyError
        If the backend's optional dependencies are not installed.
    """
    if backend_type == BackendType.SQLITEVEC:
        if dialect != Dialect.SQLITE:
            raise EmbeddingBackendConfigurationError(
                f"sqlitevec backend requires a sqlite-dialect database, got dialect: {dialect!r}."
            )
        from omop_emb.backends.sqlitevec import SQLiteVecEmbeddingBackend

        return SQLiteVecEmbeddingBackend

    elif backend_type == BackendType.PGVECTOR:
        if dialect != Dialect.POSTGRESQL:
            raise EmbeddingBackendConfigurationError(
                "The resolved URL must point to a PostgreSQL database "
                f"(pgvector extension required), got dialect: {dialect!r}."
            )
        try:
            from omop_emb.backends.pgvector import PGVectorEmbeddingBackend
        except ImportError as exc:
            raise EmbeddingBackendDependencyError(
                "pgvector backend is not installed. "
                "Install it with: pip install omop-emb[pgvector]"
            ) from exc
        return PGVectorEmbeddingBackend

    elif backend_type == BackendType.DUCKDBVSS:
        if dialect != Dialect.DUCKDB:
            raise EmbeddingBackendConfigurationError(
                f"duckdbvss backend requires a duckdb-dialect database, got dialect: {dialect!r}."
            )
        try:
            from omop_emb.backends.duckdbvss import DuckDBVSSEmbeddingBackend
        except ImportError as exc:
            raise EmbeddingBackendDependencyError(
                "duckdbvss backend is not installed. "
                "Install it with: pip install omop-emb[duckdbvss]"
            ) from exc
        return DuckDBVSSEmbeddingBackend
    else:
        raise UnknownEmbeddingBackendError(f"Unknown backend type {backend_type!r}.")


def _open_writer(
    backend_type: str | BackendType,
    *,
    database: ResolvedDatabase,
) -> EmbeddingBackend:
    """Writable backend on database: registry claims registered, registry table ensured.

    Every backend, including an in-memory sqlite-vec store, is backed by a
    real database entry: ``sqlite:///:memory:`` is still a ``database`` whose
    connection has ``dialect='sqlite'``.
    """
    resolved_backend = _parse_backend_type(backend_type)
    backend_cls = _backend_class_for(
        resolved_backend, dialect=make_url(database.connection.url).get_backend_name()
    )
    emb_engine = registry_writer_engine(
        database,
        extensions=_extensions_for(resolved_backend, writable=True),
        connect_args=_connect_args_for(resolved_backend, writable=True),
    )
    logger.info(f"Using {resolved_backend.value} backend with engine: {emb_engine.url}")
    return backend_cls(emb_engine=emb_engine)


def open_vector_store_writer(resolved: ResolvedVectorStore) -> EmbeddingBackend:
    """Open a resolved vector store for reading and writing.

    Registers the registry's schema claim and ensures its table. Reserve this
    for a CLI/entry-point boundary; library code should receive an
    ``EmbeddingBackend`` as a parameter instead.
    """
    return _open_writer(resolved.backend_type, database=resolved.database)


def open_vector_store_reader(resolved: ResolvedVectorStore) -> EmbeddingStoreReader:
    """Open a resolved vector store read-only: no registration, no DDL, no writes.

    Notes
    -----
    - Double protection against writes:
        1. @writes methods raise ReadOnlyStoreError at the Python level.
        2. The database connection itself is opened in read-only mode.
    - A store that's never been set up reports an empty registry instead of raising.
    """
    resolved_backend = _parse_backend_type(resolved.backend_type)
    database = resolved.database
    backend_cls = _backend_class_for(
        resolved_backend, dialect=make_url(database.connection.url).get_backend_name()
    )
    connect_args = _connect_args_for(resolved_backend, writable=False)
    if resolved_backend == BackendType.DUCKDBVSS:
        # DuckDB rejects read_only=True on :memory: (nothing to protect anyways)
        from omop_emb.backends.duckdbvss.duckdbvss_backend import (
            _reader_database_for_missing_file,
        )

        substituted = _reader_database_for_missing_file(database)
        if substituted is not database:
            connect_args = {}
        database = substituted
    emb_engine = registry_reader_engine(
        database,
        extensions=_extensions_for(resolved_backend, writable=False),
        connect_args=connect_args,
    )
    return backend_cls(emb_engine=emb_engine, writable=False)
