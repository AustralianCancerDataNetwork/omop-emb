from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Optional, Sequence, Type, TypeVar, Generic, Any, Callable, Tuple
from functools import wraps
import logging
from itertools import batched

from numpy import ndarray
from sqlalchemy import Engine, select, Integer, func, Select
from sqlalchemy.orm import sessionmaker, mapped_column

from omop_alchemy.cdm.model.vocabulary import Concept

from omop_emb.config import BackendType, IndexType, MetricType, ProviderType
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter, NearestConceptMatch
from omop_emb.storage.index_config import IndexConfig, INDEX_CONFIG_METADATA_KEY, _RESERVED_METADATA_KEYS

logger = logging.getLogger(__name__)


def require_registered_model(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(
        self,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        **kwargs,
    ) -> Any:
        record = self.get_registered_model(
            model_name=model_name,
            index_type=index_type,
            provider_type=provider_type,
        )
        if record is None:
            raise ValueError(
                f"Embedding model '{model_name}' with index_type='{index_type.value}' "
                f"is not registered in backend '{self.backend_type.value}'."
            )
        kwargs["_model_record"] = record
        return func(self, model_name=model_name, provider_type=provider_type, index_type=index_type, **kwargs)
    return wrapper


class ConceptIDEmbeddingBase:
    """Abstract mixin for concept-ID primary key on embedding tables.

    The FK to ``Concept.concept_id`` has been intentionally removed: the
    embedding table lives in a *separate* Postgres instance from the OMOP CDM,
    so a SQL-level foreign key across databases is impossible.  Application
    code is responsible for validating that concept IDs exist in the CDM before
    inserting embeddings.
    """
    __tablename__: str

    concept_id = mapped_column(Integer, primary_key=True)


T = TypeVar("T", bound=ConceptIDEmbeddingBase)


class EmbeddingBackend(ABC, Generic[T]):
    """Abstract interface for embedding storage and retrieval.

    Parameters
    ----------
    emb_engine : Engine
        SQLAlchemy engine connected to the **dedicated pgvector Postgres
        instance**.  Used for all embedding table operations and for the
        model registry (also stored here).  *Mandatory.*
    omop_cdm_engine : Engine
        SQLAlchemy engine connected to the **user's OMOP CDM**.  Used
        read-only for concept metadata queries (names, domains, standard
        flags).  May be any SQLAlchemy-supported dialect.  *Mandatory.*
    """

    DEFAULT_K_NEAREST = 10

    def __init__(
        self,
        emb_engine: Engine,
        omop_cdm_engine: Engine,
    ):
        super().__init__()
        self._emb_engine = emb_engine
        self._cdm_engine = omop_cdm_engine
        self._emb_session_factory = sessionmaker(emb_engine)
        self._cdm_session_factory = sessionmaker(omop_cdm_engine)
        self._embedding_table_cache: dict[Tuple[str, ProviderType, BackendType, IndexType], Type[T]] = {}
        self._initialise_store()

    # ------------------------------------------------------------------
    # Backend identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Always ``BackendType.PGVECTOR`` for the production backend."""

    @property
    def backend_name(self) -> str:
        return self.backend_type.value

    @property
    def emb_engine(self) -> Engine:
        """Engine for the dedicated pgvector/embedding Postgres instance."""
        return self._emb_engine

    @property
    def cdm_engine(self) -> Engine:
        """Engine for the user's OMOP CDM (read-only)."""
        return self._cdm_engine

    @property
    def emb_session_factory(self) -> sessionmaker:
        return self._emb_session_factory

    @property
    def cdm_session_factory(self) -> sessionmaker:
        return self._cdm_session_factory

    # ------------------------------------------------------------------
    # Store lifecycle — called automatically in __init__
    # ------------------------------------------------------------------

    def _initialise_store(self) -> None:
        """Initialise the backend and populate the embedding table cache.

        Calls ``pre_initialise_store`` first so backends can run setup steps
        (e.g. creating Postgres extensions) before the registry is queried.
        """
        self.pre_initialise_store()
        registered_models = self._registry.get_registered_models_from_db(
            backend_type=self.backend_type
        )
        for model_record in registered_models:
            self._cache_model_record(model_record=model_record)

    def pre_initialise_store(self) -> None:
        """Hook for backend-specific setup that must run before the store is initialised."""

    # ------------------------------------------------------------------
    # Model registration / deletion / configuration
    # ------------------------------------------------------------------

    def register_model(
        self,
        *,
        model_name: str,
        dimensions: int,
        provider_type: ProviderType,
        index_config: IndexConfig,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> "EmbeddingModelRecord":
        """Register an embedding model and create its storage table.

        Parameters
        ----------
        model_name : str
            Canonical name of the embedding model.
        dimensions : int
            Embedding dimensionality.
        provider_type : ProviderType
            Provider that produced the embeddings.
        index_config : IndexConfig
            Index type and parameters.
        metadata : Mapping[str, object], optional
            Additional metadata.  Must not contain reserved keys
            (``"index_config"``, ``"faiss_cache"``).
        """
        metadata = metadata or {}
        persisted_metadata: dict[str, object] = {
            INDEX_CONFIG_METADATA_KEY: index_config.to_dict(),
            **self._validate_external_metadata(metadata),
        }
        model_record = self._registry.register_model(
            model_name=model_name,
            provider_type=provider_type,
            dimensions=dimensions,
            backend_type=self.backend_type,
            index_type=index_config.index_type,
            metadata=persisted_metadata,
        )
        self._cache_model_record(model_record=model_record)
        return model_record

    def delete_model(
        self,
        model_name: str,
        *,
        provider_type: ProviderType,
        index_type: IndexType,
    ) -> None:
        """Irreversibly delete a model: drops its storage table and registry row."""
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
        if self._embedding_table_cache.pop(cache_key, None) is None:
            logger.warning(
                f"Embedding table for model '{model_name}' not found in cache during deletion."
            )
        self._delete_storage_table(model_record=record)
        self._registry.delete_model(
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
    ) -> "EmbeddingModelRecord":
        """Persist updated index configuration parameters for an existing model."""
        if index_config.index_type != index_type:
            raise ValueError(
                f"index_config.index_type ({index_config.index_type!r}) must match "
                f"the registered index_type ({index_type!r})."
            )
        new_metadata = {INDEX_CONFIG_METADATA_KEY: index_config.to_dict()}
        return self._registry.update_model_metadata(
            model_name=model_name,
            provider_type=provider_type,
            backend_type=self.backend_type,
            index_type=index_type,
            metadata=new_metadata,
        )

    @require_registered_model
    def rebuild_model_indexes(
        self,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        *,
        metric_types: Sequence[MetricType],
        batch_size: int = 100_000,
        _model_record: "EmbeddingModelRecord",
    ) -> None:
        """Rebuild backend indexes for *metric_types*."""
        return self._rebuild_model_indexes_impl(
            model_record=_model_record,
            metric_types=metric_types,
            batch_size=batch_size,
        )

    @abstractmethod
    def _rebuild_model_indexes_impl(
        self,
        model_record: "EmbeddingModelRecord",
        *,
        metric_types: Sequence[MetricType],
        batch_size: int = 100_000,
    ) -> None: ...

    # ------------------------------------------------------------------
    # Model registry queries
    # ------------------------------------------------------------------

    def get_registered_model(
        self,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
    ) -> Optional["EmbeddingModelRecord"]:
        results = self._registry.get_registered_models_from_db(
            backend_type=self.backend_type,
            model_name=model_name,
            index_type=index_type,
            provider_type=provider_type,
        )
        if not results:
            return None
        if len(results) != 1:
            raise RuntimeError(
                f"Expected exactly one registered model for '{model_name}' "
                f"(provider='{provider_type.value}', index='{index_type.value}'), "
                f"found {len(results)}."
            )
        return results[0]

    def is_model_registered(
        self,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
    ) -> bool:
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
    ) -> tuple["EmbeddingModelRecord", ...]:
        return self._registry.get_registered_models_from_db(
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
        """Return the ORM class for the model's embedding table.

        Raises ``ValueError`` on a cache miss — this means the model is not
        registered.  Because ``_initialise_store`` is called in ``__init__``,
        a miss here is a genuine logic error, not a setup-order issue.
        """
        key = self._get_embedding_table_cache_key(model_name, provider_type, index_type)
        table = self._embedding_table_cache.get(key)
        if table is not None:
            return table
        raise ValueError(
            f"Embedding table for model '{model_name}' (index='{index_type.value}') "
            f"not found.  Ensure the model is registered before accessing it."
        )

    def _cache_model_record(self, model_record: "EmbeddingModelRecord") -> None:
        dynamic_table = self._create_storage_table(model_record=model_record)
        key = self._get_embedding_table_cache_key(
            model_name=model_record.model_name,
            provider_type=model_record.provider_type,
            index_type=model_record.index_type,
        )
        self._embedding_table_cache[key] = dynamic_table

    def _get_embedding_table_cache_key(
        self,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
    ) -> Tuple[str, ProviderType, BackendType, IndexType]:
        return (model_name, provider_type, self.backend_type, index_type)

    @abstractmethod
    def _create_storage_table(self, model_record: "EmbeddingModelRecord") -> Type[T]: ...

    @abstractmethod
    def _delete_storage_table(self, model_record: "EmbeddingModelRecord") -> None: ...

    # ------------------------------------------------------------------
    # Core read / write operations
    # ------------------------------------------------------------------

    def _get_existing_concept_ids(
        self,
        concept_ids: tuple[int, ...],
        table: Type[T],
    ) -> set[int]:
        """Return concept IDs from *concept_ids* already present in *table* (emb engine)."""
        if not concept_ids:
            return set()
        with self.emb_session_factory() as session:
            result = session.execute(
                select(table.concept_id).where(table.concept_id.in_(concept_ids))
            )
            return {row[0] for row in result}

    @require_registered_model
    def upsert_embeddings(
        self,
        *,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        concept_ids: Sequence[int],
        embeddings: ndarray,
        _model_record: "EmbeddingModelRecord",
        metric_type: Optional[MetricType] = None,
    ) -> None:
        """Insert or update vector embeddings for a collection of OMOP concept IDs."""
        return self._upsert_embeddings_impl(
            model_record=_model_record,
            concept_ids=concept_ids,
            embeddings=embeddings,
            metric_type=metric_type,
        )

    @abstractmethod
    def _upsert_embeddings_impl(
        self,
        *,
        model_record: "EmbeddingModelRecord",
        concept_ids: Sequence[int],
        embeddings: ndarray,
        metric_type: Optional[MetricType] = None,
    ) -> None: ...

    def bulk_upsert_embeddings(
        self,
        *,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        batches: Iterable[Tuple[Sequence[int], ndarray]],
        metric_type: Optional[MetricType] = None,
    ) -> None:
        """Upsert embeddings from a lazy iterable of ``(concept_ids, embeddings)`` batches."""
        for concept_ids, embeddings in batches:
            self.upsert_embeddings(
                model_name=model_name,
                provider_type=provider_type,
                index_type=index_type,
                concept_ids=concept_ids,
                embeddings=embeddings,
                metric_type=metric_type,
            )

    @require_registered_model
    def get_embeddings_by_concept_ids(
        self,
        *,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        concept_ids: Sequence[int],
        _model_record: "EmbeddingModelRecord",
    ) -> Mapping[int, Sequence[float]]:
        return self._get_embeddings_by_concept_ids_impl(
            model_record=_model_record,
            concept_ids=concept_ids,
        )

    @abstractmethod
    def _get_embeddings_by_concept_ids_impl(
        self,
        model_record: "EmbeddingModelRecord",
        concept_ids: Sequence[int],
    ) -> Mapping[int, Sequence[float]]: ...

    @require_registered_model
    def get_nearest_concepts(
        self,
        *,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        query_embeddings: ndarray,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        _model_record: "EmbeddingModelRecord",
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        """Return nearest stored concepts for each query embedding.

        Returns
        -------
        Tuple[Tuple[NearestConceptMatch, ...], ...]
            Shape ``(Q, k)`` — one inner tuple per query vector.
        """
        return self._get_nearest_concepts_impl(
            model_record=_model_record,
            query_embeddings=query_embeddings,
            metric_type=metric_type,
            concept_filter=concept_filter,
        )

    @abstractmethod
    def _get_nearest_concepts_impl(
        self,
        *,
        model_record: "EmbeddingModelRecord",
        query_embeddings: ndarray,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]: ...

    # ------------------------------------------------------------------
    # Cross-DB concept helpers (CDM engine)
    # ------------------------------------------------------------------

    def _get_candidate_concept_ids_from_cdm(
        self,
        concept_filter: Optional[EmbeddingConceptFilter],
    ) -> Optional[tuple[int, ...]]:
        """Return CDM concept IDs matching *concept_filter* CDM criteria, or ``None``.

        Returns ``None`` when the filter has no CDM criteria (no domain/
        vocabulary/require_standard/concept_ids), signalling the ANN search
        should not be pre-filtered by concept identity.
        """
        if concept_filter is None:
            return None
        has_cdm_criteria = (
            concept_filter.concept_ids is not None
            or concept_filter.domains is not None
            or concept_filter.vocabularies is not None
            or concept_filter.require_standard
        )
        if not has_cdm_criteria:
            return None

        query = select(Concept.concept_id)
        if concept_filter.concept_ids is not None:
            query = query.where(Concept.concept_id.in_(concept_filter.concept_ids))
        if concept_filter.domains is not None:
            query = query.where(Concept.domain_id.in_(concept_filter.domains))
        if concept_filter.vocabularies is not None:
            query = query.where(Concept.vocabulary_id.in_(concept_filter.vocabularies))
        if concept_filter.require_standard:
            query = query.where(Concept.standard_concept.in_(["S", "C"]))

        with self.cdm_session_factory() as session:
            return tuple(row[0] for row in session.execute(query))

    def _fetch_concept_metadata(
        self,
        concept_ids: set[int],
    ) -> dict[int, Any]:
        """Fetch concept names and OMOP flags from CDM for *concept_ids*.

        Returns a dict mapping ``concept_id`` → row with attributes
        ``concept_name``, ``standard_concept``, ``invalid_reason``.
        """
        if not concept_ids:
            return {}
        query = select(
            Concept.concept_id,
            Concept.concept_name,
            Concept.standard_concept,
            Concept.invalid_reason,
        ).where(Concept.concept_id.in_(concept_ids))
        with self.cdm_session_factory() as session:
            return {row.concept_id: row for row in session.execute(query)}

    # ------------------------------------------------------------------
    # Concept queries (split across both engines)
    # ------------------------------------------------------------------

    def has_any_embeddings(
        self,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
    ) -> bool:
        table = self.get_embedding_table(
            model_name=model_name,
            index_type=index_type,
            provider_type=provider_type,
        )
        with self.emb_session_factory() as session:
            return session.execute(select(table.concept_id).limit(1)).first() is not None

    def get_concepts_without_embedding(
        self,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> Mapping[int, str]:
        """Return concept IDs and names for concepts that do not yet have embeddings.

        CDM and embedding engine queries are executed separately and the
        difference is computed in Python, supporting cross-database deployments.
        """
        # Step 1: all eligible concepts from CDM
        cdm_query = select(Concept.concept_id, Concept.concept_name)
        if concept_filter is not None:
            cdm_query = concept_filter.apply(cdm_query)
        with self.cdm_session_factory() as session:
            all_concepts = {row.concept_id: row.concept_name for row in session.execute(cdm_query)}

        # Step 2: already-embedded IDs from emb engine
        table = self.get_embedding_table(
            model_name=model_name, index_type=index_type, provider_type=provider_type
        )
        with self.emb_session_factory() as session:
            embedded_ids = {row[0] for row in session.execute(select(table.concept_id))}

        return {cid: name for cid, name in all_concepts.items() if cid not in embedded_ids}

    def get_concepts_without_embedding_batched(
        self,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        batch_size: int,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> Iterable[Mapping[int, str]]:
        """Lazily yield batches of unembedded concept IDs and names."""
        missing = self.get_concepts_without_embedding(
            model_name=model_name,
            provider_type=provider_type,
            index_type=index_type,
            concept_filter=concept_filter,
        )
        items = list(missing.items())
        for batch in batched(items, batch_size):
            yield dict(batch)

    def get_concepts_without_embedding_count(
        self,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> int:
        # Reuse the full fetch to get an accurate count for cross-DB case.
        return len(
            self.get_concepts_without_embedding(
                model_name=model_name,
                provider_type=provider_type,
                index_type=index_type,
                concept_filter=concept_filter,
            )
        )

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
            max_k = max(len(d) for d in nearest_concepts)
            raise RuntimeError(
                f"Expected at most {k} nearest neighbors per query, found {max_k}."
            )
        if len(nearest_concepts) != query_embeddings.shape[0]:
            raise RuntimeError(
                f"Expected results for {query_embeddings.shape[0]} query embeddings, "
                f"got {len(nearest_concepts)}."
            )

    @staticmethod
    def validate_embeddings(embeddings: ndarray, dimensions: int) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D array, got ndim={embeddings.ndim}.")
        if embeddings.shape[1] != dimensions:
            raise ValueError(
                f"Embedding dimensionality ({embeddings.shape[1]}) does not match "
                f"model configuration ({dimensions})."
            )

    @staticmethod
    def validate_embeddings_and_concept_ids(
        embeddings: ndarray,
        concept_ids: Sequence[int] | ndarray,
        dimensions: int,
    ) -> None:
        EmbeddingBackend.validate_embeddings(embeddings, dimensions=dimensions)
        if len(concept_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Number of concept IDs ({len(concept_ids)}) does not match "
                f"number of embeddings ({embeddings.shape[0]})."
            )

    @staticmethod
    def _validate_external_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Expected metadata to be a mapping, got {type(metadata)}")
        bad_keys = _RESERVED_METADATA_KEYS & metadata.keys()
        if bad_keys:
            raise ValueError(f"Metadata contains reserved keys: {bad_keys}")
        return metadata


# Deferred import to avoid circular references at module load time.
from omop_emb.storage.postgres.pg_registry import EmbeddingModelRecord  # noqa: E402, F401
