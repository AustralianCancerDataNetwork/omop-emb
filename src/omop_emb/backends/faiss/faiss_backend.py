from __future__ import annotations

from pathlib import Path
from typing import (
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Type,
    Dict,
    Tuple,
    Any
)

import numpy as np
from numpy import ndarray
from sqlalchemy import Engine, select, func
import logging
import shutil

from .faiss_sql import (
    FAISSConceptIDEmbeddingRegistry,
    create_faiss_embedding_registry_table,
    delete_faiss_embedding_registry_table,
    q_add_concept_ids_to_faiss_registry,
    q_concept_ids_with_embeddings,
    q_concept_ids_with_embeddings_without_metadata,
)
from omop_emb.config import BackendType, IndexType, MetricType, ProviderType
from omop_emb.backends.index_config import (
    IndexConfig,
    index_config_from_index_type_and_metadata,
)
from omop_emb.backends.faiss.storage_manager import EmbeddingStorageManager
from omop_emb.backends.base_backend import EmbeddingBackend, require_registered_model
from omop_emb.utils.embedding_utils import (
    EmbeddingConceptFilter,
    NearestConceptMatch,
    get_similarity_from_distance,
)
from omop_emb.model_registry import EmbeddingModelRecord

logger = logging.getLogger(__name__)


class FaissEmbeddingBackend(EmbeddingBackend[FAISSConceptIDEmbeddingRegistry]):
    """FAISS backend backed by HDF5 vector storage and SQL concept-id registry tables.

    Notes
    -----
    - The FAISS index files and HDF5 embedding files are stored in the filesystem,
      while the mapping between concept IDs and their positions in the FAISS index
      is stored in a SQLAlchemy registry table.
    - The HDF5 files store the raw embeddings and are used to populate the FAISS
      index. When new embeddings are upserted, they are appended to the HDF5 file
      and the FAISS index is updated accordingly. The registry table is also updated
      with any new concept IDs to maintain the mapping.

    Parameters
    ----------
    storage_base_dir : str | Path, optional
        Base directory for FAISS files and local metadata.
        Resolution is handled by ``EmbeddingBackend``: explicit argument, then
        ``OMOP_EMB_BASE_STORAGE_DIR``, then backend default.
    registry_db_name : str, optional
        Optional local model-registry database filename. If not provided, the
        registry will be created in the default location defined by
        ``EmbeddingBackend``, which resolves to ~/.omop_emb/ by default.
    """

    def __init__(
        self,
        omop_cdm_engine: Engine,
        storage_base_dir: Optional[str | Path] = None,
        registry_db_name: Optional[str] = None,
    ):
        super().__init__(
            omop_cdm_engine=omop_cdm_engine,
            storage_base_dir=storage_base_dir,
            registry_db_name=registry_db_name,
        )
        self._embedding_storage_managers: Dict[str, EmbeddingStorageManager] = {}

    @property
    def backend_type(self) -> BackendType:
        return BackendType.FAISS

    def _create_storage_table(self, model_record: EmbeddingModelRecord) -> Type[FAISSConceptIDEmbeddingRegistry]:
        return create_faiss_embedding_registry_table(engine=self.cdm_engine, model_record=model_record)

    def _delete_storage_table(self, model_record: EmbeddingModelRecord) -> None:
        delete_faiss_embedding_registry_table(engine=self.cdm_engine, model_record=model_record)

    def get_safe_model_dir(self, model_name: str) -> Path:
        return self.storage_base_dir / self._embedding_model_registry.safe_model_name(model_name)

    def has_stale_model_artifacts(self, model_name: str) -> bool:
        model_dir = self.get_safe_model_dir(model_name)
        return model_dir.exists() and any(model_dir.iterdir())

    # ------------------------------------------------------------------
    # Storage manager lifecycle
    # ------------------------------------------------------------------

    def get_storage_manager(self, model_record: EmbeddingModelRecord) -> EmbeddingStorageManager:
        self.register_storage_manager(model_record)
        return self._embedding_storage_managers[model_record.model_name]

    def register_storage_manager(self, model_record: EmbeddingModelRecord) -> None:
        """Ensure an in-memory storage manager exists for *model_record* obtained from the ModelRegistry.
        If it doesn't exist yet, create a new one using the persisted index configuration from the registry metadata.

        Raises ``ValueError`` when a manager already exists for the same model name but the
        new registration declares a different embedding dimension.  Two registrations sharing
        a model name (e.g. different index types) must agree on dimensions because they share
        the same HDF5 file.
        """
        existing = self._embedding_storage_managers.get(model_record.model_name)

        if existing is not None:
            if existing.dimensions != model_record.dimensions:
                raise ValueError(
                    f"Dimension mismatch for model '{model_record.model_name}': "
                    f"the existing storage manager was created with dimensions={existing.dimensions} "
                    f"but the new registration declares dimensions={model_record.dimensions}. "
                    "Registrations that share a model name must have the same embedding dimension "
                    "because they share the same HDF5 storage file."
                )
            return

        logger.info(
            f"Registering storage manager for '{model_record.model_name}' "
            f"(dimensions={model_record.dimensions})"
        )
        self._embedding_storage_managers[model_record.model_name] = EmbeddingStorageManager(
            file_dir=self.get_safe_model_dir(model_record.model_name),
            dimensions=model_record.dimensions,
            backend_type=self.backend_type,
        )

    # ------------------------------------------------------------------
    # Model registration / deletion / configuration update
    # ------------------------------------------------------------------

    def register_model(
        self,
        model_name: str,
        dimensions: int,
        *,
        provider_type: ProviderType,
        index_config: IndexConfig,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> EmbeddingModelRecord:
        """Register a model with the FAISS backend.

        HNSW parameters are serialized into the registry metadata so they can be
        reconstructed faithfully after a process restart — without the caller
        having to supply them again.

        The model directory is created *before* the registry row is written so that
        a filesystem failure cannot produce an orphaned metadata entry with no
        corresponding storage directory.
        """
        model_dir = self.get_safe_model_dir(model_name)
        model_dir.mkdir(parents=False, exist_ok=True)
        logger.info(f"Created model directory: {model_dir}")

        try:
            record = super().register_model(
                model_name=model_name,
                provider_type=provider_type,
                dimensions=dimensions,
                index_config=index_config,
                metadata=metadata,
            )
        except Exception:
            if model_dir.exists() and not any(model_dir.iterdir()):
                model_dir.rmdir()
                logger.debug(f"Rolled back empty model directory {model_dir} after registry write failure.")
            raise

        self.register_storage_manager(record)
        return record

    def delete_model(
        self,
        model_name: str,
        *,
        provider_type: ProviderType,
        index_type: IndexType,
    ) -> None:
        # Snapshot surviving registrations *before* the delete so we know
        # whether other index-type rows still share the same model directory.
        all_registrations = self.get_registered_models(model_name=model_name)
        other_registrations = [
            r for r in all_registrations
            if not (r.provider_type == provider_type and r.index_type == index_type)
        ]

        super().delete_model(
            model_name=model_name,
            provider_type=provider_type,
            index_type=index_type,
        )

        self._embedding_storage_managers.pop(model_name, None)
        model_dir = self.get_safe_model_dir(model_name)

        if other_registrations:
            # The HDF5 and other index files are shared — only remove the
            # on-disk FAISS index files that belong to this specific index_type.
            if model_dir.exists():
                index_dir = model_dir / f"index_{index_type.value}"
                if index_dir.exists():
                    shutil.rmtree(index_dir)
                    logger.info(
                        f"Deleted index directory '{index_dir}' for model '{model_name}' "
                        f"(index_type={index_type.value}). "
                        f"{len(other_registrations)} other registration(s) still active; "
                        "HDF5 storage preserved."
                    )
        elif model_dir.exists():
            shutil.rmtree(model_dir)
            logger.info(f"Deleted model directory '{model_dir}' for model '{model_name}'.")

    def update_model_index_configuration(
        self,
        model_name: str,
        *,
        provider_type: ProviderType,
        index_type: IndexType,
        index_config: IndexConfig,
    ) -> EmbeddingModelRecord:
        new_record = super().update_model_index_configuration(
            model_name=model_name,
            provider_type=provider_type,
            index_type=index_type,
            index_config=index_config,
        )
        # Evict the stale in-memory manager.
        old_manager = self._embedding_storage_managers.pop(model_name, None)
        if old_manager is not None:
            # Delete the on-disk FAISS index files for this index_type so a
            # subsequent search cannot silently load them with the old parameters.
            old_manager.cleanup(index_type)
            logger.info(
                f"Deleted stale on-disk FAISS index files for model '{model_name}' "
                f"(index_type={index_type.value}) after config update. "
                "Call rebuild_model_indexes() to rebuild with the new parameters."
            )
        # Reconstruct a fresh storage manager from the updated record.
        self.get_storage_manager(new_record)
        return new_record

    # ------------------------------------------------------------------
    # Consistency / maintenance helpers
    # ------------------------------------------------------------------

    def _validate_storage_consistency(
        self,
        model_record: EmbeddingModelRecord,
    ) -> None:
        """Raise if the SQL registry row count differs from the HDF5 vector count."""
        embedding_table = self.get_embedding_table(
            model_name=model_record.model_name,
            provider_type=model_record.provider_type,
            index_type=model_record.index_type,
        )

        with self.session_factory() as session:
            db_count: int = session.scalar(
                select(func.count()).select_from(embedding_table)
            ) or 0

        hdf5_count = self.get_storage_manager(model_record).get_count()
        if db_count != hdf5_count:
            raise RuntimeError(
                f"Storage consistency error for model '{model_record.model_name}': "
                f"SQL registry has {db_count} entries but HDF5 storage has {hdf5_count}. "
                "Manual reconciliation is required."
            )

    @require_registered_model
    def initialise_indexes(
        self,
        *,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        metric_types: Sequence[MetricType],
        _model_record: EmbeddingModelRecord,
        batch_size: int = 100_000,
    ) -> None:
        """Load or build FAISS indexes for the given metric types from HDF5 storage.

        Call this on process start to warm up indexes before the first search query.
        If index files already exist on disk they are loaded; otherwise they are
        built from the HDF5 embeddings and saved.

        Skips index creation when HDF5 storage is empty to avoid writing an empty
        index file that would appear valid on reload but return no results.
        """
        storage_manager = self.get_storage_manager(_model_record)
        if storage_manager.get_count() == 0:
            logger.warning(
                f"Skipping index initialisation for model '{model_name}' "
                f"(index_type={index_type.value}): HDF5 storage contains no embeddings. "
                "Upsert embeddings first, then call initialise_indexes() again."
            )
            return
        config = index_config_from_index_type_and_metadata(
            _model_record.index_type, _model_record.metadata
        )
        for metric_type in metric_types:
            storage_manager.create_index_for_metric(
                metric_type=metric_type,
                index_config=config,
                batch_size=batch_size,
            )

    def _rebuild_model_indexes_impl(
        self,
        model_record: EmbeddingModelRecord,
        *,
        metric_types: Sequence[MetricType],
        batch_size: int = 100_000,
    ) -> None:
        """Delete and rebuild the FAISS index files for the given metric types.

        Use this after calling ``update_model_index_configuration`` to apply the
        new parameters, or any time the HDF5 embeddings have changed in a way that
        makes the existing index stale.
        """
        config = index_config_from_index_type_and_metadata(model_record.index_type, model_record.metadata)
        storage_manager = self.get_storage_manager(model_record)
        for metric_type in metric_types:
            storage_manager.rebuild_index_for_metric(
                metric_type=metric_type,
                index_config=config,
                batch_size=batch_size,
            )

    # ------------------------------------------------------------------
    # Core read/write operations
    # ------------------------------------------------------------------

    def _upsert_embeddings_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
        embeddings: ndarray,
        metric_type: Optional[MetricType] = None,
    ) -> None:
        concept_id_tuple = tuple(concept_ids)
        self.validate_embeddings_and_concept_ids(
            concept_ids=concept_id_tuple,
            embeddings=embeddings,
            dimensions=model_record.dimensions,
        )

        embedding_table = self.get_embedding_table(
            model_name=model_record.model_name,
            index_type=model_record.index_type,
            provider_type=model_record.provider_type,
        )

        existing = self._get_existing_concept_ids(concept_id_tuple, embedding_table)
        if existing:
            raise ValueError(
                f"concept_ids already present in registry for model '{model_record.model_name}': {existing}. "
                "Existing concept_ids cannot be overwritten."
            )

        config = index_config_from_index_type_and_metadata(
            model_record.index_type, model_record.metadata
        )
        storage_manager = self.get_storage_manager(model_record)
        previous_count = storage_manager.get_count()

        storage_manager.append(
            concept_ids=np.array(concept_id_tuple, dtype=np.int64),
            embeddings=embeddings,
            index_config=config if metric_type is not None else None,
            metric_type=metric_type,
        )

        actual_count = storage_manager.get_count()
        expected_count = previous_count + len(concept_id_tuple)
        if actual_count != expected_count:
            raise RuntimeError(
                f"HDF5 append failed for model '{model_record.model_name}': expected {expected_count} "
                f"stored vectors after append, found {actual_count}. SQL registry was not updated."
            )

        query = q_add_concept_ids_to_faiss_registry(
            concept_ids=concept_id_tuple,
            registered_table=embedding_table,
        )

        try:
            with self.session_factory.begin() as session:
                session.execute(query)

        except Exception as e:
            # Revert HDF5 to pre-append state to maintain consistency with the SQL registry, which was not updated due to the exception.
            storage_manager._truncate_to(previous_count)
            logger.error(f"Failed to add concept IDs to FAISS registry for model '{model_record.model_name}'. Rolled back HDF5 storage to previous count of {previous_count}. Error: {e}")
            raise e

    @require_registered_model
    def bulk_upsert_embeddings(
        self,
        *,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        batches: Iterable[Tuple[Sequence[int], ndarray]],
        metric_type: Optional[MetricType] = None,
        _model_record: EmbeddingModelRecord,
    ) -> None:
        """Bulk-load embeddings using a single HDF5 open, one index rebuild, and one SQL insert.

        Parameters
        ----------
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
            When provided the FAISS index for this metric is rebuilt once all
            batches are written. Omit to defer index creation.
        _model_record : EmbeddingModelRecord
            Injected by ``@require_registered_model``; do not pass explicitly.

        Notes
        -----
        Skips per-batch duplicate checks and FAISS saves. The SQL registry
        primary-key constraint catches cross-batch duplicates at commit time.
        """
        config = index_config_from_index_type_and_metadata(
            _model_record.index_type, _model_record.metadata
        )
        storage_manager = self.get_storage_manager(_model_record)

        def _validated_batches() -> Iterable[Tuple[np.ndarray, np.ndarray]]:
            for concept_ids, embeddings in batches:
                arr_ids = np.array(tuple(concept_ids), dtype=np.int64)
                self.validate_embeddings_and_concept_ids(
                    concept_ids=arr_ids,
                    embeddings=embeddings,
                    dimensions=_model_record.dimensions,
                )
                yield arr_ids, embeddings

        pre_write_count = storage_manager.get_count()
        all_concept_ids = storage_manager._bulk_write(_validated_batches())

        if metric_type is not None and all_concept_ids:
            storage_manager.rebuild_index_for_metric(
                metric_type=metric_type,
                index_config=config,
            )

        if all_concept_ids:
            query = q_add_concept_ids_to_faiss_registry(
                concept_ids=tuple(all_concept_ids),
                registered_table=self.get_embedding_table(
                    model_name=model_name,
                    index_type=index_type,
                    provider_type=provider_type,
                )
            )
            try:
                with self.session_factory.begin() as session:
                    session.execute(query)
            except Exception as e:
                # Roll back HDF5 to its pre-write state so HDF5 and SQL stay in sync.
                storage_manager._truncate_to(pre_write_count)
                raise ValueError(
                    f"Failed to register concept IDs after bulk load for model '{model_name}'. "
                    "HDF5 storage has been rolled back to its pre-write state. "
                    f"Original error: {e}"
                ) from e

    def _get_nearest_concepts_impl(
        self,
        *,
        model_record: EmbeddingModelRecord,
        query_embeddings: ndarray,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        config = index_config_from_index_type_and_metadata(
            model_record.index_type, model_record.metadata
        )
        storage_manager = self.get_storage_manager(model_record)

        if storage_manager.get_count() == 0:
            return tuple(() for _ in range(query_embeddings.shape[0]))

        storage_manager.create_index_for_metric(
            metric_type=metric_type,
            index_config=config,
        )

        self.validate_embeddings(embeddings=query_embeddings, dimensions=model_record.dimensions)

        embedding_table = self.get_embedding_table(
            model_name=model_record.model_name,
            index_type=model_record.index_type,
            provider_type=model_record.provider_type,
        )

        if concept_filter is None or concept_filter.is_empty():
            permitted_concept_ids: Optional[np.ndarray] = None
            k = self.DEFAULT_K_NEAREST
        else:

            query = q_concept_ids_with_embeddings_without_metadata(
                embedding_table=embedding_table,
                concept_filter=concept_filter,
            )
            with self.session_factory() as session:
                permitted_rows = session.execute(query).all()

            permitted_concept_ids = np.array(
                [row.concept_id for row in permitted_rows], dtype=np.int64
            )
            k = concept_filter.limit if concept_filter.limit is not None else self.DEFAULT_K_NEAREST

        distances, returned_ids = storage_manager.search(
            query_vector=query_embeddings,
            metric_type=metric_type,
            index_type=model_record.index_type,
            k=k,
            subset_concept_ids=permitted_concept_ids,
        )

        # Fetch metadata for only the concept_ids FAISS actually returned
        unique_returned = np.unique(returned_ids[returned_ids != -1])
        concept_meta = {}
        if unique_returned.size > 0:
            meta_filter = EmbeddingConceptFilter(concept_ids=tuple(int(x) for x in unique_returned))
            with self.session_factory() as session:
                meta_rows = session.execute(
                    q_concept_ids_with_embeddings(embedding_table, meta_filter)
                ).all()
            concept_meta = {row.concept_id: row for row in meta_rows}

        matches = []
        for concept_id_per_query, distance_per_query in zip(returned_ids, distances):
            matches_per_query = []
            for concept_id, distance in zip(concept_id_per_query, distance_per_query):
                if concept_id == -1:
                    continue
                row = concept_meta.get(int(concept_id))
                if row is None:
                    logger.warning(
                        f"Concept ID {concept_id} returned by FAISS but not found in "
                        "concept metadata. Skipping."
                    )
                    continue
                similarity = get_similarity_from_distance(distance.item(), metric_type)
                if not isinstance(similarity, float):
                    raise RuntimeError(f"Expected float similarity, got {type(similarity)}")
                matches_per_query.append(NearestConceptMatch(
                    concept_id=int(concept_id),
                    concept_name=row.concept_name,
                    similarity=similarity,
                    is_standard=bool(row.is_standard),
                    is_active=bool(row.is_active),
                ))
            matches.append(tuple(matches_per_query))

        matches_tuple = tuple(matches)
        self.validate_nearest_concepts_output(matches_tuple, k, query_embeddings=query_embeddings)
        return matches_tuple

    def _get_embeddings_by_concept_ids_impl(
        self,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
    ) -> Mapping[int, Sequence[float]]:
        concept_id_tuple = tuple(concept_ids)
        if not concept_id_tuple:
            return {}

        storage_manager = self.get_storage_manager(model_record)
        return storage_manager.get_embeddings_by_concept_ids(
            concept_ids=np.array(concept_id_tuple, dtype=np.int64)
        )
