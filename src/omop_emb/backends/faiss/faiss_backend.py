from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Type

import numpy as np
from numpy import ndarray
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from omop_alchemy.cdm.model.vocabulary import Concept

from omop_emb.config import (
    BackendType,
    IndexType,
    MetricType,
    ProviderType,
    get_supported_metrics_for_backend_index,
)
from omop_emb.model_registry import EmbeddingModelRecord
from omop_emb.utils.embedding_utils import (
    EmbeddingConceptFilter,
    NearestConceptMatch,
    get_similarity_from_distance,
)

from ..base import EmbeddingBackend, require_registered_model
from .faiss_sql import (
    FAISSConceptIDEmbeddingRegistry,
    add_concept_ids_to_faiss_registry,
    create_faiss_embedding_registry_table,
    q_concept_ids_with_embeddings,
)
from .storage_manager import EmbeddingStorageManager

logger = logging.getLogger(__name__)

FAISS_METADATA_HNSW_NUM_NEIGHBORS = "hnsw_num_neighbors"
FAISS_METADATA_HNSW_EF_SEARCH = "hnsw_ef_search"
FAISS_METADATA_HNSW_EF_CONSTRUCTION = "hnsw_ef_construction"
DEFAULT_HNSW_NUM_NEIGHBORS = 32
DEFAULT_HNSW_EF_SEARCH = 64
DEFAULT_HNSW_EF_CONSTRUCTION = 200


def _coerce_positive_int(*, value: Optional[object], field_name: str, default: int) -> int:
    if value is None:
        return default
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer. Got {value!r}.") from exc
    if resolved <= 0:
        raise ValueError(f"{field_name} must be a positive integer. Got {resolved}.")
    return resolved


def build_faiss_index_metadata(
    *,
    index_type: IndexType,
    existing_metadata: Optional[Mapping[str, object]] = None,
    hnsw_num_neighbors: Optional[int] = None,
    hnsw_ef_search: Optional[int] = None,
    hnsw_ef_construction: Optional[int] = None,
) -> dict[str, object]:
    metadata = dict(existing_metadata or {})
    for key in (
        FAISS_METADATA_HNSW_NUM_NEIGHBORS,
        FAISS_METADATA_HNSW_EF_SEARCH,
        FAISS_METADATA_HNSW_EF_CONSTRUCTION,
    ):
        metadata.pop(key, None)

    if index_type == IndexType.HNSW:
        metadata[FAISS_METADATA_HNSW_NUM_NEIGHBORS] = _coerce_positive_int(
            value=hnsw_num_neighbors,
            field_name=FAISS_METADATA_HNSW_NUM_NEIGHBORS,
            default=DEFAULT_HNSW_NUM_NEIGHBORS,
        )
        metadata[FAISS_METADATA_HNSW_EF_SEARCH] = _coerce_positive_int(
            value=hnsw_ef_search,
            field_name=FAISS_METADATA_HNSW_EF_SEARCH,
            default=DEFAULT_HNSW_EF_SEARCH,
        )
        metadata[FAISS_METADATA_HNSW_EF_CONSTRUCTION] = _coerce_positive_int(
            value=hnsw_ef_construction,
            field_name=FAISS_METADATA_HNSW_EF_CONSTRUCTION,
            default=DEFAULT_HNSW_EF_CONSTRUCTION,
        )

    return metadata


def resolve_faiss_hnsw_parameters(
    metadata: Optional[Mapping[str, object]],
) -> tuple[int, int, int]:
    metadata = metadata or {}
    return (
        _coerce_positive_int(
            value=metadata.get(FAISS_METADATA_HNSW_NUM_NEIGHBORS),
            field_name=FAISS_METADATA_HNSW_NUM_NEIGHBORS,
            default=DEFAULT_HNSW_NUM_NEIGHBORS,
        ),
        _coerce_positive_int(
            value=metadata.get(FAISS_METADATA_HNSW_EF_SEARCH),
            field_name=FAISS_METADATA_HNSW_EF_SEARCH,
            default=DEFAULT_HNSW_EF_SEARCH,
        ),
        _coerce_positive_int(
            value=metadata.get(FAISS_METADATA_HNSW_EF_CONSTRUCTION),
            field_name=FAISS_METADATA_HNSW_EF_CONSTRUCTION,
            default=DEFAULT_HNSW_EF_CONSTRUCTION,
        ),
    )


class FaissEmbeddingBackend(EmbeddingBackend[FAISSConceptIDEmbeddingRegistry]):
    """FAISS backend backed by HDF5 vector storage and SQL concept-id registry tables."""

    def __init__(
        self,
        storage_base_dir: Optional[str | Path] = None,
        registry_db_name: Optional[str] = None,
    ):
        super().__init__(
            storage_base_dir=storage_base_dir,
            registry_db_name=registry_db_name,
        )
        self._embedding_storage_managers: Dict[str, EmbeddingStorageManager] = {}

    @property
    def backend_type(self) -> BackendType:
        return BackendType.FAISS

    @property
    def embedding_storage_managers(self) -> Dict[str, EmbeddingStorageManager]:
        return self._embedding_storage_managers

    def _create_storage_table(
        self,
        engine: Engine,
        model_record: EmbeddingModelRecord,
    ) -> Type[FAISSConceptIDEmbeddingRegistry]:
        return create_faiss_embedding_registry_table(engine=engine, model_record=model_record)

    def get_safe_model_dir(self, model_name: str) -> Path:
        return self.storage_base_dir / self.embedding_model_registry.safe_model_name(model_name)

    def has_stale_model_artifacts(self, model_name: str) -> bool:
        model_dir = self.get_safe_model_dir(model_name)
        return model_dir.exists() and any(model_dir.iterdir())

    def get_storage_manager(
        self,
        *,
        model_name: str,
        dimensions: int,
        index_type: IndexType,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> EmbeddingStorageManager:
        self.register_storage_manager(
            model_name=model_name,
            dimensions=dimensions,
            index_type=index_type,
            metadata=metadata,
        )
        return self.embedding_storage_managers[model_name]

    def register_storage_manager(
        self,
        *,
        model_name: str,
        dimensions: int,
        index_type: IndexType,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        hnsw_num_neighbors, hnsw_ef_search, hnsw_ef_construction = resolve_faiss_hnsw_parameters(metadata)
        existing_manager = self.embedding_storage_managers.get(model_name)
        if (
            existing_manager is None
            or existing_manager.dimensions != dimensions
            or existing_manager.hnsw_num_neighbors != hnsw_num_neighbors
            or existing_manager.hnsw_ef_search != hnsw_ef_search
            or existing_manager.hnsw_ef_construction != hnsw_ef_construction
        ):
            self.embedding_storage_managers[model_name] = EmbeddingStorageManager(
                file_dir=self.get_safe_model_dir(model_name),
                dimensions=dimensions,
                backend_type=self.backend_type,
                hnsw_num_neighbors=hnsw_num_neighbors,
                hnsw_ef_search=hnsw_ef_search,
                hnsw_ef_construction=hnsw_ef_construction,
            )

    def update_model_index_configuration(
        self,
        *,
        session: Session,
        model_name: str,
        index_type: IndexType,
        metadata: Mapping[str, object],
    ) -> EmbeddingModelRecord:
        del session
        records = self.embedding_model_registry.get_registered_models_from_db(
            backend_type=self.backend_type,
            model_name=model_name,
        )
        if not records:
            raise ValueError(f"Embedding model '{model_name}' is not registered in the FAISS backend.")
        if len(records) > 1:
            raise ValueError(
                f"Multiple FAISS registrations exist for model '{model_name}'. "
                "Use a unique model name before switching index type."
            )

        existing_record = records[0]
        previous_index_type = existing_record.index_type
        existing_table = self.embedding_table_cache.pop(
            (model_name, existing_record.provider_type, self.backend_type, previous_index_type),
            None,
        )

        self.embedding_model_registry.delete_model(
            provider_type=existing_record.provider_type,
            backend_type=self.backend_type,
            model_name=model_name,
            index_type=previous_index_type,
        )
        updated_record = self.embedding_model_registry.register_model(
            model_name=model_name,
            provider_type=existing_record.provider_type,
            dimensions=existing_record.dimensions,
            backend_type=self.backend_type,
            index_type=index_type,
            metadata=dict(metadata),
            storage_identifier=existing_record.storage_identifier,
        )
        if existing_table is not None:
            self.embedding_table_cache[(model_name, updated_record.provider_type, self.backend_type, index_type)] = existing_table

        self.embedding_storage_managers.pop(model_name, None)

        if previous_index_type != index_type:
            obsolete_index_dir = self.get_safe_model_dir(model_name) / f"index_{previous_index_type.value}"
            if obsolete_index_dir.exists():
                shutil.rmtree(obsolete_index_dir)

        return updated_record

    def _validate_storage_consistency(
        self,
        *,
        session: Session,
        model_name: str,
        index_type: IndexType,
        storage_manager: EmbeddingStorageManager,
    ) -> None:
        embedding_table = self.get_embedding_table(model_name=model_name, index_type=index_type)
        registry_count = session.scalar(select(func.count()).select_from(embedding_table))
        storage_count = storage_manager.get_count()
        if registry_count != storage_count:
            raise RuntimeError(
                f"FAISS storage is inconsistent for model '{model_name}': SQL registry has "
                f"{registry_count} concept_ids but HDF5 storage has {storage_count} vectors. "
                "Re-run with `--overwrite-model-registration` or rebuild after cleanup."
            )

    def register_model(
        self,
        engine: Engine,
        model_name: str,
        dimensions: int,
        *,
        provider_type: ProviderType,
        index_type: IndexType,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> EmbeddingModelRecord:
        record = super().register_model(
            engine=engine,
            model_name=model_name,
            provider_type=provider_type,
            dimensions=dimensions,
            index_type=index_type,
            metadata=metadata,
        )
        self.get_safe_model_dir(model_name).mkdir(parents=True, exist_ok=True)
        return record

    def delete_model(
        self,
        *,
        engine: Engine,
        session: Session,
        model_name: str,
        provider_type: ProviderType,
    ) -> bool:
        deleted = super().delete_model(
            engine=engine,
            session=session,
            model_name=model_name,
            provider_type=provider_type,
        )
        self.embedding_storage_managers.pop(model_name, None)
        model_dir = self.get_safe_model_dir(model_name)
        if model_dir.exists():
            shutil.rmtree(model_dir)
        return deleted

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
        concept_id_tuple = tuple(concept_ids)
        storage_manager = self.get_storage_manager(
            model_name=model_name,
            dimensions=_model_record.dimensions,
            index_type=_model_record.index_type,
            metadata=_model_record.metadata,
        )
        self._validate_storage_consistency(
            session=session,
            model_name=model_name,
            index_type=index_type,
            storage_manager=storage_manager,
        )
        self.validate_embeddings_and_concept_ids(
            concept_ids=concept_id_tuple,
            embeddings=embeddings,
            dimensions=_model_record.dimensions,
        )
        previous_storage_count = storage_manager.get_count()
        storage_manager.append(
            concept_ids=np.array(concept_id_tuple, dtype=np.int64),
            embeddings=embeddings,
            index_type=_model_record.index_type,
            metric_type=metric_type,
        )
        expected_storage_count = previous_storage_count + len(concept_id_tuple)
        actual_storage_count = storage_manager.get_count()
        if actual_storage_count != expected_storage_count:
            raise RuntimeError(
                f"FAISS HDF5 append failed for model '{model_name}': expected "
                f"{expected_storage_count} stored vectors after append, found "
                f"{actual_storage_count}."
            )
        add_concept_ids_to_faiss_registry(
            concept_ids=concept_id_tuple,
            session=session,
            registered_table=self.get_embedding_table(model_name=model_name, provider_type=provider_type, index_type=index_type),
        )

    @require_registered_model
    def get_nearest_concepts(
        self,
        *,
        session: Session,
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        query_embeddings: np.ndarray,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        _model_record: EmbeddingModelRecord,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        storage_manager = self.get_storage_manager(
            model_name=model_name,
            dimensions=_model_record.dimensions,
            index_type=_model_record.index_type,
            metadata=_model_record.metadata,
        )
        if storage_manager.get_count() == 0:
            return ()

        self._validate_storage_consistency(
            session=session,
            model_name=model_name,
            index_type=index_type,
            storage_manager=storage_manager,
        )
        self.validate_embeddings(embeddings=query_embeddings, dimensions=_model_record.dimensions)
        embedding_table = self.get_embedding_table(model_name=model_name, provider_type=provider_type, index_type=index_type)

        if concept_filter is None or self._filter_is_empty(concept_filter):
            permitted_concept_ids = None
            k = self.DEFAULT_K_NEAREST
        else:
            k = concept_filter.limit if concept_filter.limit is not None else self.DEFAULT_K_NEAREST
            q_permitted_concept_ids = (
                select(embedding_table.concept_id)
                .join(Concept, Concept.concept_id == embedding_table.concept_id)
            )
            q_permitted_concept_ids = concept_filter.apply(q_permitted_concept_ids)
            permitted_concept_ids = np.array(
                [int(row.concept_id) for row in session.execute(q_permitted_concept_ids)],
                dtype=np.int64,
            )

        distances, concept_ids = storage_manager.search(
            query_vector=query_embeddings,
            metric_type=metric_type,
            index_type=_model_record.index_type,
            k=k,
            subset_concept_ids=permitted_concept_ids,
        )

        returned_ids = tuple(
            int(concept_id)
            for concept_id in np.unique(concept_ids)
            if int(concept_id) != -1
        )
        if not returned_ids:
            return tuple(() for _ in range(query_embeddings.shape[0]))

        permitted_concept_ids_storage = {
            row.concept_id: row
            for row in session.execute(
                q_concept_ids_with_embeddings(
                    embedding_table=embedding_table,
                    concept_filter=EmbeddingConceptFilter(concept_ids=returned_ids),
                    limit=None,
                )
            )
        }

        matches = []
        for concept_ids_per_query, distances_per_query in zip(concept_ids, distances):
            matches_per_query = []
            for concept_id, distance in zip(concept_ids_per_query, distances_per_query):
                if concept_id == -1:
                    continue
                row = permitted_concept_ids_storage.get(int(concept_id))
                if row is None:
                    continue
                matches_per_query.append(
                    NearestConceptMatch(
                        concept_id=int(concept_id),
                        concept_name=row.concept_name,
                        similarity=get_similarity_from_distance(float(distance), metric_type),
                        is_standard=bool(row.is_standard),
                        is_active=bool(row.is_active),
                    )
                )
            matches.append(tuple(matches_per_query))

        matches_tuple = tuple(matches)
        self.validate_nearest_concepts_output(matches_tuple, k, query_embeddings=query_embeddings)
        return matches_tuple

    def rebuild_model_indexes(
        self,
        *,
        session: Session,
        model_name: str,
        metric_types: Optional[Sequence[MetricType]] = None,
        batch_size: int = 100_000,
    ) -> None:
        records = self.embedding_model_registry.get_registered_models_from_db(
            backend_type=self.backend_type,
            model_name=model_name,
        )
        if not records:
            raise ValueError(f"Embedding model '{model_name}' is not registered in the FAISS backend.")
        if len(records) > 1:
            raise ValueError(
                f"Multiple FAISS registrations exist for model '{model_name}'. "
                "Use a unique model name before rebuilding indexes."
            )
        model_record = records[0]
        storage_manager = self.get_storage_manager(
            model_name=model_name,
            dimensions=model_record.dimensions,
            index_type=model_record.index_type,
            metadata=model_record.metadata,
        )
        self._validate_storage_consistency(
            session=session,
            model_name=model_name,
            index_type=model_record.index_type,
            storage_manager=storage_manager,
        )
        metrics = tuple(metric_types) if metric_types is not None else get_supported_metrics_for_backend_index(
            self.backend_type,
            model_record.index_type,
        )
        for metric_type in metrics:
            storage_manager.rebuild_index(
                index_type=model_record.index_type,
                metric_type=metric_type,
                batch_size=batch_size,
            )

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
        if not concept_ids:
            return {}
        storage_manager = self.get_storage_manager(
            model_name=model_name,
            dimensions=_model_record.dimensions,
            index_type=_model_record.index_type,
            metadata=_model_record.metadata,
        )
        return storage_manager.get_embeddings_by_concept_ids(
            concept_ids=np.array(tuple(concept_ids), dtype=np.int64)
        )

    @staticmethod
    def _filter_is_empty(concept_filter: EmbeddingConceptFilter) -> bool:
        return (
            concept_filter.concept_ids is None
            and concept_filter.domains is None
            and concept_filter.vocabularies is None
            and not concept_filter.require_standard
        )
