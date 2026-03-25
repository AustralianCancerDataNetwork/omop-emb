from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, cast, Type

import numpy as np
from numpy import ndarray
from omop_emb.backends.config import BackendType
from sqlalchemy import Engine, delete, insert, select
from sqlalchemy.orm import Session
import logging
from dataclasses import dataclass, field

from omop_alchemy.cdm.model.vocabulary import Concept
from omop_emb.backends.registry import ModelRegistry

from ..errors import EmbeddingBackendConfigurationError
from .faiss_sql import (
    FAISSConceptIDEmbeddingMapping, 
    create_faiss_mapping_table, 
    initialise_faiss_mapping_tables
)
from ..base import (
    EmbeddingBackend,
    EmbeddingBackendCapabilities,
    EmbeddingConceptFilter,
    EmbeddingIndexConfig,
    EmbeddingModelRecord,
    NearestConceptMatch,
    require_registered_model
)

logger = logging.getLogger(__name__)

@dataclass
class FaissMetadata:
    """Strongly typed metadata required for the FAISS backend to operate."""
    index_file_path: str
    metadata_bucket: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes to a flat dictionary for SQLAlchemy JSON storage."""
        # We flatten it so it looks clean in the database
        result = self.metadata_bucket.copy()
        result["index_file_path"] = self.index_file_path
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FaissMetadata":
        """Safely reconstructs the object from the database JSON."""
        data_copy = dict(data)
        
        # Pop the required FAISS fields, raising a clear error if missing
        try:
            index_path = data_copy.pop("index_file_path")
        except KeyError as e:
            raise ValueError(
                f"Corrupted FAISS metadata in registry. Missing required key: {e}"
            )

        # Everything else goes back into user_metadata
        return cls(
            index_file_path=str(index_path),
            metadata_bucket=data_copy
        )


class FaissEmbeddingBackend(EmbeddingBackend[FAISSConceptIDEmbeddingMapping]):
    """
    File-based FAISS embedding backend.

    First-pass design
    -----------------
    - Registry is stored as JSON under a configurable base directory.
    - Embeddings are stored as ``.npy`` files alongside a FAISS index.
    - OMOP concept metadata and filter application still use SQLAlchemy.
    - Nearest-neighbor search uses FAISS when possible and falls back to
      in-memory cosine ranking for filtered subsets.

    Notes
    -----
    This backend is intentionally conservative and optimized for clarity over
    extreme scale. In particular, updates currently rewrite the per-model numpy
    arrays and rebuild the FAISS index, which is acceptable for a first pass
    but not ideal for very large incremental workloads.
    """

    DEFAULT_FAISS_DIR = ".omop_emb/faiss"
    CONCEPT_IDS_FILE = "concept_ids.npy"
    EMBEDDINGS_FILE = "embeddings.npy"
    INDEX_FILE = "index.faiss"

    def __init__(self, base_dir: Optional[str | os.PathLike[str]] = None):
        self.base_dir = Path(
            base_dir
            or os.getenv("OMOP_EMB_FAISS_DIR")
            or FaissEmbeddingBackend.DEFAULT_FAISS_DIR
        ).expanduser()

    @property
    def backend_type(self) -> BackendType:
        return BackendType.FAISS

    @property
    def capabilities(self) -> EmbeddingBackendCapabilities:
        return EmbeddingBackendCapabilities(
            stores_embeddings=True,
            supports_incremental_upsert=True,
            supports_nearest_neighbor_search=True,
            supports_server_side_similarity=False,
            supports_filter_by_concept_ids=True,
            supports_filter_by_domain=True,
            supports_filter_by_vocabulary=True,
            supports_filter_by_standard_flag=True,
            requires_explicit_index_refresh=False,
        )

    def initialise_store(self, engine) -> None:
        self.base_dir.mkdir(parents=False, exist_ok=True)
        return initialise_faiss_mapping_tables(engine, model_cache=self.model_cache)

    def _create_storage_table(self, engine: Engine, entry: ModelRegistry) -> Type[FAISSConceptIDEmbeddingMapping]:
        return create_faiss_mapping_table(engine=engine, model_registry_entry=entry)

    def register_model(
        self,
        engine: Engine,
        model_name: str,
        dimensions: int,
        *,
        index_config: EmbeddingIndexConfig,
        metadata: Mapping[str, object] = {},
    ) -> EmbeddingModelRecord:

        # Create the storage for the model on disk
        safe_name = self.safe_model_name(model_name)
        model_dir = self.base_dir / safe_name
        model_dir.mkdir(parents=False, exist_ok=True)
        logger.info(f"Created model directory: {model_dir}")

        faiss_metadata = FaissMetadata(
            index_file_path=str(self._index_path(model_name)),
            metadata_bucket=dict(metadata),
        )
        return super().register_model(
            engine=engine,
            model_name=model_name,
            dimensions=dimensions,
            index_config=index_config,
            metadata=faiss_metadata.to_dict(),
        )

    @require_registered_model
    def upsert_embeddings(
        self,
        session: Session,
        model_name: str,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
        embeddings: ndarray,
    ) -> None:
        concept_id_array = np.asarray(tuple(concept_ids), dtype=np.int64)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected a 2D embeddings array, got ndim={embeddings.ndim}.")
        if concept_id_array.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch between #concept_ids ({concept_id_array.shape[0]}) and "
                f"#embeddings ({embeddings.shape[0]})."
            )
        if embeddings.shape[1] != model_record.dimensions:
            raise ValueError(
                f"Embedding dimension mismatch for model '{model_name}': "
                f"expected {model_record.dimensions}, got {embeddings.shape[1]}."
            )

        existing_ids, existing_embeddings = self._load_embeddings(session, model_record)
        merged: dict[int, np.ndarray] = {}
        for cid, emb in zip(existing_ids.tolist(), existing_embeddings):
            merged[int(cid)] = emb.astype(np.float32, copy=False)
        for cid, emb in zip(concept_id_array.tolist(), embeddings):
            merged[int(cid)] = np.asarray(emb, dtype=np.float32)

        if merged:
            sorted_ids = np.array(sorted(merged.keys()), dtype=np.int64)
            matrix = np.vstack([merged[int(cid)] for cid in sorted_ids]).astype(np.float32)
        else:
            sorted_ids = np.empty((0,), dtype=np.int64)
            matrix = np.empty((0, model_record.dimensions), dtype=np.float32)

        self._write_embeddings(model_record, matrix)
        self._sync_mapping_table(session, model_name, sorted_ids)
        self._rebuild_index(model_record, matrix)

    @require_registered_model
    def get_embeddings_by_concept_ids(
        self,
        session: Session,
        model_name: str,
        model_record: EmbeddingModelRecord,
        concept_ids: Sequence[int],
    ) -> Mapping[int, Sequence[float]]:
        if not concept_ids:
            return {}
        matrix = self._load_embedding_matrix(model_record)
        if matrix.shape[0] == 0:
            return {}

        mapping_table = self._get_embedding_table(session, model_name)
        mapping_rows = session.execute(
            select(mapping_table.concept_id, mapping_table.index_position).where(
                mapping_table.concept_id.in_(tuple(int(cid) for cid in concept_ids))
            )
        ).all()

        result: dict[int, Sequence[float]] = {}
        for row in mapping_rows:
            row_idx = int(row.index_position)
            if 0 <= row_idx < matrix.shape[0]:
                result[int(row.concept_id)] = matrix[row_idx].astype(float).tolist()
        return result

    @require_registered_model
    def get_similarities(
        self,
        session: Session,
        model_name: str,
        model_record: EmbeddingModelRecord,
        query_embedding: Sequence[float],
        *,
        concept_ids: Optional[Sequence[int]] = None,
    ) -> Mapping[int, float]:
        indexed_ids, matrix = self._load_embeddings(session, model_record)
        if indexed_ids.size == 0:
            return {}

        query = self._normalize_matrix(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))
        if concept_ids is not None:
            allowed = {int(cid) for cid in concept_ids}
            rows = [idx for idx, cid in enumerate(indexed_ids.tolist()) if int(cid) in allowed]
            if not rows:
                return {}
            selected_ids = indexed_ids[rows]
            selected_matrix = matrix[rows]
        else:
            selected_ids = indexed_ids
            selected_matrix = matrix

        similarities = (self._normalize_matrix(selected_matrix) @ query.T).reshape(-1)
        return {
            int(cid): float(score)
            for cid, score in zip(selected_ids.tolist(), similarities.tolist())
        }

    @require_registered_model
    def get_nearest_concepts(
        self,
        session: Session,
        model_name: str,
        model_record: EmbeddingModelRecord,
        query_embedding: Sequence[float],
        *,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: int = 10,
    ) -> Sequence[NearestConceptMatch]:
        concept_filter = concept_filter or EmbeddingConceptFilter()
        indexed_ids, matrix = self._load_embeddings(session, model_record)
        if indexed_ids.size == 0:
            return ()

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)

        if self._filter_is_empty(concept_filter):
            matched_positions, matched_scores = self._search_faiss(
                model_record=model_record,
                query=query,
                limit=limit,
            )
            matched_ids_with_scores = self._lookup_concept_ids_by_positions(
                session=session,
                model_name=model_name,
                positions=matched_positions,
                scores=matched_scores,
            )
            matched_ids = [concept_id for concept_id, _ in matched_ids_with_scores]
            matched_scores = [score for _, score in matched_ids_with_scores]
        else:
            eligible_ids = self._get_filtered_concept_ids(session, concept_filter)
            if not eligible_ids:
                return ()
            matched_ids, matched_scores = self._rank_subset(
                indexed_ids=indexed_ids,
                matrix=matrix,
                query=query,
                eligible_ids=eligible_ids,
                limit=limit,
            )

        if not matched_ids:
            return ()

        concept_rows = session.execute(
            select(
                Concept.concept_id,
                Concept.concept_name,
                Concept.standard_concept,
                Concept.invalid_reason,
            ).where(Concept.concept_id.in_(tuple(matched_ids)))
        ).all()
        concept_by_id = {
            int(row.concept_id): row
            for row in concept_rows
        }

        results: list[NearestConceptMatch] = []
        for concept_id, similarity in zip(matched_ids, matched_scores):
            row = concept_by_id.get(int(concept_id))
            if row is None:
                continue
            results.append(
                NearestConceptMatch(
                    concept_id=int(row.concept_id),
                    concept_name=row.concept_name,
                    similarity=float(similarity),
                    is_standard=row.standard_concept in {"S", "C"},
                    is_active=row.invalid_reason not in {"D", "U"},
                )
            )
        return tuple(results)

    @require_registered_model
    def refresh_model_index(self, session: Session, model_name: str, model_record: EmbeddingModelRecord) -> None:
        matrix = self._load_embedding_matrix(model_record)
        self._rebuild_index(model_record, matrix)

    def _require_faiss(self):
        try:
            import faiss  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "FaissEmbeddingBackend requires the optional 'faiss' package. "
                "Install faiss-cpu or faiss-gpu to use this backend."
            ) from exc
        return faiss

    def _model_dir(self, model_name: str) -> Path:
        return self.base_dir / self.safe_model_name(model_name)

    def _concept_ids_path(self, model_name: str) -> Path:
        return self._model_dir(model_name) / FaissEmbeddingBackend.CONCEPT_IDS_FILE

    def _embeddings_path(self, model_name: str) -> Path:
        return self._model_dir(model_name) / FaissEmbeddingBackend.EMBEDDINGS_FILE

    def _index_path(self, model_name: str) -> Path:
        return self._model_dir(model_name) / FaissEmbeddingBackend.INDEX_FILE

    def _load_embedding_matrix(self, model_record: EmbeddingModelRecord) -> np.ndarray:
        embeddings_path = self._embeddings_path(model_record.model_name)
        if not embeddings_path.exists():
            return np.empty((0, model_record.dimensions), dtype=np.float32)

        embeddings = np.load(embeddings_path).astype(np.float32, copy=False)
        if embeddings.ndim != 2:
            raise ValueError(f"Stored embeddings for model '{model_record.model_name}' are not a 2D matrix.")
        return embeddings

    def _load_embeddings(self, session: Session, model_record: EmbeddingModelRecord) -> tuple[np.ndarray, np.ndarray]:
        matrix = self._load_embedding_matrix(model_record)
        if matrix.shape[0] == 0:
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0, model_record.dimensions), dtype=np.float32),
            )

        mapping_table = self._get_embedding_table(session, model_record.model_name)
        mapping_rows = session.execute(
            select(mapping_table.concept_id, mapping_table.index_position).order_by(mapping_table.index_position)
        ).all()

        concept_ids = np.asarray([int(row.concept_id) for row in mapping_rows], dtype=np.int64)
        return concept_ids, matrix

    def _write_embeddings(
        self,
        model_record: EmbeddingModelRecord,
        embeddings: np.ndarray,
    ) -> None:
        model_dir = self._model_dir(model_record.model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        np.save(self._embeddings_path(model_record.model_name), embeddings.astype(np.float32, copy=False))

    def _sync_mapping_table(
        self,
        session: Session,
        model_name: str,
        concept_ids: np.ndarray,
    ) -> None:
        mapping_table = self._get_embedding_table(session, model_name)
        session.execute(delete(mapping_table))
        if concept_ids.size == 0:
            return

        rows = [
            {
                "concept_id": int(concept_id),
                "index_position": int(index_position),
            }
            for index_position, concept_id in enumerate(concept_ids.tolist())
        ]
        session.execute(insert(mapping_table), rows)

    def _rebuild_index(
        self,
        model_record: EmbeddingModelRecord,
        embeddings: np.ndarray,
    ) -> None:
        faiss = self._require_faiss()
        model_dir = self._model_dir(model_record.model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        index_path = self._index_path(model_record.model_name)

        if embeddings.shape[0] == 0:
            if index_path.exists():
                index_path.unlink()
            return

        normalized = self._normalize_matrix(embeddings.astype(np.float32, copy=False))
        index_type = (
            model_record.index_config.index_type
            if model_record.index_config is not None and model_record.index_config.index_type
            else "IndexFlatIP"
        )
        index = self._create_faiss_index(
            faiss=faiss,
            index_type=index_type,
            dimensions=model_record.dimensions,
        )
        index.add(normalized)
        faiss.write_index(index, str(index_path))

    def _search_faiss(
        self,
        *,
        model_record: EmbeddingModelRecord,
        query: np.ndarray,
        limit: int,
    ) -> tuple[list[int], list[float]]:
        faiss = self._require_faiss()
        matrix = self._load_embedding_matrix(model_record)
        if matrix.shape[0] == 0:
            return [], []

        index_path = self._index_path(model_record.model_name)
        if not index_path.exists():
            self._rebuild_index(model_record, matrix)

        index = faiss.read_index(str(index_path))
        normalized_query = self._normalize_matrix(query.astype(np.float32, copy=False))
        distances, indices = index.search(normalized_query, min(limit, matrix.shape[0]))
        matched_positions: list[int] = []
        matched_scores: list[float] = []
        for idx, score in zip(indices[0].tolist(), distances[0].tolist()):
            if idx < 0:
                continue
            matched_positions.append(int(idx))
            matched_scores.append(float(score))
        return matched_positions, matched_scores

    def _lookup_concept_ids_by_positions(
        self,
        *,
        session: Session,
        model_name: str,
        positions: Sequence[int],
        scores: Sequence[float],
    ) -> list[tuple[int, float]]:
        if not positions:
            return []

        mapping_table = self._get_embedding_table(session, model_name)
        rows = session.execute(
            select(mapping_table.concept_id, mapping_table.index_position).where(
                mapping_table.index_position.in_(tuple(int(pos) for pos in positions))
            )
        ).all()
        concept_by_position = {int(row.index_position): int(row.concept_id) for row in rows}
        return [
            (concept_by_position[pos], float(score))
            for pos, score in zip(positions, scores)
            if pos in concept_by_position
        ]

    def _rank_subset(
        self,
        *,
        indexed_ids: np.ndarray,
        matrix: np.ndarray,
        query: np.ndarray,
        eligible_ids: Sequence[int],
        limit: int,
    ) -> tuple[list[int], list[float]]:
        allowed = {int(cid) for cid in eligible_ids}
        selected_rows = [idx for idx, cid in enumerate(indexed_ids.tolist()) if int(cid) in allowed]
        if not selected_rows:
            return [], []

        selected_ids = indexed_ids[selected_rows]
        selected_matrix = matrix[selected_rows]
        scores = (self._normalize_matrix(selected_matrix) @ self._normalize_matrix(query).T).reshape(-1)
        top_idx = np.argsort(scores)[::-1][:limit]
        return (
            [int(selected_ids[idx]) for idx in top_idx.tolist()],
            [float(scores[idx]) for idx in top_idx.tolist()],
        )

    def _get_filtered_concept_ids(
        self,
        session: Session,
        concept_filter: EmbeddingConceptFilter,
    ) -> tuple[int, ...]:
        stmt = select(Concept.concept_id)
        if concept_filter.concept_ids is not None:
            stmt = stmt.where(Concept.concept_id.in_(concept_filter.concept_ids))
        if concept_filter.domains is not None:
            stmt = stmt.where(Concept.domain_id.in_(concept_filter.domains))
        if concept_filter.vocabularies is not None:
            stmt = stmt.where(Concept.vocabulary_id.in_(concept_filter.vocabularies))
        if concept_filter.require_standard:
            stmt = stmt.where(Concept.standard_concept.in_(["S", "C"]))
        return tuple(int(row.concept_id) for row in session.execute(stmt))

    @staticmethod
    def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    @staticmethod
    def _filter_is_empty(concept_filter: EmbeddingConceptFilter) -> bool:
        return (
            concept_filter.concept_ids is None
            and concept_filter.domains is None
            and concept_filter.vocabularies is None
            and not concept_filter.require_standard
        )

    @staticmethod
    def _create_faiss_index(*, faiss, index_type: str, dimensions: int):
        if index_type in {"IndexFlatIP", "flat", "flatip"}:
            return faiss.IndexFlatIP(dimensions)
        raise ValueError(
            f"Unsupported FAISS index_type={index_type!r} in first-pass backend. "
            "Currently supported: IndexFlatIP."
        )

    @staticmethod
    def _index_type_for_config(index_config: Optional[EmbeddingIndexConfig]) -> str:
        if index_config is None or not index_config.index_type or index_config.index_type == "auto":
            return "IndexFlatIP"
        return index_config.index_type
