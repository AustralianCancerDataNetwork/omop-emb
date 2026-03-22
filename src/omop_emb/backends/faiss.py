from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from numpy import ndarray
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from omop_alchemy.cdm.model.vocabulary import Concept
from omop_emb.registry import ModelRegistry, ensure_model_registry_schema

from .errors import EmbeddingBackendConfigurationError
from .base import (
    EmbeddingBackend,
    EmbeddingBackendCapabilities,
    EmbeddingConceptFilter,
    EmbeddingIndexConfig,
    EmbeddingModelRecord,
    NearestConceptMatch,
)


DEFAULT_FAISS_DIR = ".omop_emb/faiss"
CONCEPT_IDS_FILE = "concept_ids.npy"
EMBEDDINGS_FILE = "embeddings.npy"
INDEX_FILE = "index.faiss"


class FaissEmbeddingBackend(EmbeddingBackend):
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

    def __init__(self, base_dir: Optional[str | os.PathLike[str]] = None):
        self.base_dir = Path(
            base_dir
            or os.getenv("OMOP_EMB_FAISS_DIR")
            or DEFAULT_FAISS_DIR
        ).expanduser()

    @property
    def backend_name(self) -> str:
        return "faiss"

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
        ensure_model_registry_schema(engine)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_registered_models(self, session: Session) -> Sequence[EmbeddingModelRecord]:
        return tuple(
            self._record_from_registry_row(row)
            for row in session.scalars(
                select(ModelRegistry).where(ModelRegistry.backend_name == self.backend_name)
            ).all()
        )

    def get_registered_model(
        self,
        session: Session,
        model_name: str,
    ) -> Optional[EmbeddingModelRecord]:
        row = session.scalar(
            select(ModelRegistry).where(
                ModelRegistry.name == model_name,
                ModelRegistry.backend_name == self.backend_name,
            )
        )
        if row is None:
            return None
        return self._record_from_registry_row(row)

    def register_model(
        self,
        engine: Engine,
        model_name: str,
        dimensions: int,
        *,
        index_config: Optional[EmbeddingIndexConfig] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> EmbeddingModelRecord:
        self.initialise_store(engine)
        safe_name = self._safe_model_name(model_name)
        model_dir = self.base_dir / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)
        resolved_index_type = self._index_type_for_config(index_config)

        with Session(engine, expire_on_commit=False) as session:
            existing_row = session.scalar(
                select(ModelRegistry).where(ModelRegistry.name == model_name)
            )
            if existing_row is not None:
                if existing_row.backend_name != self.backend_name:
                    raise EmbeddingBackendConfigurationError(
                        f"Model '{model_name}' is already registered for backend "
                        f"'{existing_row.backend_name}', not '{self.backend_name}'."
                    )
                if existing_row.dimensions != dimensions:
                    raise EmbeddingBackendConfigurationError(
                        f"Model '{model_name}' is already registered with dimensions "
                        f"{existing_row.dimensions}, not {dimensions}."
                    )
                if existing_row.index_method != resolved_index_type:
                    raise EmbeddingBackendConfigurationError(
                        f"Model '{model_name}' is already registered with "
                        f"index_method='{existing_row.index_method}', not "
                        f"'{resolved_index_type}'. Reuse the existing model "
                        "configuration or register a new model name."
                    )
                return self._record_from_registry_row(existing_row)

            new_row = ModelRegistry(
                name=model_name,
                dimensions=dimensions,
                storage_identifier=str(model_dir),
                index_method=resolved_index_type,
                backend_name=self.backend_name,
            )
            session.add(new_row)
            session.commit()
            return self._record_from_registry_row(new_row)

    def has_any_embeddings(self, session: Session, model_name: str) -> bool:
        model_record = self._require_registered_model(session, model_name)
        concept_ids, _ = self._load_embeddings(model_record)
        return concept_ids.size > 0

    def upsert_embeddings(
        self,
        session: Session,
        model_name: str,
        concept_ids: Sequence[int],
        embeddings: ndarray,
    ) -> None:
        model_record = self._require_registered_model(session, model_name)
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

        existing_ids, existing_embeddings = self._load_embeddings(model_record)
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

        self._write_embeddings(model_record, sorted_ids, matrix)
        self._rebuild_index(model_record, matrix)

    def get_embeddings_by_concept_ids(
        self,
        session: Session,
        model_name: str,
        concept_ids: Sequence[int],
    ) -> Mapping[int, Sequence[float]]:
        if not concept_ids:
            return {}

        model_record = self._require_registered_model(session, model_name)
        indexed_ids, matrix = self._load_embeddings(model_record)
        if indexed_ids.size == 0:
            return {}
        id_to_row = {int(cid): idx for idx, cid in enumerate(indexed_ids.tolist())}
        result: dict[int, Sequence[float]] = {}
        for concept_id in concept_ids:
            row_idx = id_to_row.get(int(concept_id))
            if row_idx is not None:
                result[int(concept_id)] = matrix[row_idx].astype(float).tolist()
        return result

    def get_similarities(
        self,
        session: Session,
        model_name: str,
        query_embedding: Sequence[float],
        *,
        concept_ids: Optional[Sequence[int]] = None,
    ) -> Mapping[int, float]:
        model_record = self._require_registered_model(session, model_name)
        indexed_ids, matrix = self._load_embeddings(model_record)
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

    def get_nearest_concepts(
        self,
        session: Session,
        model_name: str,
        query_embedding: Sequence[float],
        *,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
        limit: int = 10,
    ) -> Sequence[NearestConceptMatch]:
        concept_filter = concept_filter or EmbeddingConceptFilter()
        model_record = self._require_registered_model(session, model_name)
        indexed_ids, matrix = self._load_embeddings(model_record)
        if indexed_ids.size == 0:
            return ()

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)

        if self._filter_is_empty(concept_filter):
            matched_ids, matched_scores = self._search_faiss(
                model_record=model_record,
                query=query,
                limit=limit,
            )
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

    def refresh_model_index(self, session: Session, model_name: str) -> None:
        model_record = self._require_registered_model(session, model_name)
        _, matrix = self._load_embeddings(model_record)
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
        return self.base_dir / self._safe_model_name(model_name)

    def _concept_ids_path(self, model_name: str) -> Path:
        return self._model_dir(model_name) / CONCEPT_IDS_FILE

    def _embeddings_path(self, model_name: str) -> Path:
        return self._model_dir(model_name) / EMBEDDINGS_FILE

    def _index_path(self, model_name: str) -> Path:
        return self._model_dir(model_name) / INDEX_FILE

    def _require_registered_model(
        self,
        session: Session,
        model_name: str,
    ) -> EmbeddingModelRecord:
        record = self.get_registered_model(session, model_name)
        if record is None:
            raise ValueError(f"Embedding model '{model_name}' is not registered in the FAISS backend.")
        return record

    def _record_from_registry_row(self, row: ModelRegistry) -> EmbeddingModelRecord:
        return EmbeddingModelRecord(
            model_name=row.name,
            dimensions=row.dimensions,
            backend_name=row.backend_name,
            storage_identifier=row.storage_identifier,
            index_config=EmbeddingIndexConfig(
                index_type=row.index_method,
                distance_metric="cosine",
            ),
            metadata={},
        )

    def _load_embeddings(self, model_record: EmbeddingModelRecord) -> tuple[np.ndarray, np.ndarray]:
        concept_ids_path = self._concept_ids_path(model_record.model_name)
        embeddings_path = self._embeddings_path(model_record.model_name)
        if not concept_ids_path.exists() or not embeddings_path.exists():
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0, model_record.dimensions), dtype=np.float32),
            )

        concept_ids = np.load(concept_ids_path)
        embeddings = np.load(embeddings_path).astype(np.float32, copy=False)
        if embeddings.ndim != 2:
            raise ValueError(f"Stored embeddings for model '{model_name}' are not a 2D matrix.")
        return concept_ids.astype(np.int64, copy=False), embeddings

    def _write_embeddings(
        self,
        model_record: EmbeddingModelRecord,
        concept_ids: np.ndarray,
        embeddings: np.ndarray,
    ) -> None:
        model_dir = self._model_dir(model_record.model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        np.save(self._concept_ids_path(model_record.model_name), concept_ids)
        np.save(self._embeddings_path(model_record.model_name), embeddings.astype(np.float32, copy=False))

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
        concept_ids, matrix = self._load_embeddings(model_record)
        if concept_ids.size == 0:
            return [], []

        index_path = self._index_path(model_record.model_name)
        if not index_path.exists():
            self._rebuild_index(model_record, matrix)

        index = faiss.read_index(str(index_path))
        normalized_query = self._normalize_matrix(query.astype(np.float32, copy=False))
        distances, indices = index.search(normalized_query, min(limit, concept_ids.shape[0]))
        matched_ids: list[int] = []
        matched_scores: list[float] = []
        for idx, score in zip(indices[0].tolist(), distances[0].tolist()):
            if idx < 0:
                continue
            matched_ids.append(int(concept_ids[idx]))
            matched_scores.append(float(score))
        return matched_ids, matched_scores

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
    def _safe_model_name(model_name: str) -> str:
        name = model_name.lower()
        sanitized = re.sub(r"[^\w]+", "_", name)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        return sanitized

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
