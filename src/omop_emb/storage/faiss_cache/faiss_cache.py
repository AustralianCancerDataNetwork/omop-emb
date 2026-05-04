"""FAISS sidecar cache for the pgvector backend.

``FAISSCache`` is **not** a storage backend.  It is a read-acceleration layer
that exports vectors from the pgvector source of truth into an on-disk FAISS
index.  It is entirely optional: ``faiss-cpu`` and ``h5py`` are declared as
optional dependencies in ``pyproject.toml``.

Metadata contract
-----------------
Staleness metadata is stored in the embedding registry's ``details`` JSON
under the reserved key :const:`~omop_emb.storage.index_config.FAISS_CACHE_METADATA_KEY`::

    {
        "faiss_cache": {
            "exported_at":  "<ISO-8601 timestamp, UTC>",
            "row_count":    <int>,
            "cache_dir":    "<absolute path>",
            "metric_types": ["cosine", "l2"],
            "index_params": {<serialised IndexConfig dict>}
        }
    }

This lives **alongside** ``"index_config"`` in the same JSON blob and is
written atomically by :meth:`FAISSCache.export` via
:meth:`~omop_emb.storage.postgres.pg_registry.PostgresRegistryManager.patch_model_metadata_key`.
Users must not set ``"faiss_cache"`` in external metadata passed to
``register_model``.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

from omop_emb.config import IndexType, MetricType, ProviderType
from omop_emb.storage.index_config import FAISS_CACHE_METADATA_KEY
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter, NearestConceptMatch

if TYPE_CHECKING:
    from omop_emb.storage.postgres.pg_backend import PGVectorEmbeddingBackend
    from omop_emb.storage.postgres.pg_registry import EmbeddingModelRecord

logger = logging.getLogger(__name__)


def _require_faiss():
    try:
        import faiss  # noqa: F401
        import h5py  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "FAISSCache requires the 'faiss' optional dependency. "
            "Install it with: pip install omop-emb[faiss]"
        ) from exc


class FAISSCache:
    """In-memory FAISS acceleration cache backed by a pgvector source of truth.

    Exports embeddings from a ``PGVectorEmbeddingBackend`` into FAISS indices
    stored on disk.  Subsequent searches use FAISS directly (no SQL round-trip)
    while the pgvector backend remains the authoritative store.

    Parameters
    ----------
    backend : PGVectorEmbeddingBackend
        The pgvector backend that owns the embeddings.
    model_name : str
        Registered canonical model name.
    provider_type : ProviderType
        Provider the model was registered with.
    index_type : IndexType
        Index type the model was registered with.
    cache_dir : Path | str
        Local directory where FAISS index files and HDF5 export are stored.
        Created on first :meth:`export` call.
    """

    def __init__(
        self,
        backend: "PGVectorEmbeddingBackend",
        model_name: str,
        provider_type: ProviderType,
        index_type: IndexType,
        cache_dir: Path | str,
    ) -> None:
        _require_faiss()
        self._backend = backend
        self._model_name = model_name
        self._provider_type = provider_type
        self._index_type = index_type
        self._cache_dir = Path(cache_dir).expanduser().resolve()

        self._model_record: Optional["EmbeddingModelRecord"] = None
        self._index_managers: dict[MetricType, object] = {}  # lazy-loaded

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_record(self) -> "EmbeddingModelRecord":
        if self._model_record is None:
            record = self._backend.get_registered_model(
                model_name=self._model_name,
                provider_type=self._provider_type,
                index_type=self._index_type,
            )
            if record is None:
                raise ValueError(
                    f"Model '{self._model_name}' is not registered in the backend."
                )
            self._model_record = record
        return self._model_record

    @property
    def cache_metadata(self) -> Optional[dict]:
        """Return the ``"faiss_cache"`` sub-dict from the registry, or ``None``."""
        meta = self.model_record.metadata
        return meta.get(FAISS_CACHE_METADATA_KEY) if meta else None  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Staleness
    # ------------------------------------------------------------------

    def is_stale(self) -> bool:
        """Return ``True`` if the FAISS cache is absent or out-of-date.

        The cache is considered stale when:
        * It has never been exported (``cache_metadata`` is ``None``).
        * The cached row count does not match the current pgvector row count.
        * The ``cache_dir`` no longer exists on disk.
        """
        meta = self.cache_metadata
        if meta is None:
            return True
        if not Path(meta.get("cache_dir", "")).exists():
            return True
        # Compare row counts between pgvector and the last export snapshot
        current_count = self._current_row_count()
        cached_count = meta.get("row_count", -1)
        return current_count != cached_count

    def staleness_info(self) -> dict:
        """Return a summary dict describing the current staleness state."""
        meta = self.cache_metadata or {}
        return {
            "is_stale": self.is_stale(),
            "exported_at": meta.get("exported_at"),
            "cached_row_count": meta.get("row_count"),
            "current_row_count": self._current_row_count(),
            "cache_dir": meta.get("cache_dir"),
            "metric_types": meta.get("metric_types", []),
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(
        self,
        metric_types: Sequence[MetricType],
        batch_size: int = 100_000,
    ) -> None:
        """Export embeddings from pgvector to local FAISS indices.

        Streams all concept IDs and their embeddings from the pgvector table,
        writes them to an HDF5 file, then builds FAISS indices (one per
        metric type).  Updates the registry with fresh staleness metadata.

        Parameters
        ----------
        metric_types : Sequence[MetricType]
            Metrics for which FAISS indices should be built.
        batch_size : int
            Number of rows fetched per SQL round-trip.
        """
        from omop_emb.storage.faiss_cache.faiss_export import FaissExporter

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        record = self.model_record

        exporter = FaissExporter(
            cache_dir=self._cache_dir,
            dimensions=record.dimensions,
            index_config=record.index_config,
        )

        logger.info(
            f"Exporting embeddings for '{self._model_name}' "
            f"to FAISS cache at '{self._cache_dir}'…"
        )

        # Stream concept IDs + embeddings from pgvector
        all_concept_ids, total_rows = exporter.export_from_backend(
            backend=self._backend,
            model_record=record,
            metric_types=list(metric_types),
            batch_size=batch_size,
        )

        # Persist staleness metadata into registry
        export_meta = {
            "exported_at": datetime.now(tz=timezone.utc).isoformat(),
            "row_count": total_rows,
            "cache_dir": str(self._cache_dir),
            "metric_types": [m.value for m in metric_types],
            "index_params": record.index_config.to_dict(),
        }
        self._backend._registry.patch_model_metadata_key(
            model_name=self._model_name,
            provider_type=self._provider_type,
            backend_type=self._backend.backend_type,
            index_type=self._index_type,
            key=FAISS_CACHE_METADATA_KEY,
            value=export_meta,
        )
        # Invalidate local model record cache so next access picks up fresh metadata
        self._model_record = None
        self._index_managers.clear()

        logger.info(
            f"FAISS export complete: {total_rows:,} vectors, "
            f"metrics={[m.value for m in metric_types]}."
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embeddings,
        metric_type: MetricType,
        concept_filter: Optional[EmbeddingConceptFilter] = None,
    ) -> Tuple[Tuple[NearestConceptMatch, ...], ...]:
        """Search the local FAISS index and enrich results from CDM.

        Parameters
        ----------
        query_embeddings : ndarray
            Shape ``(Q, D)``.
        metric_type : MetricType
            Must have been included in the last :meth:`export` call.
        concept_filter : EmbeddingConceptFilter, optional
            Limit and optional CDM-side filters.

        Returns
        -------
        Tuple[Tuple[NearestConceptMatch, ...], ...]
            Shape ``(Q, k)``.
        """
        from omop_emb.storage.faiss_cache.faiss_export import FaissExporter
        import numpy as np

        record = self.model_record
        k = (concept_filter.limit if concept_filter else None) or self._backend.DEFAULT_K_NEAREST

        self._backend.validate_embeddings(query_embeddings, record.dimensions)

        exporter = FaissExporter(
            cache_dir=self._cache_dir,
            dimensions=record.dimensions,
            index_config=record.index_config,
        )

        # Pre-filter via CDM if needed
        candidate_ids = self._backend._get_candidate_concept_ids_from_cdm(concept_filter)

        distances, concept_ids_matrix = exporter.search(
            query_embeddings=np.asarray(query_embeddings, dtype=np.float32),
            metric_type=metric_type,
            k=k,
            candidate_concept_ids=candidate_ids,
        )

        # Fetch CDM metadata for all result concept IDs
        flat_ids = {int(cid) for row in concept_ids_matrix for cid in row if cid != -1}
        concept_metadata = self._backend._fetch_concept_metadata(flat_ids)

        from omop_emb.utils.embedding_utils import get_similarity_from_distance

        results: list[list[NearestConceptMatch]] = [[] for _ in range(len(query_embeddings))]
        for q_idx, (dist_row, id_row) in enumerate(zip(distances, concept_ids_matrix)):
            for dist, cid in zip(dist_row, id_row):
                if cid == -1:
                    continue
                meta = concept_metadata.get(int(cid))
                if meta is None:
                    continue
                similarity = get_similarity_from_distance(float(dist), metric_type)
                results[q_idx].append(
                    NearestConceptMatch(
                        concept_id=int(cid),
                        concept_name=meta.concept_name,
                        similarity=float(similarity),
                        is_standard=meta.standard_concept in ("S", "C"),
                        is_active=meta.invalid_reason not in ("D", "U"),
                    )
                )

        return tuple(tuple(r) for r in results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_row_count(self) -> int:
        """Return the number of vectors currently stored in pgvector for this model."""
        from sqlalchemy import func, select

        table = self._backend.get_embedding_table(
            model_name=self._model_name,
            provider_type=self._provider_type,
            index_type=self._index_type,
        )
        with self._backend.emb_session_factory() as session:
            return session.scalar(select(func.count()).select_from(table)) or 0
