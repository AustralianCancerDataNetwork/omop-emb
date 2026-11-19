"""Read-only access to a pgvector store, and the queries the benchmark sends it."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from oa_configurator import Resolver
from omop_llm import build_model_backend_from_resolved
from omop_llm.providers import canonical_model_name
from sqlalchemy import Engine, event, func, select, text
from sqlalchemy.orm import Session

from omop_emb.backends import open_vector_store_reader, open_vector_store_writer
from omop_emb.backends.embedding_table import PGEmbeddingTable
from omop_emb.backends.index_config import HNSWIndexConfig
from omop_emb.backends.pgvector import PGVectorEmbeddingBackend
from omop_emb.backends.pgvector.pg_sql import get_distance, query_nearest_concept_ids
from omop_emb.config import BackendType, MetricType, OmopEmbConfig, VectorPrecision
from omop_emb.model_registry import EmbeddingModelRecord
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter

_READ_ONLY_STATEMENTS = ("select", "set local", "explain", "with", "show", "prepare", "deallocate")


def guard_read_only(engine: Engine) -> None:
    """Make engine refuse every statement that is not a read or a transaction-local setting."""

    @event.listens_for(engine, "before_cursor_execute")
    def _only_reads(_conn, _cursor, statement, _parameters, _context, _many) -> None:
        if not statement.lstrip().lower().startswith(_READ_ONLY_STATEMENTS):
            raise RuntimeError(f"Blocked non-read-only SQL: {statement[:80]!r}")


@dataclass(frozen=True)
class Setting:
    """Query-time HNSW settings of one run."""

    ef_search: int
    max_scan_tuples: int | None = None

    @property
    def name(self) -> str:
        cap = "default" if self.max_scan_tuples is None else str(self.max_scan_tuples)
        return f"ef{self.ef_search}/cap{cap}"


@dataclass
class Store:
    """A registered pgvector model opened for the benchmark."""

    backend: PGVectorEmbeddingBackend
    record: EmbeddingModelRecord
    table: type[PGEmbeddingTable]
    storage_precision: VectorPrecision

    @property
    def engine(self) -> Engine:
        return self.backend.emb_engine

    @property
    def metric(self) -> MetricType:
        return self.record.metric_type or MetricType.COSINE

    @property
    def index_config(self) -> HNSWIndexConfig | None:
        config = self.record.index_config
        return config if isinstance(config, HNSWIndexConfig) else None

    @property
    def inlines_single_domain(self) -> bool:
        """Whether the production query path writes a single domain into the statement."""
        config = self.index_config
        return config is not None and config.partial_domain_indexes

    def search(
        self,
        vector: np.ndarray,
        concept_filter: EmbeddingConceptFilter | None,
        k: int,
        setting: Setting,
    ) -> tuple[list[tuple[int, float]], float]:
        """The approximate nearest concepts as the production query path finds them.

        Returns
        -------
        tuple[list[tuple[int, float]], float]
            ``(concept_id, distance)`` pairs closest first, and the milliseconds the
            settings and the query took on an open connection.
        """
        config = self.index_config
        with Session(self.engine) as session:
            session.execute(text("SET LOCAL transaction_read_only = on"))
            started = time.perf_counter()
            if config is not None:
                self.backend._configure_hnsw_scan(
                    session,
                    ef_search=setting.ef_search,
                    k=k,
                    filtered=concept_filter is not None,
                    max_scan_tuples=setting.max_scan_tuples,
                )
            rows = query_nearest_concept_ids(
                session,
                self.table,
                vector,
                self.metric,
                k,
                concept_filter=concept_filter,
                over_half_cast=config is not None and config.casts_storage(self.storage_precision),
                inline_single_domain=self.inlines_single_domain,
            )
            elapsed = (time.perf_counter() - started) * 1000
            ordered = sorted(
                ((int(r.concept_id), float(r.distance)) for r in rows), key=lambda p: p[1]
            )
            return ordered, elapsed

    def exact(
        self, vector: np.ndarray, concept_filter: EmbeddingConceptFilter | None, k: int
    ) -> list[tuple[int, float]]:
        """The exact nearest concepts, by a scan of every row that passes the filter."""
        with Session(self.engine) as session:
            session.execute(text("SET LOCAL transaction_read_only = on"))
            session.execute(text("SET LOCAL enable_indexscan = off"))
            session.execute(text("SET LOCAL enable_bitmapscan = off"))
            rows = query_nearest_concept_ids(
                session, self.table, vector, self.metric, k, concept_filter=concept_filter
            )
            return [(int(r.concept_id), float(r.distance)) for r in rows]

    def exact_distances(self, vector: np.ndarray, concept_ids: list[int]) -> dict[int, float]:
        """Full-precision distance of the stored vectors of concept_ids to vector."""
        if not concept_ids:
            return {}
        distance = get_distance(self.table, vector, self.metric)
        with Session(self.engine) as session:
            session.execute(text("SET LOCAL transaction_read_only = on"))
            rows = session.execute(
                select(self.table.concept_id, distance).where(
                    self.table.concept_id.in_(concept_ids)
                )
            )
            return {int(cid): float(d) for cid, d in rows}

    def row_count(self) -> int:
        with Session(self.engine) as session:
            return int(session.execute(select(func.count(self.table.concept_id))).scalar_one())


def resolve_model_backend(model_name: str | None):
    """The embedding model backend of a ``[models.*]`` entry, or of the configured model."""
    cfg = OmopEmbConfig.get_config()
    resolved_model = Resolver.from_active_config().resolve_model(
        model_name or cfg.embedding_model_name
    )
    return build_model_backend_from_resolved(resolved_model)


def open_store(model_name: str | None, *, writable: bool = False) -> Store:
    """Open the configured pgvector store and the registered model.

    Parameters
    ----------
    model_name : str, optional
        Name of a ``[models.*]`` entry. Defaults to the configured model.
    writable : bool
        Open the store for writing, which only ``build`` and ``drop`` do. A
        read-only store also refuses writes at the engine.

    Returns
    -------
    Store
        The registered model with its table.
    """
    cfg = OmopEmbConfig.get_config()
    resolver = Resolver.from_active_config()
    resolved_store = resolver.resolve_vector_store(cfg.vector_store_name)
    if BackendType(resolved_store.backend_type) != BackendType.PGVECTOR:
        raise RuntimeError("The recall benchmark needs a pgvector vector store.")
    model_backend = resolve_model_backend(model_name)
    backend = (
        open_vector_store_writer(resolved_store)
        if writable
        else open_vector_store_reader(resolved_store)
    )
    if not isinstance(backend, PGVectorEmbeddingBackend):
        raise TypeError("The recall benchmark needs a pgvector vector store.")
    if not writable:
        guard_read_only(backend.emb_engine)
    canonical = canonical_model_name(model_backend.provider, model_backend.model)
    record = backend.get_registered_model(model_name=canonical)
    if record is None:
        raise RuntimeError(f"No registered model '{canonical}' in the vector store.")
    store = Store(
        backend=backend,
        record=record,
        table=backend._get_storage_table_descriptor(record),
        storage_precision=backend.storage_precision_of(record),
    )
    return store
