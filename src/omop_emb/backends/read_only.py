"""Read-only inspection for configured embedding stores.

Constructing an :class:`EmbeddingBackend` is intentionally a maintenance
operation: it creates the registry schema and, for pgvector, enables the
extension. Setup diagnostics must not use that constructor. This module opens
the same stores with ordinary read queries and exposes the small identity
surface needed by setup planners.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from oa_configurator import Dialect, ResolvedVectorStore
from sqlalchemy import Engine, event, inspect, select

from omop_emb.backends.base_backend import (
    EmbeddingBackend,
    resolve_backend_from_resolved_vector_store,
)
from omop_emb.backends.embedding_table import concept_metadata_table_descriptor
from omop_emb.config import BackendType, parse_backend_type
from omop_emb.model_registry import EmbeddingModelRecord, RegistryManager
from omop_emb.utils.cdm import streamed


@dataclass(frozen=True)
class StoredEmbedding:
    """Identity and filter metadata for one stored concept vector."""

    concept_id: int
    domain_id: str
    vocabulary_id: str
    is_standard: bool
    is_valid: bool


class ReadOnlyEmbeddingStore:
    """Inspect an existing embedding store without issuing DDL or DML."""

    def __init__(
        self,
        engine: Engine,
        *,
        backend_type: str | BackendType,
        schema: str | None,
    ) -> None:
        self._engine = engine
        self.backend_type = parse_backend_type(backend_type)
        self.schema = schema
        self._registry = RegistryManager.read_only(engine)

    def __enter__(self) -> ReadOnlyEmbeddingStore:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    @property
    def initialized(self) -> bool:
        """Whether the model registry table already exists."""

        return self._registry.registry_available

    def registered_models(self) -> tuple[EmbeddingModelRecord, ...]:
        """Return registry rows, or an empty tuple for an uninitialized store."""

        return self._registry.get_registered_models()

    def model(self, model_name: str) -> EmbeddingModelRecord | None:
        """Return one registered model without creating a registry."""

        records = self._registry.get_registered_models(model_name=model_name)
        return records[0] if records else None

    def stored_embeddings(self, model_name: str) -> tuple[StoredEmbedding, ...]:
        """Read concept metadata from one existing model table."""

        return tuple(self.iter_stored_embeddings(model_name))

    def iter_stored_embeddings(
        self,
        model_name: str,
        *,
        batch_size: int = 10_000,
    ) -> Iterator[StoredEmbedding]:
        """Stream concept metadata from one existing model table."""

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        record = self.model(model_name)
        if record is None:
            return
        inspector = inspect(self._engine)
        if not inspector.has_table(record.storage_identifier, schema=self.schema):
            return
        schema = (
            None
            if self._engine.dialect.name == Dialect.SQLITE and self.schema == "main"
            else self.schema
        )
        table = concept_metadata_table_descriptor(
            record.storage_identifier,
            schema=schema,
        )
        statement = streamed(
            select(
                table.c.concept_id,
                table.c.domain_id,
                table.c.vocabulary_id,
                table.c.is_standard,
                table.c.is_valid,
            ),
            batch_size,
        )
        with self._engine.connect() as connection:
            rows = connection.execute(statement).mappings()
            for row in rows:
                yield StoredEmbedding(
                    concept_id=int(row["concept_id"]),
                    domain_id=str(row["domain_id"]),
                    vocabulary_id=str(row["vocabulary_id"]),
                    is_standard=bool(row["is_standard"]),
                    is_valid=bool(row["is_valid"]),
                )

    def physical_indexes(self, model_name: str) -> tuple[str, ...]:
        """Return existing PostgreSQL indexes without creating or changing them."""

        if self._engine.dialect.name != Dialect.POSTGRESQL:
            return ()
        record = self.model(model_name)
        if record is None:
            return ()
        expected_prefix = f"idx_{record.storage_identifier}_"
        return tuple(
            str(item["name"])
            for item in inspect(self._engine).get_indexes(
                record.storage_identifier,
                schema=self.schema,
            )
            if str(item["name"]).startswith(expected_prefix)
        )

    def drop_index_sql(self, model_name: str) -> tuple[str, ...]:
        """Return reviewed index-removal statements without executing them."""

        quote = self._engine.dialect.identifier_preparer.quote
        schema_prefix = f"{quote(self.schema)}." if self.schema else ""
        return tuple(
            f"DROP INDEX IF EXISTS {schema_prefix}{quote(name)};"
            for name in self.physical_indexes(model_name)
        )

    def close(self) -> None:
        """Dispose the read-only engine."""

        self._engine.dispose()


def inspect_resolved_vector_store(
    resolved: ResolvedVectorStore,
) -> ReadOnlyEmbeddingStore:
    """Open a resolved vector store for read-only setup inspection.

    SQLite-vec's extension is loaded into the connection so existing virtual
    tables can be queried. Loading the extension is not schema mutation. The
    returned handle must be closed by the caller.
    """

    engine = resolved.database.create_engine()
    if resolved.backend_type == BackendType.SQLITEVEC:
        try:
            import sqlite_vec
        except ImportError as exc:  # pragma: no cover - optional extra
            engine.dispose()
            raise RuntimeError(
                "sqlite-vec inspection requires the omop-emb sqlitevec extra."
            ) from exc

        @event.listens_for(engine, "connect")
        def _load_sqlite_vec(dbapi_connection, _connection_record):
            dbapi_connection.enable_load_extension(True)
            sqlite_vec.load(dbapi_connection)
            dbapi_connection.enable_load_extension(False)

    try:
        return ReadOnlyEmbeddingStore(
            engine,
            backend_type=resolved.backend_type,
            schema=resolved.database.schema_name,
        )
    except Exception:
        engine.dispose()
        raise


def initialize_resolved_vector_store(
    resolved: ResolvedVectorStore,
) -> EmbeddingBackend:
    """Explicitly initialize a resolved store and return its writable backend.

    This is the reviewed maintenance boundary. Diagnostics and population
    planning must call :func:`inspect_resolved_vector_store` instead.
    """

    return resolve_backend_from_resolved_vector_store(resolved)


__all__ = [
    "ReadOnlyEmbeddingStore",
    "StoredEmbedding",
    "initialize_resolved_vector_store",
    "inspect_resolved_vector_store",
]
