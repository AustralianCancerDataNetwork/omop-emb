"""Read/write split of EmbeddingBackend: the EmbeddingStoreReader view, the
@writes guard, and the reader path running no DDL."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import InternalError, OperationalError

from omop_emb.backends import (
    EmbeddingStoreReader,
    open_vector_store_reader,
    open_vector_store_writer,
)
from omop_emb.backends.base_backend import EmbeddingBackend
from omop_emb.backends.embedding_table import ConceptEmbeddingRecord
from omop_emb.backends.index_config import FlatIndexConfig
from omop_emb.config import MetricType
from omop_emb.storage.embedding_bundle import export_bundle
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter
from omop_emb.utils.errors import ReadOnlyStoreError

from .conftest import CONCEPT_RECORDS, sqlite_resolved_vector_store

# Public EmbeddingBackend members that are neither reads nor writes: identity,
# engine access, constants and static validators.
_NEITHER = frozenset({
    "DEFAULT_K_NEAREST",
    "backend_name",
    "dialect",
    "emb_engine",
    "emb_session_factory",
    "validate_embeddings",
    "validate_embeddings_and_records",
})

_MODEL = "test-model"
_RECORDS = (
    ConceptEmbeddingRecord(1, "Condition", "SNOMED", True, True),
    ConceptEmbeddingRecord(2, "Drug", "RxNorm", False, True),
)
_VECTORS = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def _populated_store(tmp_path):
    return _populate(sqlite_resolved_vector_store(str(tmp_path / "test.db")))


def _populate(store):
    with open_vector_store_writer(store) as writer:
        for record in writer.get_registered_models():
            writer.delete_model(model_name=record.model_name)
        writer.register_model(model_name=_MODEL, provider_type="ollama", index_config=FlatIndexConfig(), dimensions=3)
        writer.upsert_embeddings(model_name=_MODEL, records=list(_RECORDS), embeddings=_VECTORS)
    return store


def test_every_public_backend_member_is_classified_exactly_once():
    public = {name for name in dir(EmbeddingBackend) if not name.startswith("_")}
    reader = {name for name in vars(EmbeddingStoreReader) if not name.startswith("_")}
    writes = {
        name for name in public
        if getattr(getattr(EmbeddingBackend, name), "__omop_emb_writes__", False)
    }
    assert reader <= public
    assert not reader & writes
    assert not (reader | writes) & _NEITHER
    assert public == reader | writes | _NEITHER


def _all_subclasses(cls: type) -> set[type]:
    direct = set(cls.__subclasses__())
    return direct | {grandchild for child in direct for grandchild in _all_subclasses(child)}


def test_every_concrete_backend_override_of_a_write_method_stays_classified_as_a_write():
    """A subclass overriding an @writes-decorated base method must re-apply
    @writes itself: the classification test above only inspects
    EmbeddingBackend directly, so an override that drops the decorator
    (while still working correctly through super()) would otherwise be
    invisible to it."""
    from omop_emb.backends.pgvector import PGVectorEmbeddingBackend  # noqa: F401
    from omop_emb.backends.sqlitevec import SQLiteVecEmbeddingBackend  # noqa: F401

    write_names = {
        name for name in dir(EmbeddingBackend)
        if not name.startswith("_")
        and getattr(getattr(EmbeddingBackend, name), "__omop_emb_writes__", False)
    }
    subclasses = _all_subclasses(EmbeddingBackend)
    assert subclasses, "expected at least one concrete EmbeddingBackend subclass to be importable"
    for subclass in subclasses:
        for name in write_names:
            if name not in vars(subclass):
                continue  # inherited as-is, already classified via the base class
            overridden = vars(subclass)[name]
            assert getattr(overridden, "__omop_emb_writes__", False), (
                f"{subclass.__name__}.{name} overrides a @writes method without "
                "re-applying @writes"
            )


def test_reader_rejects_every_write(tmp_path):
    store = sqlite_resolved_vector_store(str(tmp_path / "test.db"))
    with open_vector_store_writer(store) as writer:
        writer.register_model(
            model_name=_MODEL, provider_type="ollama", index_config=FlatIndexConfig(), dimensions=1,
        )

    with open_vector_store_reader(store) as reader:
        backend = reader
        assert isinstance(backend, EmbeddingBackend)
        with pytest.raises(ReadOnlyStoreError):
            backend.register_model(model_name="other-model", provider_type="ollama", dimensions=1)
        with pytest.raises(ReadOnlyStoreError):
            backend.upsert_embeddings(
                model_name=_MODEL,
                records=CONCEPT_RECORDS[:1], embeddings=np.zeros((1, 1), dtype=np.float32),
            )
        with pytest.raises(ReadOnlyStoreError):
            backend.delete_model(model_name=_MODEL)
        assert [record.model_name for record in reader.get_registered_models()] == [_MODEL]


def _assert_every_read_works_without_mutating_sql(store) -> None:
    statements: list[str] = []

    def capture(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement.strip().lower())

    with open_vector_store_reader(store) as reader:
        engine = reader.emb_engine  # ty: ignore[unresolved-attribute]
        event.listen(engine, "before_cursor_execute", capture)
        try:
            assert reader.initialized is True
            assert reader.get_registered_model(model_name=_MODEL) is not None
            assert len(reader.get_registered_models()) == 1
            assert tuple(reader.iter_stored_embeddings(_MODEL)) == _RECORDS
            assert reader.has_any_embeddings(model_name=_MODEL) is True
            assert reader.get_stored_concept_ids(model_name=_MODEL) == {1, 2}
            assert reader.get_stored_concept_ids(
                model_name=_MODEL,
                concept_filter=EmbeddingConceptFilter(concept_ids=(1, 2), domains=("Drug",), vocabularies=("RxNorm",)),
            ) == {2}
            assert set(reader.get_embeddings_by_concept_ids(model_name=_MODEL, concept_ids=[1, 2])) == {1, 2}
            assert reader.get_concept_filter_metadata(model_name=_MODEL, concept_ids=[1, 2]) == {
                r.concept_id: r for r in _RECORDS
            }
            assert reader.get_embedding_count(model_name=_MODEL) == 2
            assert reader.get_embedding_count_by_vocabulary(model_name=_MODEL) == {"SNOMED": 1, "RxNorm": 1}
            matches = reader.get_nearest_concepts(
                model_name=_MODEL,
                metric_type=MetricType.COSINE,
                query_embeddings=_VECTORS[:1],
                concept_filter=EmbeddingConceptFilter(domains=("Condition",)),
                k=1,
            )
            assert matches[0][0].concept_id == 1
            assert reader.physical_indexes(_MODEL) == ()
        finally:
            event.remove(engine, "before_cursor_execute", capture)

    assert statements
    assert not any(
        statement.startswith(("create ", "alter ", "drop ", "insert ", "update ", "delete "))
        for statement in statements
    )


def test_reader_runs_every_read_without_mutating_sql(tmp_path):
    _assert_every_read_works_without_mutating_sql(_populated_store(tmp_path))


def test_reader_connection_refuses_writes_at_database_level(tmp_path):
    """Writes that bypass @writes still fail: the connection itself is query-only."""
    store = _populated_store(tmp_path)
    with open_vector_store_reader(store) as reader:
        engine = reader.emb_engine  # ty: ignore[unresolved-attribute]
        with pytest.raises(OperationalError, match="readonly"):
            with engine.begin() as connection:
                connection.execute(text("CREATE TEMPORARY TABLE _probe (id INTEGER)"))


def test_export_and_faiss_build_run_on_a_reader(tmp_path):
    pytest.importorskip("faiss")
    from omop_emb.storage.faiss import FAISSCache

    store = _populated_store(tmp_path)
    with open_vector_store_reader(store) as reader:
        meta, h5_path = export_bundle(reader, _MODEL, tmp_path / "export")
        cache = FAISSCache(model_name=_MODEL, cache_dir=tmp_path / "faiss")
        cache.build_from_backend(reader, MetricType.COSINE, FlatIndexConfig())

    assert meta.row_count == 2
    assert h5_path.exists()
    assert cache.faiss_path(MetricType.COSINE, FlatIndexConfig()).exists()


@pytest.mark.postgresql
@pytest.mark.db_dialect
def test_only_the_writer_attaches_the_vector_extension_hook(pg_db, monkeypatch):
    from oa_configurator import ResolvedVectorStore

    from omop_emb.backends.pgvector import pg_backend

    calls: list[object] = []

    def _record(dbapi_connection, _connection_record):
        calls.append(dbapi_connection)

    monkeypatch.setattr(pg_backend, "_create_vector_extension", _record)
    store = ResolvedVectorStore(
        name="default", backend_type="pgvector", database=pg_db.resolved, faiss_cache_dir=None, configuration={},
    )

    with open_vector_store_reader(store) as reader:
        reader.get_registered_models()
    assert calls == []

    with open_vector_store_writer(store) as writer:
        writer.get_registered_models()
    assert calls


@pytest.mark.postgresql
@pytest.mark.db_dialect
class TestPostgresReader:
    @pytest.fixture
    def store(self, pg_db):
        from oa_configurator import ResolvedVectorStore

        from omop_emb.backends.embedding_table import EmbeddingTableBase

        store = _populate(ResolvedVectorStore(
            name="default", backend_type="pgvector", database=pg_db.resolved, faiss_cache_dir=None, configuration={},
        ))
        yield store
        with open_vector_store_writer(store) as writer:
            writer.delete_model(model_name=_MODEL)
        EmbeddingTableBase.metadata.clear()
        EmbeddingTableBase.registry._class_registry.clear()

    def test_every_read_works_on_a_read_only_connection(self, store):
        _assert_every_read_works_without_mutating_sql(store)

    def test_connection_refuses_writes_at_database_level(self, store):
        with open_vector_store_reader(store) as reader:
            engine = reader.emb_engine  # ty: ignore[unresolved-attribute]
            with pytest.raises(InternalError, match="read-only transaction"):
                with engine.begin() as connection:
                    connection.execute(text("CREATE TEMPORARY TABLE _probe (id INTEGER)"))
