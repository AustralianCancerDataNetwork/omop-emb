"""Tests for the generic, FAISS-agnostic embedding export/import bundle.

These exercise round-trip correctness and schema validation directly,
with no FAISS dependency.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pytest

from omop_emb import open_vector_store_writer
from omop_emb.backends.base_backend import ConceptEmbeddingRecord
from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.backends.sqlitevec import SQLiteVecEmbeddingBackend
from omop_emb.config import MetricType, VectorPrecision
from omop_emb.storage import embedding_bundle
from omop_emb.utils.errors import UnsupportedIndexConfigError

from .conftest import duckdbvss_resolved_vector_store, make_svec_backend, populate

pytest.importorskip("h5py", reason="h5py not installed")

_MODEL = "test-bundle-model:v1"
_PROVIDER = "ollama"


def _populate(
    backend: SQLiteVecEmbeddingBackend, *, dim: int, ids: list[int], vecs: np.ndarray
) -> None:
    records = [
        ConceptEmbeddingRecord(
            concept_id=i, domain_id="Drug", vocabulary_id="RxNorm", is_standard=True, is_valid=True
        )
        for i in ids
    ]
    populate(backend, records, vecs, model_name=_MODEL)


@pytest.mark.unit
class TestBundleRoundTrip:
    """Regression coverage for the lossy-COSINE-round-trip bug: a bundle
    export/import must preserve the original raw embedding magnitudes."""

    def test_cosine_round_trip_preserves_magnitude(self, tmp_path):
        dim = 3
        ids = [1, 2, 3]
        # Deliberately non-unit-length vectors: normalizing these would
        # change their values, which is exactly what must NOT happen here.
        vecs = np.array([[3.0, 0.0, 0.0], [0.0, 5.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)

        backend = make_svec_backend()
        _populate(backend, dim=dim, ids=ids, vecs=vecs)

        meta, bundle_path = embedding_bundle.export_bundle(
            backend=backend,
            model_name=_MODEL,
            output_dir=tmp_path,
        )
        assert meta.row_count == len(ids)

        target_backend = make_svec_backend()
        imported = embedding_bundle.import_bundle(backend=target_backend, h5_path=bundle_path)
        assert imported == len(ids)

        roundtripped = target_backend.get_embeddings_by_concept_ids(
            model_name=_MODEL, concept_ids=ids
        )
        for cid, original in zip(ids, vecs):
            got = np.asarray(roundtripped[cid], dtype=np.float32)
            np.testing.assert_allclose(got, original, atol=1e-6)
            # None of the originals are unit-length; confirm normalization
            # never touched the bundle's copy.
            assert np.linalg.norm(got) == pytest.approx(np.linalg.norm(original), abs=1e-5)

    def test_l2_round_trip_preserves_values(self, tmp_path):
        dim = 2
        ids = [10, 20]
        vecs = np.array([[7.0, -2.0], [0.5, 0.5]], dtype=np.float32)

        backend = make_svec_backend()
        _populate(backend, dim=dim, ids=ids, vecs=vecs)

        _, bundle_path = embedding_bundle.export_bundle(
            backend=backend, model_name=_MODEL, output_dir=tmp_path
        )

        target_backend = make_svec_backend()
        embedding_bundle.import_bundle(backend=target_backend, h5_path=bundle_path)

        roundtripped = target_backend.get_embeddings_by_concept_ids(
            model_name=_MODEL, concept_ids=ids
        )
        for cid, original in zip(ids, vecs):
            np.testing.assert_allclose(
                np.asarray(roundtripped[cid], dtype=np.float32), original, atol=1e-6
            )

    def test_import_refuses_overwrite_without_force(self, tmp_path):
        dim = 2
        ids = [1]
        vecs = np.array([[1.0, 2.0]], dtype=np.float32)

        backend = make_svec_backend()
        _populate(backend, dim=dim, ids=ids, vecs=vecs)

        _, bundle_path = embedding_bundle.export_bundle(
            backend=backend, model_name=_MODEL, output_dir=tmp_path
        )

        # Importing into the same backend (already has these embeddings)
        # must refuse without force=True.
        with pytest.raises(RuntimeError):
            embedding_bundle.import_bundle(backend=backend, h5_path=bundle_path)

        # force=True allows it.
        imported = embedding_bundle.import_bundle(backend=backend, h5_path=bundle_path, force=True)
        assert imported == len(ids)

    def test_force_overwrite_bumps_updated_at(self, tmp_path):
        """A force re-import must invalidate any FAISS cache built before it.

        Regression test: import_bundle() used to call bulk_upsert_embeddings()
        directly, bypassing the interface-layer wrapper that bumps
        registry.updated_at: so a FAISS cache built before a force
        re-import kept reporting is_fresh()==True forever, even though the
        re-import had replaced the embedding content underneath it.
        """
        dim = 2
        ids = [1]
        vecs = np.array([[1.0, 2.0]], dtype=np.float32)

        backend = make_svec_backend()
        _populate(backend, dim=dim, ids=ids, vecs=vecs)

        _, bundle_path = embedding_bundle.export_bundle(
            backend=backend, model_name=_MODEL, output_dir=tmp_path
        )

        before = backend.get_registered_model(model_name=_MODEL)
        assert before is not None

        embedding_bundle.import_bundle(backend=backend, h5_path=bundle_path, force=True)

        after = backend.get_registered_model(model_name=_MODEL)
        assert after is not None
        assert after.updated_at is not None
        assert before.updated_at is not None
        assert after.updated_at > before.updated_at

    def test_fresh_registration_keeps_backdated_timestamp(self, tmp_path):
        """A brand-new registration must NOT be bumped to "now": it stays
        backdated to the bundle's own exported_at so a FAISS cache shipped
        alongside the bundle still validates as fresh (see
        TestCrossMachineCacheReuse). Only the *overwrite* path bumps
        updated_at.
        """
        dim = 2
        ids = [1]
        vecs = np.array([[1.0, 2.0]], dtype=np.float32)

        source_backend = make_svec_backend()
        _populate(source_backend, dim=dim, ids=ids, vecs=vecs)

        meta, bundle_path = embedding_bundle.export_bundle(
            backend=source_backend, model_name=_MODEL, output_dir=tmp_path
        )

        target_backend = make_svec_backend()
        embedding_bundle.import_bundle(backend=target_backend, h5_path=bundle_path)

        record = target_backend.get_registered_model(model_name=_MODEL)
        assert record is not None
        assert record.updated_at is not None
        assert record.updated_at == datetime.fromisoformat(meta.exported_at)


@pytest.mark.unit
class TestImportRebuildIndex:
    """import_bundle(rebuild_index=True) applies the bundle's own
    index_config without the caller re-specifying it."""

    def test_rebuild_index_calls_backend_with_bundles_index_config(self, tmp_path, monkeypatch):
        dim = 2
        backend = make_svec_backend()
        _populate(
            backend,
            dim=dim,
            ids=[1],
            vecs=np.array([[1.0, 0.0]], dtype=np.float32),
        )
        _, bundle_path = embedding_bundle.export_bundle(
            backend=backend, model_name=_MODEL, output_dir=tmp_path
        )

        target_backend = make_svec_backend()
        calls = []
        original_rebuild_index = target_backend.rebuild_index

        def _spy_rebuild_index(*, model_name, index_config):
            calls.append((model_name, index_config))
            return original_rebuild_index(model_name=model_name, index_config=index_config)

        monkeypatch.setattr(target_backend, "rebuild_index", _spy_rebuild_index)

        embedding_bundle.import_bundle(
            backend=target_backend, h5_path=bundle_path, rebuild_index=True
        )

        assert calls == [(_MODEL, FlatIndexConfig(metric_type=MetricType.COSINE))]

    def test_rebuild_index_false_by_default_skips_backend_call(self, tmp_path, monkeypatch):
        dim = 2
        backend = make_svec_backend()
        _populate(
            backend,
            dim=dim,
            ids=[1],
            vecs=np.array([[1.0, 0.0]], dtype=np.float32),
        )
        _, bundle_path = embedding_bundle.export_bundle(
            backend=backend, model_name=_MODEL, output_dir=tmp_path
        )

        target_backend = make_svec_backend()
        calls = []
        monkeypatch.setattr(target_backend, "rebuild_index", lambda **kwargs: calls.append(kwargs))

        embedding_bundle.import_bundle(backend=target_backend, h5_path=bundle_path)

        assert calls == []


def _bundle_with_index_config(tmp_path, index_config: dict) -> Path:
    """A bundle whose recorded index config is replaced by ``index_config``."""
    backend = make_svec_backend()
    _populate(backend, dim=2, ids=[1], vecs=np.array([[1.0, 0.0]], dtype=np.float32))
    _, bundle_path = embedding_bundle.export_bundle(
        backend=backend, model_name=_MODEL, output_dir=tmp_path
    )
    with h5py.File(bundle_path, "r+") as f:
        f.attrs[embedding_bundle.ATTR_INDEX_CONFIG] = json.dumps(index_config)
    return bundle_path


_FLOAT16_HNSW = {
    "index_type": "hnsw",
    "metric_type": "cosine",
    "num_neighbors": 8,
    "ef_search": 40,
    "ef_construction": 64,
    "max_scan_tuples": 50_000,
    "index_precision": "float16",
}


@pytest.mark.unit
class TestImportResolvesIndexConfigBeforeLoading:
    """An index config the target cannot honour is handled before any vector loads."""

    def test_unsupported_index_type_fails_before_registering_the_model(self, tmp_path):
        bundle_path = _bundle_with_index_config(tmp_path, _FLOAT16_HNSW)
        target = make_svec_backend()

        with pytest.raises(UnsupportedIndexConfigError):
            embedding_bundle.import_bundle(backend=target, h5_path=bundle_path, rebuild_index=True)

        assert target.get_registered_model(model_name=_MODEL) is None

    def test_without_rebuild_the_index_config_is_not_checked(self, tmp_path):
        bundle_path = _bundle_with_index_config(tmp_path, _FLOAT16_HNSW)
        target = make_svec_backend()

        assert embedding_bundle.import_bundle(backend=target, h5_path=bundle_path) == 1

    @pytest.mark.duckdbvss
    def test_unsupported_options_are_dropped_with_a_warning(self, tmp_path, caplog):
        bundle_path = _bundle_with_index_config(tmp_path, _FLOAT16_HNSW)
        store = duckdbvss_resolved_vector_store(str(tmp_path / "target.duckdb"))

        with open_vector_store_writer(store) as target:
            embedding_bundle.import_bundle(backend=target, h5_path=bundle_path, rebuild_index=True)
            record = target.get_registered_model(model_name=_MODEL)
            assert record is not None
            config = record.index_config

        assert config == HNSWIndexConfig(
            metric_type=MetricType.COSINE, num_neighbors=8, ef_search=40, ef_construction=64
        )
        assert "does not support index_precision or max_scan_tuples" in caplog.text


@pytest.mark.unit
class TestImportBackdatesRegistration:
    """A brand-new registration created by import_bundle() is backdated to
    the bundle's own exported_at, not "now": required so a FAISS cache
    shipped alongside the bundle (built before the bundle's exported_at on
    the source machine) still validates as fresh on the target machine."""

    def test_new_registration_uses_bundles_exported_at(self, tmp_path):
        dim = 2
        backend = make_svec_backend()
        _populate(
            backend,
            dim=dim,
            ids=[1],
            vecs=np.array([[1.0, 0.0]], dtype=np.float32),
        )
        _, bundle_path = embedding_bundle.export_bundle(
            backend=backend, model_name=_MODEL, output_dir=tmp_path
        )

        # Force a clearly-distinguishable timestamp so the test fails if
        # import_bundle() falls back to "now" instead of backdating.
        fake_exported_at = "2020-01-01T00:00:00+00:00"
        with h5py.File(bundle_path, "r+") as f:
            f.attrs[embedding_bundle.ATTR_EXPORTED_AT] = fake_exported_at

        target_backend = make_svec_backend()
        embedding_bundle.import_bundle(backend=target_backend, h5_path=bundle_path)

        record = target_backend.get_registered_model(model_name=_MODEL)
        assert record is not None
        expected = datetime.fromisoformat(fake_exported_at)
        assert record.created_at == expected
        assert record.updated_at == expected


@pytest.mark.unit
class TestExportNaming:
    def test_filename_derived_from_storage_identifier(self, tmp_path):
        backend = make_svec_backend()
        _populate(backend, dim=2, ids=[1], vecs=np.array([[1.0, 0.0]], dtype=np.float32))

        record = backend.get_registered_model(model_name=_MODEL)
        meta, bundle_path = embedding_bundle.export_bundle(
            backend=backend, model_name=_MODEL, output_dir=tmp_path
        )
        assert record is not None
        assert bundle_path == tmp_path / f"{record.storage_identifier}.h5"
        assert meta.index_config == record.index_config


def _bundle(tmp_path) -> Path:
    backend = make_svec_backend()
    _populate(backend, dim=2, ids=[1], vecs=np.array([[1.0, 0.0]], dtype=np.float32))
    _, bundle_path = embedding_bundle.export_bundle(
        backend=backend, model_name=_MODEL, output_dir=tmp_path
    )
    return bundle_path


def _as_v1(bundle_path: Path, *, metric_type: str, index_config: dict | None = None) -> None:
    """Rewrite a bundle's attributes the way schema version 1 wrote them."""
    with h5py.File(bundle_path, "a") as f:
        f.attrs[embedding_bundle.ATTR_SCHEMA_VERSION] = 1
        f.attrs[embedding_bundle.ATTR_V1_METRIC_TYPE] = metric_type
        if index_config is not None:
            f.attrs[embedding_bundle.ATTR_INDEX_CONFIG] = json.dumps(index_config)


_HNSW_L2 = {
    "index_type": "hnsw",
    "metric_type": "l2",
    "num_neighbors": 32,
    "ef_search": 16,
    "ef_construction": 64,
}


@pytest.mark.unit
class TestBundleSchemaVersions:
    def test_export_writes_version_2_without_metric(self, tmp_path):
        with h5py.File(_bundle(tmp_path)) as f:
            assert f.attrs[embedding_bundle.ATTR_SCHEMA_VERSION] == 2
            assert embedding_bundle.ATTR_V1_METRIC_TYPE not in f.attrs

    def test_version_1_flat_bundle_imports(self, tmp_path):
        bundle_path = _bundle(tmp_path)
        _as_v1(bundle_path, metric_type=MetricType.COSINE.value)
        assert embedding_bundle.import_bundle(backend=make_svec_backend(), h5_path=bundle_path) == 1

    def test_version_1_hnsw_bundle_with_matching_metric_reads(self, tmp_path):
        bundle_path = _bundle(tmp_path)
        _as_v1(bundle_path, metric_type=MetricType.L2.value, index_config=_HNSW_L2)
        with h5py.File(bundle_path) as f:
            meta = embedding_bundle.BundleMetadata.from_h5_attrs(f.attrs)
        assert meta.index_config.metric_type is MetricType.L2

    def test_version_1_hnsw_bundle_with_mismatching_metric_raises(self, tmp_path):
        bundle_path = _bundle(tmp_path)
        _as_v1(bundle_path, metric_type=MetricType.COSINE.value, index_config=_HNSW_L2)
        with pytest.raises(
            embedding_bundle.BundleCorruptionError, match=f"built for '{MetricType.L2.value}'"
        ):
            embedding_bundle.import_bundle(backend=make_svec_backend(), h5_path=bundle_path)

    def test_version_1_bundle_without_metric_raises(self, tmp_path):
        bundle_path = _bundle(tmp_path)
        with h5py.File(bundle_path, "a") as f:
            f.attrs[embedding_bundle.ATTR_SCHEMA_VERSION] = 1
        with pytest.raises(embedding_bundle.BundleCorruptionError, match="metric_type"):
            embedding_bundle.import_bundle(backend=make_svec_backend(), h5_path=bundle_path)

    def test_missing_schema_version_raises(self, tmp_path):
        bundle_path = _bundle(tmp_path)
        with h5py.File(bundle_path, "a") as f:
            del f.attrs[embedding_bundle.ATTR_SCHEMA_VERSION]
        with pytest.raises(embedding_bundle.BundleCorruptionError, match="schema_version"):
            embedding_bundle.import_bundle(backend=make_svec_backend(), h5_path=bundle_path)

    def test_unsupported_schema_version_raises(self, tmp_path):
        bundle_path = _bundle(tmp_path)
        with h5py.File(bundle_path, "a") as f:
            f.attrs[embedding_bundle.ATTR_SCHEMA_VERSION] = 3
        with pytest.raises(embedding_bundle.UnsupportedBundleVersionError, match="version 3"):
            embedding_bundle.import_bundle(backend=make_svec_backend(), h5_path=bundle_path)


@pytest.mark.unit
class TestBundleSchemaValidation:
    """Missing required datasets/attributes must be reported clearly, not
    surfaced as a generic KeyError deep inside import logic."""

    def _make_bundle(self, tmp_path):
        dim = 2
        backend = make_svec_backend()
        _populate(
            backend,
            dim=dim,
            ids=[1],
            vecs=np.array([[1.0, 0.0]], dtype=np.float32),
        )
        _, bundle_path = embedding_bundle.export_bundle(
            backend=backend, model_name=_MODEL, output_dir=tmp_path
        )
        return bundle_path

    def test_missing_dataset_raises_corruption_error(self, tmp_path):
        bundle_path = self._make_bundle(tmp_path)
        with h5py.File(bundle_path, "a") as f:
            del f[embedding_bundle.VOCABULARY_IDS]

        with pytest.raises(embedding_bundle.BundleCorruptionError):
            embedding_bundle.import_bundle(backend=make_svec_backend(), h5_path=bundle_path)

    def test_missing_attr_raises_corruption_error(self, tmp_path):
        bundle_path = self._make_bundle(tmp_path)
        with h5py.File(bundle_path, "a") as f:
            del f.attrs[embedding_bundle.ATTR_ROW_COUNT]

        with pytest.raises(embedding_bundle.BundleCorruptionError):
            embedding_bundle.import_bundle(backend=make_svec_backend(), h5_path=bundle_path)

    def test_inconsistent_row_count_raises_corruption_error(self, tmp_path):
        bundle_path = self._make_bundle(tmp_path)
        with h5py.File(bundle_path, "a") as f:
            f.attrs[embedding_bundle.ATTR_ROW_COUNT] = 999

        with pytest.raises(embedding_bundle.BundleCorruptionError):
            embedding_bundle.import_bundle(backend=make_svec_backend(), h5_path=bundle_path)


@pytest.mark.unit
class TestBundleStoragePrecision:
    def test_export_records_the_stored_precision(self, tmp_path):
        with h5py.File(_bundle(tmp_path)) as f:
            meta = embedding_bundle.BundleMetadata.from_h5_attrs(f.attrs)
        assert meta.storage_precision == VectorPrecision.FLOAT32

    def test_bundle_without_the_attribute_reads_as_unrecorded_and_imports(self, tmp_path):
        bundle_path = _bundle(tmp_path)
        with h5py.File(bundle_path, "r+") as f:
            del f.attrs[embedding_bundle.ATTR_STORAGE_PRECISION]
            assert embedding_bundle.BundleMetadata.from_h5_attrs(f.attrs).storage_precision is None
        assert embedding_bundle.import_bundle(backend=make_svec_backend(), h5_path=bundle_path) == 1

    def test_unsupported_precision_falls_back_to_the_backend_default(self, tmp_path, caplog):
        bundle_path = _bundle(tmp_path)
        with h5py.File(bundle_path, "r+") as f:
            f.attrs[embedding_bundle.ATTR_STORAGE_PRECISION] = VectorPrecision.FLOAT16.value

        target = make_svec_backend()
        with caplog.at_level("WARNING"):
            embedding_bundle.import_bundle(backend=target, h5_path=bundle_path)

        record = target.get_registered_model(model_name=_MODEL)
        assert record is not None
        assert record.storage_precision == VectorPrecision.FLOAT32
        assert VectorPrecision.FLOAT16.value in caplog.text
