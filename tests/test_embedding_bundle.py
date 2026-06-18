"""Tests for the generic, FAISS-agnostic embedding export/import bundle.

These exercise round-trip correctness and schema validation directly,
with no FAISS dependency.
"""

from __future__ import annotations

from datetime import datetime

import h5py
import numpy as np
import pytest

from omop_emb.backends.base_backend import ConceptEmbeddingRecord
from omop_emb.backends.index_config import FlatIndexConfig
from omop_emb.backends.sqlitevec import (
    SQLiteVecEmbeddingBackend,
    create_sqlitevec_engine,
)
from omop_emb.config import MetricType, ProviderType
from omop_emb.storage import embedding_bundle

pytest.importorskip("h5py", reason="h5py not installed")

_MODEL = "test-bundle-model:v1"
_PROVIDER = ProviderType.OLLAMA


def _make_backend() -> SQLiteVecEmbeddingBackend:
    engine = create_sqlitevec_engine(":memory:")
    return SQLiteVecEmbeddingBackend(emb_engine=engine)


def _populate(
    backend: SQLiteVecEmbeddingBackend,
    *,
    dim: int,
    ids: list[int],
    vecs: np.ndarray,
    metric_type: MetricType,
) -> None:
    backend.register_model(
        model_name=_MODEL,
        provider_type=_PROVIDER,
        index_config=FlatIndexConfig(),
        dimensions=dim,
    )
    records = [
        ConceptEmbeddingRecord(
            concept_id=i, domain_id="Drug", vocabulary_id="RxNorm", is_standard=True
        )
        for i in ids
    ]
    backend.upsert_embeddings(
        model_name=_MODEL, metric_type=metric_type, records=records, embeddings=vecs
    )


@pytest.mark.unit
class TestBundleRoundTrip:
    """Regression coverage for the lossy-COSINE-round-trip bug: a bundle
    export/import must preserve the original raw embedding magnitudes."""

    def test_cosine_round_trip_preserves_magnitude(self, tmp_path):
        dim = 3
        ids = [1, 2, 3]
        # Deliberately non-unit-length vectors -- normalizing these would
        # change their values, which is exactly what must NOT happen here.
        vecs = np.array(
            [[3.0, 0.0, 0.0], [0.0, 5.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32
        )

        backend = _make_backend()
        _populate(backend, dim=dim, ids=ids, vecs=vecs, metric_type=MetricType.COSINE)

        meta, bundle_path = embedding_bundle.export_bundle(
            backend=backend,
            model_name=_MODEL,
            output_dir=tmp_path,
        )
        assert meta.row_count == len(ids)

        target_backend = _make_backend()
        imported = embedding_bundle.import_bundle(backend=target_backend, h5_path=bundle_path)
        assert imported == len(ids)

        roundtripped = target_backend.get_embeddings_by_concept_ids(
            model_name=_MODEL, metric_type=MetricType.COSINE, concept_ids=ids
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

        backend = _make_backend()
        _populate(backend, dim=dim, ids=ids, vecs=vecs, metric_type=MetricType.L2)

        _, bundle_path = embedding_bundle.export_bundle(
            backend=backend, model_name=_MODEL, output_dir=tmp_path
        )

        target_backend = _make_backend()
        embedding_bundle.import_bundle(backend=target_backend, h5_path=bundle_path)

        roundtripped = target_backend.get_embeddings_by_concept_ids(
            model_name=_MODEL, metric_type=MetricType.L2, concept_ids=ids
        )
        for cid, original in zip(ids, vecs):
            np.testing.assert_allclose(
                np.asarray(roundtripped[cid], dtype=np.float32), original, atol=1e-6
            )

    def test_import_refuses_overwrite_without_force(self, tmp_path):
        dim = 2
        ids = [1]
        vecs = np.array([[1.0, 2.0]], dtype=np.float32)

        backend = _make_backend()
        _populate(backend, dim=dim, ids=ids, vecs=vecs, metric_type=MetricType.COSINE)

        _, bundle_path = embedding_bundle.export_bundle(
            backend=backend, model_name=_MODEL, output_dir=tmp_path
        )

        # Importing into the same backend (already has these embeddings)
        # must refuse without force=True.
        with pytest.raises(RuntimeError):
            embedding_bundle.import_bundle(backend=backend, h5_path=bundle_path)

        # force=True allows it.
        imported = embedding_bundle.import_bundle(
            backend=backend, h5_path=bundle_path, force=True
        )
        assert imported == len(ids)


@pytest.mark.unit
class TestImportRebuildIndex:
    """import_bundle(rebuild_index=True) applies the bundle's own
    index_config without the caller re-specifying it."""

    def test_rebuild_index_calls_backend_with_bundles_index_config(
        self, tmp_path, monkeypatch
    ):
        dim = 2
        backend = _make_backend()
        _populate(
            backend,
            dim=dim,
            ids=[1],
            vecs=np.array([[1.0, 0.0]], dtype=np.float32),
            metric_type=MetricType.COSINE,
        )
        _, bundle_path = embedding_bundle.export_bundle(
            backend=backend, model_name=_MODEL, output_dir=tmp_path
        )

        target_backend = _make_backend()
        calls = []
        original_rebuild_index = target_backend.rebuild_index

        def _spy_rebuild_index(*, model_name, index_config):
            calls.append((model_name, index_config))
            return original_rebuild_index(model_name=model_name, index_config=index_config)

        monkeypatch.setattr(target_backend, "rebuild_index", _spy_rebuild_index)

        embedding_bundle.import_bundle(
            backend=target_backend, h5_path=bundle_path, rebuild_index=True
        )

        assert calls == [(_MODEL, FlatIndexConfig())]

    def test_rebuild_index_false_by_default_skips_backend_call(self, tmp_path, monkeypatch):
        dim = 2
        backend = _make_backend()
        _populate(
            backend,
            dim=dim,
            ids=[1],
            vecs=np.array([[1.0, 0.0]], dtype=np.float32),
            metric_type=MetricType.COSINE,
        )
        _, bundle_path = embedding_bundle.export_bundle(
            backend=backend, model_name=_MODEL, output_dir=tmp_path
        )

        target_backend = _make_backend()
        calls = []
        monkeypatch.setattr(
            target_backend, "rebuild_index", lambda **kwargs: calls.append(kwargs)
        )

        embedding_bundle.import_bundle(backend=target_backend, h5_path=bundle_path)

        assert calls == []


@pytest.mark.unit
class TestImportBackdatesRegistration:
    """A brand-new registration created by import_bundle() is backdated to
    the bundle's own exported_at, not "now" -- required so a FAISS cache
    shipped alongside the bundle (built before the bundle's exported_at on
    the source machine) still validates as fresh on the target machine."""

    def test_new_registration_uses_bundles_exported_at(self, tmp_path):
        dim = 2
        backend = _make_backend()
        _populate(
            backend,
            dim=dim,
            ids=[1],
            vecs=np.array([[1.0, 0.0]], dtype=np.float32),
            metric_type=MetricType.COSINE,
        )
        _, bundle_path = embedding_bundle.export_bundle(
            backend=backend, model_name=_MODEL, output_dir=tmp_path
        )

        # Force a clearly-distinguishable timestamp so the test fails if
        # import_bundle() falls back to "now" instead of backdating.
        fake_exported_at = "2020-01-01T00:00:00+00:00"
        with h5py.File(bundle_path, "r+") as f:
            f.attrs[embedding_bundle.ATTR_EXPORTED_AT] = fake_exported_at

        target_backend = _make_backend()
        embedding_bundle.import_bundle(backend=target_backend, h5_path=bundle_path)

        record = target_backend.get_registered_model(model_name=_MODEL)
        assert record is not None
        expected = datetime.fromisoformat(fake_exported_at)
        assert record.created_at == expected
        assert record.updated_at == expected


@pytest.mark.unit
class TestExportNaming:
    """export_bundle() takes no metric_type: the filename and the recorded
    metric are both derived from the registry, never from a caller-supplied
    value."""

    def test_filename_derived_from_storage_identifier_and_metric_defaults_to_cosine(
        self, tmp_path
    ):
        dim = 2
        backend = _make_backend()
        _populate(
            backend,
            dim=dim,
            ids=[1],
            vecs=np.array([[1.0, 0.0]], dtype=np.float32),
            metric_type=MetricType.COSINE,
        )

        record = backend.get_registered_model(model_name=_MODEL)
        meta, bundle_path = embedding_bundle.export_bundle(
            backend=backend, model_name=_MODEL, output_dir=tmp_path
        )
        assert record is not None
        assert bundle_path == tmp_path / f"{record.storage_identifier}.h5"
        assert bundle_path.exists()
        # Model is FLAT-registered (the only state register_model() allows),
        # so record.metric_type is None and the bundle defaults to COSINE.
        assert record.metric_type is None
        assert meta.metric_type == MetricType.COSINE


@pytest.mark.unit
class TestBundleSchemaValidation:
    """Missing required datasets/attributes must be reported clearly, not
    surfaced as a generic KeyError deep inside import logic."""

    def _make_bundle(self, tmp_path):
        dim = 2
        backend = _make_backend()
        _populate(
            backend,
            dim=dim,
            ids=[1],
            vecs=np.array([[1.0, 0.0]], dtype=np.float32),
            metric_type=MetricType.COSINE,
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
            embedding_bundle.import_bundle(backend=_make_backend(), h5_path=bundle_path)

    def test_missing_attr_raises_corruption_error(self, tmp_path):
        bundle_path = self._make_bundle(tmp_path)
        with h5py.File(bundle_path, "a") as f:
            del f.attrs[embedding_bundle.ATTR_ROW_COUNT]

        with pytest.raises(embedding_bundle.BundleCorruptionError):
            embedding_bundle.import_bundle(backend=_make_backend(), h5_path=bundle_path)

    def test_inconsistent_row_count_raises_corruption_error(self, tmp_path):
        bundle_path = self._make_bundle(tmp_path)
        with h5py.File(bundle_path, "a") as f:
            f.attrs[embedding_bundle.ATTR_ROW_COUNT] = 999

        with pytest.raises(embedding_bundle.BundleCorruptionError):
            embedding_bundle.import_bundle(backend=_make_backend(), h5_path=bundle_path)
