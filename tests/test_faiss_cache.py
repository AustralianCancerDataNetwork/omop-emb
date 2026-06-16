"""Integration tests for FAISSCache — distance conversion and caching correctness.

All tests are self-contained: they build a temporary FAISS index from an
in-memory SQLiteVec backend and assert expected similarity scores.
"""

from __future__ import annotations

import numpy as np
import pytest

import faiss
from omop_emb.backends.base_backend import ConceptEmbeddingRecord
from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.backends.sqlitevec import SQLiteVecEmbeddingBackend, create_sqlitevec_engine
from omop_emb.config import MetricType, ProviderType
from omop_emb.storage.faiss.faiss_cache import FAISSCache

pytest.importorskip("faiss", reason="faiss-cpu not installed")

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_MODEL = "test-faiss-model:v1"
_PROVIDER = ProviderType.OLLAMA

# 2-D unit vectors — exact cosine inner products without floating-point error.
#   ID 1: [1, 0]   aligned with query → IP = 1  → cosine_dist = 0  → sim = 1.0
#   ID 2: [0, 1]   orthogonal         → IP = 0  → cosine_dist = 1  → sim = 0.5
#   ID 3: [-1, 0]  opposite           → IP = -1 → cosine_dist = 2  → sim = 0.0
_COSINE_DIM = 2
_COSINE_IDS = [1, 2, 3]
_COSINE_VECS = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
_COSINE_QUERY = np.array([[1, 0]], dtype=np.float32)
_COSINE_EXPECTED = {1: 1.0, 2: 0.5, 3: 0.0}

_COSINE_RECORDS = [
    ConceptEmbeddingRecord(concept_id=i, domain_id="Test", vocabulary_id="Test", is_standard=True)
    for i in _COSINE_IDS
]

# 2-D L2 vectors — true distances are 0, 1, 2, 4 from the origin query.
#   ID 1: [0, 0]  d=0  → sim = 1/(1+0) = 1.0
#   ID 2: [1, 0]  d=1  → sim = 1/(1+1) = 0.5
#   ID 3: [2, 0]  d=2  → sim = 1/(1+2) ≈ 0.333
#   ID 4: [4, 0]  d=4  → sim = 1/(1+4) = 0.2
_L2_DIM = 2
_L2_IDS = [1, 2, 3, 4]
_L2_VECS = np.array([[0, 0], [1, 0], [2, 0], [4, 0]], dtype=np.float32)
_L2_QUERY = np.array([[0, 0]], dtype=np.float32)
_L2_EXPECTED = {1: 1.0, 2: 0.5, 3: 1 / 3, 4: 0.2}

_L2_RECORDS = [
    ConceptEmbeddingRecord(concept_id=i, domain_id="Test", vocabulary_id="Test", is_standard=True)
    for i in _L2_IDS
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_svec_backend(dim: int) -> SQLiteVecEmbeddingBackend:
    engine = create_sqlitevec_engine(":memory:")
    return SQLiteVecEmbeddingBackend(emb_engine=engine)


def _populate_backend(
    backend: SQLiteVecEmbeddingBackend,
    *,
    dim: int,
    records: list[ConceptEmbeddingRecord],
    vecs: np.ndarray,
    metric_type: MetricType,
) -> None:
    backend.register_model(
        model_name=_MODEL,
        provider_type=_PROVIDER,
        index_config=FlatIndexConfig(),
        dimensions=dim,
    )
    backend.upsert_embeddings(
        model_name=_MODEL,
        metric_type=metric_type,
        records=records,
        embeddings=vecs,
    )


def _build_faiss(
    tmp_path,
    backend: SQLiteVecEmbeddingBackend,
    metric_type: MetricType,
    index_config=None,
) -> FAISSCache:
    if index_config is None:
        index_config = FlatIndexConfig()
    cache = FAISSCache(model_name=_MODEL, cache_dir=tmp_path)
    cache.export(backend=backend, metric_type=metric_type, index_config=index_config)
    return cache


# ---------------------------------------------------------------------------
# COSINE metric correctness
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCosineCorrectness:
    """FAISS COSINE path: inner product → cosine distance → similarity."""

    @pytest.fixture
    def faiss_cosine(self, tmp_path) -> FAISSCache:
        backend = _make_svec_backend(_COSINE_DIM)
        _populate_backend(backend, dim=_COSINE_DIM, records=_COSINE_RECORDS,
                          vecs=_COSINE_VECS, metric_type=MetricType.COSINE)
        return _build_faiss(tmp_path, backend, MetricType.COSINE)

    def test_identical_vectors_score_one(self, faiss_cosine: FAISSCache):
        results = faiss_cosine.search(_COSINE_QUERY, k=3, metric_type=MetricType.COSINE,
                                      index_config=FlatIndexConfig())
        by_id = {r.concept_id: r.similarity for r in results[0]}
        assert by_id[1] == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors_score_half(self, faiss_cosine: FAISSCache):
        results = faiss_cosine.search(_COSINE_QUERY, k=3, metric_type=MetricType.COSINE,
                                      index_config=FlatIndexConfig())
        by_id = {r.concept_id: r.similarity for r in results[0]}
        assert by_id[2] == pytest.approx(0.5, abs=1e-5)

    def test_opposite_vectors_score_zero(self, faiss_cosine: FAISSCache):
        results = faiss_cosine.search(_COSINE_QUERY, k=3, metric_type=MetricType.COSINE,
                                      index_config=FlatIndexConfig())
        by_id = {r.concept_id: r.similarity for r in results[0]}
        assert by_id[3] == pytest.approx(0.0, abs=1e-5)

    def test_ranking_order(self, faiss_cosine: FAISSCache):
        """Most-similar concept returned first."""
        results = faiss_cosine.search(_COSINE_QUERY, k=3, metric_type=MetricType.COSINE,
                                      index_config=FlatIndexConfig())
        ids_in_order = [r.concept_id for r in results[0]]
        assert ids_in_order == [1, 2, 3]


# ---------------------------------------------------------------------------
# L2 metric correctness
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestL2Correctness:
    """FAISS L2 path: squared distance → sqrt → true L2 → similarity."""

    @pytest.fixture
    def faiss_l2(self, tmp_path) -> FAISSCache:
        backend = _make_svec_backend(_L2_DIM)
        _populate_backend(backend, dim=_L2_DIM, records=_L2_RECORDS,
                          vecs=_L2_VECS, metric_type=MetricType.L2)
        return _build_faiss(tmp_path, backend, MetricType.L2)

    @pytest.mark.parametrize("concept_id,expected_sim", list(_L2_EXPECTED.items()))
    def test_l2_similarity_values(self, faiss_l2: FAISSCache, concept_id: int, expected_sim: float):
        results = faiss_l2.search(_L2_QUERY, k=4, metric_type=MetricType.L2,
                                  index_config=FlatIndexConfig())
        by_id = {r.concept_id: r.similarity for r in results[0]}
        assert by_id[concept_id] == pytest.approx(expected_sim, abs=1e-5)


# ---------------------------------------------------------------------------
# Cross-backend parity: FAISS vs SQLiteVec
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCrossBackendParity:
    """FAISS and SQLiteVec must return the same similarity scores."""

    def _svec_search(
        self,
        backend: SQLiteVecEmbeddingBackend,
        query: np.ndarray,
        k: int,
        metric_type: MetricType,
    ) -> dict[int, float]:
        results = backend.get_nearest_concepts(
            model_name=_MODEL,
            query_embeddings=query,
            k=k,
            metric_type=metric_type,
        )
        return {r.concept_id: r.similarity for r in results[0]}

    def test_cosine_parity(self, tmp_path):
        backend = _make_svec_backend(_COSINE_DIM)
        _populate_backend(backend, dim=_COSINE_DIM, records=_COSINE_RECORDS,
                          vecs=_COSINE_VECS, metric_type=MetricType.COSINE)
        cache = _build_faiss(tmp_path, backend, MetricType.COSINE)

        svec = self._svec_search(backend, _COSINE_QUERY, k=3, metric_type=MetricType.COSINE)
        faiss_results = cache.search(_COSINE_QUERY, k=3, metric_type=MetricType.COSINE,
                                     index_config=FlatIndexConfig())
        faiss_sims = {r.concept_id: r.similarity for r in faiss_results[0]}

        for cid in _COSINE_IDS:
            assert faiss_sims[cid] == pytest.approx(svec[cid], abs=1e-4), (
                f"Concept {cid}: FAISS={faiss_sims[cid]:.6f} SQLiteVec={svec[cid]:.6f}"
            )

    def test_l2_parity(self, tmp_path):
        backend = _make_svec_backend(_L2_DIM)
        _populate_backend(backend, dim=_L2_DIM, records=_L2_RECORDS,
                          vecs=_L2_VECS, metric_type=MetricType.L2)
        cache = _build_faiss(tmp_path, backend, MetricType.L2)

        svec = self._svec_search(backend, _L2_QUERY, k=4, metric_type=MetricType.L2)
        faiss_results = cache.search(_L2_QUERY, k=4, metric_type=MetricType.L2,
                                     index_config=FlatIndexConfig())
        faiss_sims = {r.concept_id: r.similarity for r in faiss_results[0]}

        for cid in _L2_IDS:
            assert faiss_sims[cid] == pytest.approx(svec[cid], abs=1e-4), (
                f"Concept {cid}: FAISS={faiss_sims[cid]:.6f} SQLiteVec={svec[cid]:.6f}"
            )


# ---------------------------------------------------------------------------
# Cache identity
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIndexCache:
    """Single-entry cache returns the same object on repeated loads."""

    def test_cache_hit_returns_same_object(self, tmp_path):
        backend = _make_svec_backend(_L2_DIM)
        _populate_backend(backend, dim=_L2_DIM, records=_L2_RECORDS,
                          vecs=_L2_VECS, metric_type=MetricType.L2)
        cache = _build_faiss(tmp_path, backend, MetricType.L2)

        idx_a = cache._load_index(MetricType.L2, FlatIndexConfig())
        idx_b = cache._load_index(MetricType.L2, FlatIndexConfig())
        assert idx_a is idx_b

    def test_cache_miss_on_different_metric(self, tmp_path):
        """Switching metric evicts the old entry; a new load returns a fresh object."""
        backend = _make_svec_backend(_COSINE_DIM)
        _populate_backend(backend, dim=_COSINE_DIM, records=_COSINE_RECORDS,
                          vecs=_COSINE_VECS, metric_type=MetricType.L2)
        _populate_backend_metric(backend, dim=_COSINE_DIM, records=_COSINE_RECORDS,
                                  vecs=_COSINE_VECS, metric_type=MetricType.COSINE)

        cache = FAISSCache(model_name=_MODEL, cache_dir=tmp_path)
        cache.export(backend=backend, metric_type=MetricType.L2, index_config=FlatIndexConfig())
        cache.export(backend=backend, metric_type=MetricType.COSINE, index_config=FlatIndexConfig())

        idx_l2 = cache._load_index(MetricType.L2, FlatIndexConfig())
        idx_cosine = cache._load_index(MetricType.COSINE, FlatIndexConfig())
        # After loading COSINE the cache slot holds the COSINE index
        idx_l2_reload = cache._load_index(MetricType.L2, FlatIndexConfig())
        # They are different objects (cache was evicted between loads)
        assert idx_l2 is not idx_cosine
        assert idx_l2_reload is not idx_cosine


# ---------------------------------------------------------------------------
# HNSW efSearch applied at load time
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHNSWEfSearch:
    """efSearch from HNSWIndexConfig is applied to the loaded index."""

    def test_ef_search_applied(self, tmp_path):
        backend = _make_svec_backend(_COSINE_DIM)
        _populate_backend(backend, dim=_COSINE_DIM, records=_COSINE_RECORDS,
                          vecs=_COSINE_VECS, metric_type=MetricType.COSINE)

        index_config = HNSWIndexConfig(metric_type=MetricType.COSINE, ef_search=32)
        cache = _build_faiss(tmp_path, backend, MetricType.COSINE, index_config=index_config)

        index = cache._load_index(MetricType.COSINE, index_config)
        # IndexIDMap wraps the HNSW index; downcast to expose .hnsw
        inner = faiss.downcast_index(index.index)
        assert hasattr(inner, "hnsw"), "Expected an HNSW sub-index"
        assert inner.hnsw.efSearch == 32

    def test_ef_search_default_overridden(self, tmp_path):
        """Index was built (and serialised) with FAISS default efSearch=16;
        loading with ef_search=64 applies the new value."""
        backend = _make_svec_backend(_COSINE_DIM)
        _populate_backend(backend, dim=_COSINE_DIM, records=_COSINE_RECORDS,
                          vecs=_COSINE_VECS, metric_type=MetricType.COSINE)

        build_config = HNSWIndexConfig(metric_type=MetricType.COSINE, ef_search=16)
        cache = _build_faiss(tmp_path, backend, MetricType.COSINE, index_config=build_config)

        load_config = HNSWIndexConfig(metric_type=MetricType.COSINE, ef_search=64)
        index = cache._load_index(MetricType.COSINE, load_config)
        inner = faiss.downcast_index(index.index)
        assert inner.hnsw.efSearch == 64


# ---------------------------------------------------------------------------
# Internal helper (not a test class)
# ---------------------------------------------------------------------------

def _populate_backend_metric(
    backend: SQLiteVecEmbeddingBackend,
    *,
    dim: int,
    records: list[ConceptEmbeddingRecord],
    vecs: np.ndarray,
    metric_type: MetricType,
) -> None:
    """Upsert embeddings for a model already registered in *backend*."""
    backend.upsert_embeddings(
        model_name=_MODEL,
        metric_type=metric_type,
        records=records,
        embeddings=vecs,
    )
