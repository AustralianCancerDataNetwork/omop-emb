"""Shared backend tests — run against any EmbeddingBackend implementation.

Subclass SharedBackendTests in backend-specific test modules and provide a
``backend`` fixture that returns a registered-free instance of the backend
under test.  The ``metric_type`` fixture defaults to L2; override it in the
subclass for metric-specific variants.
"""

from __future__ import annotations

import numpy as np
import pytest

from omop_emb.backends.base_backend import EmbeddingBackend
from omop_emb.backends.index_config import FlatIndexConfig
from omop_emb.config import MetricType
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter

from .conftest import (
    ASPIRIN_ID,
    CONCEPT_EMBEDDINGS,
    CONCEPT_RECORDS,
    DIABETES_ID,
    EMBEDDING_DIM,
    HYPERTENSION_ID,
    MODEL_NAME,
    NON_STANDARD_ID,
    PROVIDER_TYPE,
    QUERY_EMBEDDING,
)


class SharedBackendTests:
    """Mixin class — drop into any backend test class."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _register(self, backend: EmbeddingBackend):
        return backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(),
            dimensions=EMBEDDING_DIM,
        )

    def _upsert_all(self, backend: EmbeddingBackend, metric_type: MetricType = MetricType.L2):
        self._register(backend)
        backend.upsert_embeddings(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            metric_type=metric_type,
            records=list(CONCEPT_RECORDS),
            embeddings=CONCEPT_EMBEDDINGS,
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def test_register_model(self, backend: EmbeddingBackend):
        record = self._register(backend)
        assert record.model_name == MODEL_NAME
        assert record.dimensions == EMBEDDING_DIM

    def test_register_model_is_idempotent(self, backend: EmbeddingBackend):
        r1 = self._register(backend)
        r2 = self._register(backend)
        assert r1.storage_identifier == r2.storage_identifier

    def test_register_model_dimension_conflict_raises(self, backend: EmbeddingBackend):
        self._register(backend)
        with pytest.raises(Exception):
            backend.register_model(
                model_name=MODEL_NAME,
                provider_type=PROVIDER_TYPE,
                index_config=FlatIndexConfig(),
                dimensions=EMBEDDING_DIM + 1,
            )

    def test_is_model_registered(self, backend: EmbeddingBackend):
        assert not backend.is_model_registered(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
        )
        self._register(backend)
        assert backend.is_model_registered(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
        )

    def test_get_registered_model(self, backend: EmbeddingBackend):
        self._register(backend)
        record = backend.get_registered_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
        )
        assert record is not None
        assert record.model_name == MODEL_NAME
        assert record.dimensions == EMBEDDING_DIM

    def test_get_registered_models_returns_all(self, backend: EmbeddingBackend):
        backend.register_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            index_config=FlatIndexConfig(),
            dimensions=EMBEDDING_DIM,
        )
        records = backend.get_registered_models(model_name=MODEL_NAME, provider_type=PROVIDER_TYPE)
        assert len(records) >= 1

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def test_upsert_and_has_embeddings(self, backend: EmbeddingBackend):
        self._upsert_all(backend)
        assert backend.has_any_embeddings(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            metric_type=MetricType.L2,
        )

    def test_upsert_count_matches(self, backend: EmbeddingBackend):
        self._upsert_all(backend)
        count = backend.get_embedding_count(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            metric_type=MetricType.L2,
        )
        assert count == len(CONCEPT_RECORDS)

    def test_upsert_is_idempotent(self, backend: EmbeddingBackend):
        self._upsert_all(backend)
        backend.upsert_embeddings(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            metric_type=MetricType.L2,
            records=list(CONCEPT_RECORDS),
            embeddings=CONCEPT_EMBEDDINGS,
        )
        count = backend.get_embedding_count(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            metric_type=MetricType.L2,
        )
        assert count == len(CONCEPT_RECORDS)

    # ------------------------------------------------------------------
    # Read — concept IDs
    # ------------------------------------------------------------------

    def test_get_all_stored_concept_ids(self, backend: EmbeddingBackend):
        self._upsert_all(backend)
        stored = backend.get_all_stored_concept_ids(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            metric_type=MetricType.L2,
        )
        expected = {r.concept_id for r in CONCEPT_RECORDS}
        assert stored == expected

    def test_get_embeddings_by_concept_ids(self, backend: EmbeddingBackend):
        self._upsert_all(backend)
        result = backend.get_embeddings_by_concept_ids(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            metric_type=MetricType.L2,
            concept_ids=[HYPERTENSION_ID, DIABETES_ID],
        )
        assert set(result.keys()) == {HYPERTENSION_ID, DIABETES_ID}
        assert np.isclose(result[HYPERTENSION_ID][0], -10.0)
        assert np.isclose(result[DIABETES_ID][0], 0.0)

    # ------------------------------------------------------------------
    # KNN — basic
    # ------------------------------------------------------------------

    def test_knn_returns_results(self, backend: EmbeddingBackend):
        self._upsert_all(backend)
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            metric_type=MetricType.L2,
            query_embeddings=QUERY_EMBEDDING,
            k=3,
        )
        assert len(results) == 1
        assert len(results[0]) == 3

    def test_knn_top1_is_closest(self, backend: EmbeddingBackend):
        # Query [-1] is closest to Diabetes [0] under L2
        self._upsert_all(backend)
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            metric_type=MetricType.L2,
            query_embeddings=QUERY_EMBEDDING,
            k=1,
        )
        assert results[0][0].concept_id == DIABETES_ID

    def test_knn_exact_hypertension_query(self, backend: EmbeddingBackend):
        # Querying exactly the stored vector should return itself first
        self._upsert_all(backend)
        query = np.array([[-10.0]], dtype=np.float32)
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            metric_type=MetricType.L2,
            query_embeddings=query,
            k=1,
        )
        assert results[0][0].concept_id == HYPERTENSION_ID

    def test_knn_batch_queries(self, backend: EmbeddingBackend):
        self._upsert_all(backend)
        queries = np.array([[-10.0], [10.0]], dtype=np.float32)
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            metric_type=MetricType.L2,
            query_embeddings=queries,
            k=1,
        )
        assert len(results) == 2
        assert results[0][0].concept_id == HYPERTENSION_ID
        assert results[1][0].concept_id == ASPIRIN_ID

    # ------------------------------------------------------------------
    # KNN — filters
    # ------------------------------------------------------------------

    def test_knn_domain_filter(self, backend: EmbeddingBackend):
        self._upsert_all(backend)
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            metric_type=MetricType.L2,
            query_embeddings=QUERY_EMBEDDING,
            concept_filter=EmbeddingConceptFilter(domains=("Drug",), limit=10),
        )
        returned_ids = {r.concept_id for r in results[0]}
        assert ASPIRIN_ID in returned_ids
        assert HYPERTENSION_ID not in returned_ids
        assert DIABETES_ID not in returned_ids

    def test_knn_vocabulary_filter(self, backend: EmbeddingBackend):
        self._upsert_all(backend)
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            metric_type=MetricType.L2,
            query_embeddings=QUERY_EMBEDDING,
            concept_filter=EmbeddingConceptFilter(vocabularies=("SNOMED",), limit=10),
        )
        returned_ids = {r.concept_id for r in results[0]}
        assert HYPERTENSION_ID in returned_ids
        assert DIABETES_ID in returned_ids
        assert ASPIRIN_ID not in returned_ids

    def test_knn_concept_id_filter(self, backend: EmbeddingBackend):
        self._upsert_all(backend)
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            metric_type=MetricType.L2,
            query_embeddings=QUERY_EMBEDDING,
            concept_filter=EmbeddingConceptFilter(
                concept_ids=(HYPERTENSION_ID, ASPIRIN_ID), limit=10
            ),
        )
        returned_ids = {r.concept_id for r in results[0]}
        assert returned_ids == {HYPERTENSION_ID, ASPIRIN_ID}

    def test_knn_require_standard_filter(self, backend: EmbeddingBackend):
        self._upsert_all(backend)
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            metric_type=MetricType.L2,
            query_embeddings=QUERY_EMBEDDING,
            concept_filter=EmbeddingConceptFilter(require_standard=True, limit=10),
        )
        returned_ids = {r.concept_id for r in results[0]}
        assert NON_STANDARD_ID not in returned_ids
        assert HYPERTENSION_ID in returned_ids

    # ------------------------------------------------------------------
    # KNN — similarity math
    # ------------------------------------------------------------------

    def test_l2_similarity_values(self, backend: EmbeddingBackend):
        """L2 sim = 1/(1+dist).  Query=[-1], stored=[-10,0,10,20]."""
        self._upsert_all(backend)
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            metric_type=MetricType.L2,
            query_embeddings=QUERY_EMBEDDING,
            k=len(CONCEPT_RECORDS),
        )
        expected = {
            HYPERTENSION_ID: 1.0 / (1.0 + 9.0),   # dist=9
            DIABETES_ID: 1.0 / (1.0 + 1.0),        # dist=1
            ASPIRIN_ID: 1.0 / (1.0 + 11.0),        # dist=11
            NON_STANDARD_ID: 1.0 / (1.0 + 21.0),   # dist=21
        }
        for match in results[0]:
            assert np.isclose(match.similarity, expected[match.concept_id], rtol=1e-4), (
                f"concept_id={match.concept_id}: got {match.similarity}, "
                f"expected {expected[match.concept_id]}"
            )

    # ------------------------------------------------------------------
    # Delete model
    # ------------------------------------------------------------------

    def test_delete_model_removes_registry_entry(self, backend: EmbeddingBackend):
        self._register(backend)
        assert backend.is_model_registered(
            model_name=MODEL_NAME, provider_type=PROVIDER_TYPE,
        )
        backend.delete_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
        )
        assert not backend.is_model_registered(
            model_name=MODEL_NAME, provider_type=PROVIDER_TYPE,
        )

    def test_patch_model_metadata(self, backend: EmbeddingBackend):
        self._register(backend)
        backend.patch_model_metadata(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
            key="custom_key",
            value={"info": "test"},
        )
        record = backend.get_registered_model(
            model_name=MODEL_NAME,
            provider_type=PROVIDER_TYPE,
        )
        assert record.metadata.get("custom_key") == {"info": "test"}
