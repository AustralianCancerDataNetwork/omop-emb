"""Shared backend tests: run against any EmbeddingBackend implementation.

Subclass SharedBackendTests in backend-specific test modules and provide a
``backend`` fixture that returns a registered-free instance of the backend
under test. Queries use the cosine metric, the one metric every backend supports.
"""

from __future__ import annotations

import numpy as np
import pytest

from omop_emb.backends.base_backend import EmbeddingBackend
from omop_emb.backends.index_config import FlatIndexConfig, HNSWIndexConfig
from omop_emb.config import MetricType
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter

from .conftest import (
    ASPIRIN_ID,
    CONCEPT_EMBEDDINGS,
    CONCEPT_RECORDS,
    COSINE_SIMILARITY,
    DIABETES_ID,
    EMBEDDING_DIM,
    HYPERTENSION_ID,
    MODEL_NAME,
    NON_STANDARD_ID,
    PROVIDER_TYPE,
    QUERY_EMBEDDING,
    register_flat,
    upsert_all,
)


class SharedBackendTests:
    """Mixin class: drop into any backend test class."""

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def test_register_model(self, backend: EmbeddingBackend):
        record = register_flat(backend)
        assert record.model_name == MODEL_NAME
        assert record.dimensions == EMBEDDING_DIM

    def test_register_model_is_idempotent(self, backend: EmbeddingBackend):
        """Re-registering returns the same record: one row and one physical table per model."""
        r1 = register_flat(backend)
        r2 = register_flat(backend)
        assert r1.storage_identifier == r2.storage_identifier
        assert len(backend.get_registered_models(model_name=MODEL_NAME)) == 1

    def test_flat_registration_takes_the_backends_default_metric(self, backend: EmbeddingBackend):
        """None where queries choose the metric (pgvector), the table's metric where it is locked (sqlite-vec)."""
        assert register_flat(backend).metric_type == backend.default_flat_metric

    def test_hnsw_registration_raises(self, backend: EmbeddingBackend):
        with pytest.raises(ValueError, match="Only FLAT index is allowed at registration"):
            backend.register_model(
                model_name=MODEL_NAME,
                provider_type=PROVIDER_TYPE,
                index_config=HNSWIndexConfig(metric_type=MetricType.COSINE),
                dimensions=EMBEDDING_DIM,
            )

    def test_register_model_dimension_conflict_raises(self, backend: EmbeddingBackend):
        register_flat(backend)
        with pytest.raises(Exception):
            backend.register_model(
                model_name=MODEL_NAME,
                provider_type=PROVIDER_TYPE,
                index_config=FlatIndexConfig(),
                dimensions=EMBEDDING_DIM + 1,
            )

    def test_get_registered_model_before_and_after_registration(self, backend: EmbeddingBackend):
        assert backend.get_registered_model(model_name=MODEL_NAME) is None, (
            "Model should not be registered before registration"
        )
        register_flat(backend)
        assert backend.get_registered_model(model_name=MODEL_NAME) is not None, (
            "Model should be registered after registration"
        )

    def test_get_registered_model(self, backend: EmbeddingBackend):
        register_flat(backend)
        record = backend.get_registered_model(model_name=MODEL_NAME)
        assert record is not None, "get_registered_model should return a record after registration"
        assert record.model_name == MODEL_NAME, (
            f"Returned record should have the correct model_name. Got: {record.model_name}. Expected: {MODEL_NAME}"
        )
        assert record.dimensions == EMBEDDING_DIM, (
            f"Returned record should have the correct dimensions. Got: {record.dimensions}. Expected: {EMBEDDING_DIM}"
        )

    def test_get_registered_models_returns_all(self, backend: EmbeddingBackend):
        register_flat(backend)
        records = backend.get_registered_models(model_name=MODEL_NAME, provider_type=PROVIDER_TYPE)
        assert len(records) >= 1

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def test_upsert_and_has_embeddings(self, backend: EmbeddingBackend):
        upsert_all(backend)
        assert backend.has_any_embeddings(
            model_name=MODEL_NAME,
        )

    def test_upsert_count_matches(self, backend: EmbeddingBackend):
        upsert_all(backend)
        count = backend.get_embedding_count(
            model_name=MODEL_NAME,
        )
        assert count == len(CONCEPT_RECORDS)

    def test_get_embedding_count_by_vocabulary(self, backend: EmbeddingBackend):
        upsert_all(backend)
        counts = backend.get_embedding_count_by_vocabulary(
            model_name=MODEL_NAME,
        )
        assert counts == {"SNOMED": 2, "RxNorm": 2}

    def test_upsert_is_idempotent(self, backend: EmbeddingBackend):
        upsert_all(backend)
        backend.upsert_embeddings(
            model_name=MODEL_NAME,
            records=list(CONCEPT_RECORDS),
            embeddings=CONCEPT_EMBEDDINGS,
        )
        count = backend.get_embedding_count(
            model_name=MODEL_NAME,
        )
        assert count == len(CONCEPT_RECORDS)

    def test_get_embeddings_by_concept_ids_returns_float32_arrays(self, backend: EmbeddingBackend):
        upsert_all(backend)
        result = backend.get_embeddings_by_concept_ids(
            model_name=MODEL_NAME,
            concept_ids=[HYPERTENSION_ID],
        )
        vector = result[HYPERTENSION_ID]
        assert isinstance(vector, np.ndarray)
        assert vector.dtype == np.float32
        assert np.allclose(vector, CONCEPT_EMBEDDINGS[0])

    # ------------------------------------------------------------------
    # Embedding array validation
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "embeddings, message",
        [
            (np.ones((len(CONCEPT_RECORDS), EMBEDDING_DIM + 1)), "dimensionality"),
            (np.ones(EMBEDDING_DIM), "2-D"),
            (np.ones((len(CONCEPT_RECORDS) - 1, EMBEDDING_DIM)), "does not match"),
            (np.full((len(CONCEPT_RECORDS), EMBEDDING_DIM), np.nan), "NaN or infinite"),
        ],
        ids=["wrong dimensionality", "1-D", "row count", "NaN"],
    )
    def test_upsert_rejects_malformed_embeddings(
        self, backend: EmbeddingBackend, embeddings, message
    ):
        register_flat(backend)
        with pytest.raises(ValueError, match=message):
            backend.upsert_embeddings(
                model_name=MODEL_NAME, records=list(CONCEPT_RECORDS), embeddings=embeddings
            )
        assert backend.get_embedding_count(model_name=MODEL_NAME) == 0

    @pytest.mark.parametrize(
        "query, message",
        [
            (np.ones((1, EMBEDDING_DIM + 1)), "dimensionality"),
            (np.ones(EMBEDDING_DIM), "2-D"),
            (np.array([[np.inf] * EMBEDDING_DIM]), "NaN or infinite"),
        ],
        ids=["wrong dimensionality", "1-D", "infinite"],
    )
    def test_query_rejects_malformed_embeddings(self, backend: EmbeddingBackend, query, message):
        upsert_all(backend)
        with pytest.raises(ValueError, match=message):
            backend.get_nearest_concepts(
                model_name=MODEL_NAME, metric_type=MetricType.COSINE, query_embeddings=query, k=1
            )

    def test_float64_embeddings_are_accepted_and_stored_as_float32(self, backend: EmbeddingBackend):
        register_flat(backend)
        backend.upsert_embeddings(
            model_name=MODEL_NAME,
            records=list(CONCEPT_RECORDS),
            embeddings=CONCEPT_EMBEDDINGS.astype(np.float64),
        )
        (matches,) = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.COSINE,
            query_embeddings=QUERY_EMBEDDING.astype(np.float64),
            k=1,
        )
        assert matches[0].concept_id == DIABETES_ID

    # ------------------------------------------------------------------
    # Delete embeddings
    # ------------------------------------------------------------------

    def test_delete_embeddings_removes_rows_and_counts_existing_ones(
        self, backend: EmbeddingBackend
    ):
        upsert_all(backend)
        deleted = backend.delete_embeddings(
            model_name=MODEL_NAME,
            concept_ids=[HYPERTENSION_ID, NON_STANDARD_ID, 999_999],
        )
        assert deleted == 2
        assert backend.get_stored_concept_ids(model_name=MODEL_NAME) == {DIABETES_ID, ASPIRIN_ID}
        assert backend.get_embedding_count(model_name=MODEL_NAME) == 2

    def test_delete_embeddings_removes_rows_from_search_results(self, backend: EmbeddingBackend):
        upsert_all(backend)
        backend.delete_embeddings(model_name=MODEL_NAME, concept_ids=[DIABETES_ID])
        (matches,) = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.COSINE,
            query_embeddings=QUERY_EMBEDDING,
            k=4,
        )
        assert DIABETES_ID not in {m.concept_id for m in matches}
        assert len(matches) == 3

    def test_delete_embeddings_without_ids_deletes_nothing(self, backend: EmbeddingBackend):
        upsert_all(backend)
        assert backend.delete_embeddings(model_name=MODEL_NAME, concept_ids=[]) == 0
        assert backend.get_embedding_count(model_name=MODEL_NAME) == len(CONCEPT_RECORDS)

    def test_delete_embeddings_runs_in_chunks(self, backend: EmbeddingBackend, monkeypatch):
        monkeypatch.setattr(backend, "_DELETE_CHUNK_SIZE", 1)
        upsert_all(backend)
        deleted = backend.delete_embeddings(
            model_name=MODEL_NAME,
            concept_ids=[HYPERTENSION_ID, DIABETES_ID, ASPIRIN_ID],
        )
        assert deleted == 3
        assert backend.get_stored_concept_ids(model_name=MODEL_NAME) == {NON_STANDARD_ID}

    # ------------------------------------------------------------------
    # Read: concept IDs
    # ------------------------------------------------------------------

    def test_get_stored_concept_ids_without_filter(self, backend: EmbeddingBackend):
        upsert_all(backend)
        stored = backend.get_stored_concept_ids(
            model_name=MODEL_NAME,
        )
        expected = {r.concept_id for r in CONCEPT_RECORDS}
        assert stored == expected

    def test_get_embeddings_by_concept_ids(self, backend: EmbeddingBackend):
        upsert_all(backend)
        result = backend.get_embeddings_by_concept_ids(
            model_name=MODEL_NAME,
            concept_ids=[HYPERTENSION_ID, DIABETES_ID],
        )
        assert set(result.keys()) == {HYPERTENSION_ID, DIABETES_ID}
        assert np.allclose(result[HYPERTENSION_ID], CONCEPT_EMBEDDINGS[0], atol=1e-6)
        assert np.allclose(result[DIABETES_ID], CONCEPT_EMBEDDINGS[1], atol=1e-6)

    # ------------------------------------------------------------------
    # KNN: basic
    # ------------------------------------------------------------------

    def test_knn_returns_results(self, backend: EmbeddingBackend):
        upsert_all(backend)
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.COSINE,
            query_embeddings=QUERY_EMBEDDING,
            k=3,
        )
        assert len(results) == 1
        assert len(results[0]) == 3

    def test_knn_returns_concept_filter_metadata(self, backend: EmbeddingBackend):
        """domain_id/vocabulary_id/is_standard/is_active must come back on
        every match, populated straight from the embedding table (no CDM
        engine involved at this layer)."""
        upsert_all(backend)
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.COSINE,
            query_embeddings=QUERY_EMBEDDING,
            k=len(CONCEPT_RECORDS),
        )
        matches_by_id = {m.concept_id: m for m in results[0]}
        for record in CONCEPT_RECORDS:
            match = matches_by_id[record.concept_id]
            assert match.domain_id == record.domain_id
            assert match.vocabulary_id == record.vocabulary_id
            assert match.is_standard == record.is_standard
            assert match.is_active == record.is_valid

    def test_knn_top1_is_closest(self, backend: EmbeddingBackend):
        # Query [1, 0] is closest to Diabetes (10 degrees)
        upsert_all(backend)
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.COSINE,
            query_embeddings=QUERY_EMBEDDING,
            k=1,
        )
        assert results[0][0].concept_id == DIABETES_ID

    def test_knn_exact_hypertension_query(self, backend: EmbeddingBackend):
        # Querying exactly the stored vector should return itself first
        upsert_all(backend)
        query = CONCEPT_EMBEDDINGS[:1]
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.COSINE,
            query_embeddings=query,
            k=1,
        )
        assert results[0][0].concept_id == HYPERTENSION_ID

    def test_knn_batch_queries(self, backend: EmbeddingBackend):
        upsert_all(backend)
        queries = CONCEPT_EMBEDDINGS[[0, 2]]
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.COSINE,
            query_embeddings=queries,
            k=1,
        )
        assert len(results) == 2
        assert results[0][0].concept_id == HYPERTENSION_ID
        assert results[1][0].concept_id == ASPIRIN_ID

    # ------------------------------------------------------------------
    # KNN: filters
    # ------------------------------------------------------------------

    def test_knn_domain_filter(self, backend: EmbeddingBackend):
        upsert_all(backend)
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.COSINE,
            query_embeddings=QUERY_EMBEDDING,
            concept_filter=EmbeddingConceptFilter(domains=("Drug",)),
        )
        returned_ids = {r.concept_id for r in results[0]}
        assert ASPIRIN_ID in returned_ids
        assert HYPERTENSION_ID not in returned_ids
        assert DIABETES_ID not in returned_ids

    def test_knn_vocabulary_filter(self, backend: EmbeddingBackend):
        upsert_all(backend)
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.COSINE,
            query_embeddings=QUERY_EMBEDDING,
            concept_filter=EmbeddingConceptFilter(vocabularies=("SNOMED",)),
        )
        returned_ids = {r.concept_id for r in results[0]}
        assert HYPERTENSION_ID in returned_ids
        assert DIABETES_ID in returned_ids
        assert ASPIRIN_ID not in returned_ids

    def test_knn_concept_id_filter(self, backend: EmbeddingBackend):
        upsert_all(backend)
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.COSINE,
            query_embeddings=QUERY_EMBEDDING,
            concept_filter=EmbeddingConceptFilter(concept_ids=(HYPERTENSION_ID, ASPIRIN_ID)),
        )
        returned_ids = {r.concept_id for r in results[0]}
        assert returned_ids == {HYPERTENSION_ID, ASPIRIN_ID}

    def test_knn_require_standard_filter(self, backend: EmbeddingBackend):
        upsert_all(backend)
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.COSINE,
            query_embeddings=QUERY_EMBEDDING,
            concept_filter=EmbeddingConceptFilter(require_standard=True),
        )
        returned_ids = {r.concept_id for r in results[0]}
        assert NON_STANDARD_ID not in returned_ids
        assert HYPERTENSION_ID in returned_ids

    def test_knn_k_controls_result_count_with_filter(self, backend: EmbeddingBackend):
        """k is the sole result-count control; EmbeddingConceptFilter has no such field."""
        upsert_all(backend)
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.COSINE,
            query_embeddings=QUERY_EMBEDDING,
            k=1,
            concept_filter=EmbeddingConceptFilter(vocabularies=("SNOMED",)),
        )
        assert len(results[0]) == 1
        assert results[0][0].concept_id == DIABETES_ID  # nearest SNOMED concept by cosine

    # ------------------------------------------------------------------
    # KNN: similarity math
    # ------------------------------------------------------------------

    def test_cosine_similarity_values(self, backend: EmbeddingBackend):
        """Cosine sim = 1 - dist/2 = (1 + cos(angle)) / 2 for the angles in CONCEPT_ANGLES_DEGREES."""
        upsert_all(backend)
        results = backend.get_nearest_concepts(
            model_name=MODEL_NAME,
            metric_type=MetricType.COSINE,
            query_embeddings=QUERY_EMBEDDING,
            k=len(CONCEPT_RECORDS),
        )
        expected = COSINE_SIMILARITY
        for match in results[0]:
            assert np.isclose(match.similarity, expected[match.concept_id], atol=1e-5), (
                f"concept_id={match.concept_id}: got {match.similarity}, "
                f"expected {expected[match.concept_id]}"
            )

    # ------------------------------------------------------------------
    # get_stored_concept_ids (FAISS pre-filter source)
    # ------------------------------------------------------------------

    def test_get_stored_concept_ids_domain_filter(self, backend: EmbeddingBackend):
        upsert_all(backend)
        matching = backend.get_stored_concept_ids(
            model_name=MODEL_NAME,
            concept_filter=EmbeddingConceptFilter(domains=("Drug",)),
        )
        assert matching == {ASPIRIN_ID, NON_STANDARD_ID}

    def test_get_stored_concept_ids_require_standard_filter(self, backend: EmbeddingBackend):
        upsert_all(backend)
        matching = backend.get_stored_concept_ids(
            model_name=MODEL_NAME,
            concept_filter=EmbeddingConceptFilter(require_standard=True),
        )
        assert matching == {HYPERTENSION_ID, DIABETES_ID, ASPIRIN_ID}

    def test_get_stored_concept_ids_empty_filter_returns_all(self, backend: EmbeddingBackend):
        upsert_all(backend)
        matching = backend.get_stored_concept_ids(
            model_name=MODEL_NAME,
            concept_filter=EmbeddingConceptFilter(),
        )
        assert matching == {r.concept_id for r in CONCEPT_RECORDS}

    # ------------------------------------------------------------------
    # Delete model
    # ------------------------------------------------------------------

    def test_delete_model_removes_registry_entry(self, backend: EmbeddingBackend):
        register_flat(backend)
        assert backend.get_registered_model(model_name=MODEL_NAME) is not None
        backend.delete_model(model_name=MODEL_NAME)
        assert backend.get_registered_model(model_name=MODEL_NAME) is None

    def test_patch_model_metadata(self, backend: EmbeddingBackend):
        register_flat(backend)
        backend.patch_model_metadata(
            model_name=MODEL_NAME,
            key="custom_key",
            value={"info": "test"},
        )
        record = backend.get_registered_model(
            model_name=MODEL_NAME,
        )
        assert record is not None
        assert record.metadata.get("custom_key") == {"info": "test"}
