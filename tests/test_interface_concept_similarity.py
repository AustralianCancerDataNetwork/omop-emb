"""Tests for EmbeddingReaderInterface.get_joint_embedding / get_similar_concepts."""

from unittest.mock import Mock

import numpy as np
import pytest

from omop_emb.config import MetricType
from omop_emb.interface import EmbeddingReaderInterface, _resolve_k
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter, NearestConceptMatch


def _make_backend() -> Mock:
    backend = Mock()
    backend.backend_name = "pgvector"
    backend.get_registered_model.return_value = None
    return backend


def _make_interface(backend: Mock, metric_type: MetricType) -> EmbeddingReaderInterface:
    return EmbeddingReaderInterface(
        model="test-model:v1",
        backend=backend,
        metric_type=metric_type,
    )


@pytest.mark.unit
class TestResolveK:
    def test_explicit_k_wins(self):
        assert _resolve_k(5, EmbeddingConceptFilter(limit=3), default=1) == 5

    def test_falls_back_to_filter_limit(self):
        assert _resolve_k(None, EmbeddingConceptFilter(limit=3), default=1) == 3

    def test_falls_back_to_default(self):
        assert _resolve_k(None, None, default=1) == 1


@pytest.mark.unit
class TestGetJointEmbedding:
    def test_unweighted_mean(self):
        backend = _make_backend()
        backend.get_embeddings_by_concept_ids.return_value = {
            1: [1.0, 0.0],
            2: [0.0, 1.0],
        }
        interface = _make_interface(backend, MetricType.L2)

        centroid = interface.get_joint_embedding((1, 2))

        np.testing.assert_allclose(centroid, [0.5, 0.5])

    def test_weighted_mean(self):
        backend = _make_backend()
        backend.get_embeddings_by_concept_ids.return_value = {
            1: [1.0, 0.0],
            2: [0.0, 1.0],
        }
        interface = _make_interface(backend, MetricType.L2)

        centroid = interface.get_joint_embedding((1, 2), weights=(3.0, 1.0))

        np.testing.assert_allclose(centroid, [0.75, 0.25])

    def test_not_normalised_for_cosine_metric(self):
        backend = _make_backend()
        backend.get_embeddings_by_concept_ids.return_value = {
            1: [3.0, 0.0],
            2: [1.0, 0.0],
        }
        interface = _make_interface(backend, MetricType.COSINE)

        centroid = interface.get_joint_embedding((1, 2))

        # Backends compute cosine distance directly from raw vectors, so the
        # centroid is left as the plain mean regardless of metric_type.
        np.testing.assert_allclose(centroid, [2.0, 0.0])

    def test_empty_concept_ids_raises(self):
        interface = _make_interface(_make_backend(), MetricType.L2)

        with pytest.raises(ValueError, match="non-empty"):
            interface.get_joint_embedding(())

    def test_mismatched_weights_length_raises(self):
        interface = _make_interface(_make_backend(), MetricType.L2)

        with pytest.raises(ValueError, match="same length"):
            interface.get_joint_embedding((1, 2), weights=(1.0,))

    def test_zero_sum_weights_raises(self):
        interface = _make_interface(_make_backend(), MetricType.L2)

        with pytest.raises(ValueError, match="sum to zero"):
            interface.get_joint_embedding((1, 2), weights=(1.0, -1.0))

    def test_missing_embedding_raises(self):
        backend = _make_backend()
        backend.get_embeddings_by_concept_ids.return_value = {1: [1.0, 0.0]}
        interface = _make_interface(backend, MetricType.L2)

        with pytest.raises(ValueError, match="No stored embedding"):
            interface.get_joint_embedding((1, 2))


@pytest.mark.unit
class TestGetSimilarConcepts:
    def test_bare_int_returns_single_row_excluding_self(self):
        backend = _make_backend()
        backend.get_embeddings_by_concept_ids.return_value = {1: [1.0, 0.0]}
        backend.get_nearest_concepts.return_value = (
            (
                NearestConceptMatch(concept_id=1, similarity=1.0),
                NearestConceptMatch(concept_id=2, similarity=0.9),
                NearestConceptMatch(concept_id=3, similarity=0.8),
            ),
        )
        interface = _make_interface(backend, MetricType.COSINE)

        result = interface.get_similar_concepts(1, k=2)

        assert len(result) == 1
        assert [m.concept_id for m in result[0]] == [2, 3]
        # requests k+1 so the self-match can be dropped without losing a slot
        assert backend.get_nearest_concepts.call_args.kwargs["k"] == 3

    def test_truncates_to_k_when_no_self_match_present(self):
        backend = _make_backend()
        backend.get_embeddings_by_concept_ids.return_value = {1: [1.0, 0.0]}
        backend.get_nearest_concepts.return_value = (
            (
                NearestConceptMatch(concept_id=2, similarity=0.9),
                NearestConceptMatch(concept_id=3, similarity=0.8),
                NearestConceptMatch(concept_id=4, similarity=0.7),
            ),
        )
        interface = _make_interface(backend, MetricType.COSINE)

        result = interface.get_similar_concepts(1, k=2)

        assert [m.concept_id for m in result[0]] == [2, 3]

    def test_sequence_of_ids_returns_one_row_per_id_in_order(self):
        backend = _make_backend()
        backend.get_embeddings_by_concept_ids.return_value = {
            1: [1.0, 0.0],
            2: [0.0, 1.0],
        }
        backend.get_nearest_concepts.return_value = (
            (
                NearestConceptMatch(concept_id=1, similarity=1.0),
                NearestConceptMatch(concept_id=3, similarity=0.5),
            ),
            (
                NearestConceptMatch(concept_id=2, similarity=1.0),
                NearestConceptMatch(concept_id=4, similarity=0.4),
            ),
        )
        interface = _make_interface(backend, MetricType.COSINE)

        result = interface.get_similar_concepts((1, 2), k=1)

        assert len(result) == 2
        assert [m.concept_id for m in result[0]] == [3]
        assert [m.concept_id for m in result[1]] == [4]

    def test_empty_sequence_raises(self):
        interface = _make_interface(_make_backend(), MetricType.COSINE)

        with pytest.raises(ValueError, match="non-empty"):
            interface.get_similar_concepts(())

    def test_missing_embedding_raises(self):
        backend = _make_backend()
        backend.get_embeddings_by_concept_ids.return_value = {1: [1.0, 0.0]}
        interface = _make_interface(backend, MetricType.COSINE)

        with pytest.raises(ValueError, match="No stored embedding"):
            interface.get_similar_concepts((1, 2), k=1)

    def test_concept_filter_limit_used_as_fallback_k(self):
        backend = _make_backend()
        backend.get_embeddings_by_concept_ids.return_value = {1: [1.0, 0.0]}
        backend.get_nearest_concepts.return_value = (
            (
                NearestConceptMatch(concept_id=1, similarity=1.0),
                NearestConceptMatch(concept_id=2, similarity=0.9),
                NearestConceptMatch(concept_id=3, similarity=0.8),
            ),
        )
        interface = _make_interface(backend, MetricType.COSINE)

        result = interface.get_similar_concepts(1, concept_filter=EmbeddingConceptFilter(limit=2))

        assert [m.concept_id for m in result[0]] == [2, 3]
        assert backend.get_nearest_concepts.call_args.kwargs["k"] == 3
