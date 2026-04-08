"""Validation tests for strict EmbeddingInterface input contracts."""

from unittest.mock import Mock

import numpy as np
import pytest

from omop_emb.config import IndexType, MetricType
from omop_emb.interface import EmbeddingInterface


@pytest.mark.unit
class TestInterfaceValidation:
    def test_get_nearest_concepts_requires_index_type(self):
        """Index type is required as part of the strict core interface."""
        interface = EmbeddingInterface(backend=Mock(), embedding_client=Mock())
        kwargs = {
            "session": Mock(),
            "model_name": "test-model",
            "query_embedding": np.zeros((1, 1), dtype=np.float32),
            "metric_type": MetricType.COSINE,
        }

        with pytest.raises(TypeError):
            interface.get_nearest_concepts(**kwargs)

    def test_get_nearest_concepts_requires_metric_type(self):
        """Metric type is required as part of the strict core interface."""
        interface = EmbeddingInterface(backend=Mock(), embedding_client=Mock())
        kwargs = {
            "session": Mock(),
            "model_name": "test-model",
            "index_type": IndexType.FLAT,
            "query_embedding": np.zeros((1, 1), dtype=np.float32),
        }

        with pytest.raises(TypeError):
            interface.get_nearest_concepts(**kwargs)

    def test_get_nearest_concepts_rejects_non_enum_metric_type(self):
        """Core interface rejects non-MetricType values with a clear error."""
        interface = EmbeddingInterface(backend=Mock(), embedding_client=Mock())

        with pytest.raises(TypeError, match="metric_type must be MetricType"):
            interface.get_nearest_concepts(
                session=Mock(),
                model_name="test-model",
                index_type=IndexType.FLAT,
                query_embedding=np.zeros((1, 1), dtype=np.float32),
                metric_type="cosine",  # type: ignore[arg-type]
            )
