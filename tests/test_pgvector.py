"""Tests for pgvector backend."""

import pytest

from .shared_backend_tests import SharedBackendTests


@pytest.mark.pgvector
@pytest.mark.unit
class TestPGVectorBackend(SharedBackendTests):
    """Test pgvector embedding backend."""

    @pytest.fixture
    def backend(self, pgvector_backend):
        """Map the pgvector backend fixture to the generic backend fixture."""
        return pgvector_backend
