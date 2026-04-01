"""Tests for FAISS backend."""

import pytest

from .shared_backend_tests import SharedBackendTests


@pytest.mark.faiss
@pytest.mark.unit
class TestFaissBackend(SharedBackendTests):
    """Test FAISS embedding backend."""

    @pytest.fixture
    def backend(self, faiss_backend):
        """Map the FAISS backend fixture to the generic backend fixture."""
        return faiss_backend
