from __future__ import annotations

import os
from typing import Optional

from .base import EmbeddingBackend
from ..config import (
    BackendType, 
    ENV_OMOP_EMB_BACKEND,
)
from .errors import (
    EmbeddingBackendConfigurationError,
    EmbeddingBackendDependencyError,
    UnknownEmbeddingBackendError,
)

def normalize_backend_name(backend_name: Optional[str]) -> BackendType:
    """
    Normalize an embedding backend name from an explicit argument or env var.

    Resolution order:
    1. explicit ``backend_name``
    2. ``OMOP_EMB_BACKEND`` environment variable
    """

    backend_name = backend_name or os.getenv(ENV_OMOP_EMB_BACKEND)
    if backend_name is None:
        raise AttributeError(f"No embedding backend specified. Provide an explicit backend_name or set the {ENV_OMOP_EMB_BACKEND} environment variable.")
    else:
        try:
            backend_type = BackendType(backend_name)
        except ValueError:
            raise UnknownEmbeddingBackendError(
                f"Unknown embedding backend {backend_name!r}. "
                f"Expected one of {[member.value for member in BackendType]}."
            )
        return backend_type


def get_embedding_backend(
    backend_name: Optional[str] = None,
    *,
    faiss_base_dir: Optional[str] = None,
) -> EmbeddingBackend:
    """
    Construct an embedding backend implementation by name.

    This factory keeps backend imports local so optional dependencies only need
    to be present when their backend is actually requested.
    """

    resolved = normalize_backend_name(backend_name)

    if resolved == BackendType.PGVECTOR:
        try:
            from .pgvector import PGVectorEmbeddingBackend
        except ImportError as exc:
            raise EmbeddingBackendDependencyError(
                "PGVector embedding backend requested but its dependencies are not "
                "available. Install the package using `pip install omop-emb[pgvector]`."
            ) from exc
        return PGVectorEmbeddingBackend()

    if resolved == BackendType.FAISS:
        try:
            from .faiss import FaissEmbeddingBackend
        except ImportError as exc:
            raise EmbeddingBackendDependencyError(
                "FAISS embedding backend requested but its dependencies are not "
                "available. Install the package with the FAISS extra using `pip install omop-emb[faiss]` or omop-emb[faiss-gpu]."
            ) from exc
        return FaissEmbeddingBackend(base_dir=faiss_base_dir)

    raise EmbeddingBackendConfigurationError(
        f"Backend factory reached an unexpected state for backend={resolved!r}."
    )
