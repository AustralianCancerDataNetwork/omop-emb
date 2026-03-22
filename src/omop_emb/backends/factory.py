from __future__ import annotations

import os
from typing import Optional

from .base import EmbeddingBackend
from .errors import (
    EmbeddingBackendConfigurationError,
    EmbeddingBackendDependencyError,
    UnknownEmbeddingBackendError,
)


DEFAULT_BACKEND = "postgres"
SUPPORTED_BACKENDS = ("postgres", "faiss")


def normalize_backend_name(backend_name: Optional[str]) -> str:
    """
    Normalize an embedding backend name from an explicit argument or env var.

    Resolution order:
    1. explicit ``backend_name``
    2. ``OMOP_EMB_BACKEND``
    3. ``DEFAULT_BACKEND``
    """

    resolved = (backend_name or os.getenv("OMOP_EMB_BACKEND") or DEFAULT_BACKEND).strip().lower()
    if resolved not in SUPPORTED_BACKENDS:
        raise UnknownEmbeddingBackendError(
            f"Unknown embedding backend {resolved!r}. "
            f"Expected one of {SUPPORTED_BACKENDS}."
        )
    return resolved


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

    if resolved == "postgres":
        try:
            from .postgres import PostgresEmbeddingBackend
        except ImportError as exc:
            raise EmbeddingBackendDependencyError(
                "Postgres embedding backend requested but its dependencies are not "
                "available. Install the package with the PostgreSQL embedding extras."
            ) from exc
        return PostgresEmbeddingBackend()

    if resolved == "faiss":
        try:
            from .faiss import FaissEmbeddingBackend
        except ImportError as exc:
            raise EmbeddingBackendDependencyError(
                "FAISS embedding backend requested but its dependencies are not "
                "available. Install the package with the FAISS embedding extras."
            ) from exc
        return FaissEmbeddingBackend(base_dir=faiss_base_dir)

    raise EmbeddingBackendConfigurationError(
        f"Backend factory reached an unexpected state for backend={resolved!r}."
    )
