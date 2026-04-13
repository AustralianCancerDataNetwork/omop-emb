from __future__ import annotations

import os
from typing import Optional

from .base import EmbeddingBackend
from ..config import (
    BackendType, 
    ENV_OMOP_EMB_BACKEND,
    parse_backend_type,
)
from omop_emb.utils.errors import (
    EmbeddingBackendConfigurationError,
    EmbeddingBackendDependencyError,
    UnknownEmbeddingBackendError,
)

def normalize_backend_name(backend_name: Optional[str | BackendType]) -> BackendType:
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
            backend_type = parse_backend_type(backend_name)
        except ValueError:
            raise UnknownEmbeddingBackendError(
                f"Unknown embedding backend {backend_name!r}. "
                f"Expected one of {[member.value for member in BackendType]}."
            )
        return backend_type


def get_embedding_backend(
    backend_name: Optional[str | BackendType] = None,
    storage_base_dir: Optional[str] = None,
    registry_db_name: Optional[str] = None,
) -> EmbeddingBackend:
    """
    Construct an embedding backend implementation by name.

    This factory keeps backend imports local so optional dependencies only need
    to be present when their backend is actually requested.

    Resolution
    ----------
    ``backend_name`` is resolved via :func:`normalize_backend_name`:
    1. explicit ``backend_name`` argument
    2. ``OMOP_EMB_BACKEND`` environment variable

    ``storage_base_dir`` is forwarded to backend constructors, where each
    backend applies its own fallback policy (commonly explicit arg, then
    ``OMOP_EMB_BASE_STORAGE_DIR``, then backend default path).
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
        return PGVectorEmbeddingBackend(storage_base_dir=storage_base_dir, registry_db_name=registry_db_name)

    if resolved == BackendType.FAISS:
        try:
            from .faiss import FaissEmbeddingBackend
        except ImportError as exc:
            raise EmbeddingBackendDependencyError(
                "FAISS embedding backend requested but its dependencies are not "
                "available. Install the package with the FAISS extra using `pip install omop-emb[faiss]` or omop-emb[faiss-gpu]."
            ) from exc
        return FaissEmbeddingBackend(storage_base_dir=storage_base_dir, registry_db_name=registry_db_name)

    raise EmbeddingBackendConfigurationError(
        f"Backend factory reached an unexpected state for backend={resolved!r}."
    )
