"""Compatibility shim — the backends package has moved to omop_emb.storage.

This file re-exports from the new locations so any code that still imports
from ``omop_emb.backends`` continues to work during the migration period.
Remove once all call-sites have been updated.
"""
import warnings

warnings.warn(
    "omop_emb.backends is deprecated; import from omop_emb.storage instead.",
    DeprecationWarning,
    stacklevel=2,
)

from omop_emb.storage.base import EmbeddingBackend, require_registered_model  # noqa: F401, E402
from omop_emb.storage import (  # noqa: F401, E402
    IndexConfig,
    FlatIndexConfig,
    HNSWIndexConfig,
    PGVectorEmbeddingBackend,
)

# The factory function is replaced by direct instantiation of PGVectorEmbeddingBackend.
# Kept as a stub so CLI code that referenced it doesn't import-error.
def get_embedding_backend(*args, **kwargs):  # noqa: ANN001, ANN201
    raise NotImplementedError(
        "get_embedding_backend() has been removed. "
        "Instantiate PGVectorEmbeddingBackend directly with emb_engine and omop_cdm_engine."
    )


def normalize_backend_name(*args, **kwargs):  # noqa: ANN001, ANN201
    raise NotImplementedError(
        "normalize_backend_name() has been removed. "
        "The only supported backend is PGVectorEmbeddingBackend."
    )
