"""Compatibility shim — the model registry has moved to omop_emb.storage.postgres.

Remove once all call-sites have been updated.
"""
import warnings

warnings.warn(
    "omop_emb.model_registry is deprecated; "
    "import EmbeddingModelRecord and PostgresRegistryManager from "
    "omop_emb.storage.postgres instead.",
    DeprecationWarning,
    stacklevel=2,
)

from omop_emb.storage.postgres.pg_registry import (  # noqa: F401, E402
    EmbeddingModelRecord,
    PostgresRegistryManager as ModelRegistryManager,
)
