from __future__ import annotations


class EmbeddingBackendError(RuntimeError):
    """Base class for embedding backend selection and initialization errors."""


class UnknownEmbeddingBackendError(EmbeddingBackendError, ValueError):
    """An unrecognized backend name."""


class EmbeddingBackendDependencyError(EmbeddingBackendError, ImportError):
    """A backend was requested without its optional dependencies installed."""


class EmbeddingBackendConfigurationError(EmbeddingBackendError):
    """A backend was selected for a database it cannot run on."""


class MisplacedRegistryError(EmbeddingBackendError):
    """A model registry table exists outside the registry schema, where it would be ignored."""


class ModelRegistrationConflictError(Exception):
    def __init__(self, message: str, conflict_field: str):
        super().__init__(message)
        self.conflict_field = conflict_field


class MissingStorageTableError(EmbeddingBackendError):
    """A model is registered but its physical storage table does not exist.

    Signals a divergence between the registry and the physical store that needs
    manual investigation. The store never silently recreates a table it did not
    just create itself via register_model.
    """


class ReadOnlyStoreError(EmbeddingBackendError):
    """A write was attempted on a store opened with open_vector_store_reader()."""
