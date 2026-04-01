from __future__ import annotations


class EmbeddingBackendError(RuntimeError):
    """Base class for embedding backend selection and initialization errors."""


class UnknownEmbeddingBackendError(EmbeddingBackendError):
    """Raised when a requested backend name is not recognized."""


class EmbeddingBackendDependencyError(EmbeddingBackendError, ImportError):
    """Raised when a backend was requested but its optional dependencies are missing."""


class EmbeddingBackendConfigurationError(EmbeddingBackendError):
    """Raised when backend selection or configuration is internally inconsistent."""

class ModelRegistrationConflictError(Exception):
    def __init__(self, message: str, conflict_field: str):
        super().__init__(message)
        self.conflict_field = conflict_field
