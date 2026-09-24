from __future__ import annotations


class EmbeddingBackendError(RuntimeError):
    """Base class for embedding backend selection and initialization errors."""


class UnknownEmbeddingBackendError(EmbeddingBackendError):
    """Error type for an unrecognized backend name.

    Not currently raised anywhere in this package: ``resolve_backend()``
    raises a plain ``RuntimeError`` for an unknown backend name instead.
    """


class EmbeddingBackendDependencyError(EmbeddingBackendError, ImportError):
    """Error type for a backend requested without its optional dependencies installed.

    Not currently raised anywhere in this package.
    """


class EmbeddingBackendConfigurationError(EmbeddingBackendError):
    """Error type for an internally inconsistent backend selection or configuration.

    Not currently raised anywhere in this package.
    """


class ModelRegistrationConflictError(Exception):
    def __init__(self, message: str, conflict_field: str):
        super().__init__(message)
        self.conflict_field = conflict_field
