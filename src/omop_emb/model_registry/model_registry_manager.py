from __future__ import annotations

import logging
import re
from typing import Mapping, Optional

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from omop_emb.backends.index_config import (
    RESERVED_METADATA_KEYS,
    IndexConfig, 
    index_config_from_orm_row
)
from omop_emb.config import ProviderType
from omop_emb.model_registry.model_registry_orm import ModelRegistry, ensure_registry_schema
from omop_emb.model_registry.model_registry_types import EmbeddingModelRecord
from omop_emb.utils.errors import ModelRegistrationConflictError

logger = logging.getLogger(__name__)


class RegistryManager:
    """Registry of embedding models co-located with the embedding store.

    For sqlite-vec the engine points at the same .db file as the vec0 tables.
    For pgvector it points at the same Postgres database.

    Parameters
    ----------
    embedding_engine : Engine
        SQLAlchemy embedding engine connected to the embedding store.
    """

    def __init__(self, embedding_engine: Engine) -> None:
        self._embedding_engine = embedding_engine
        self._embedding_sessionmaker = sessionmaker(self._embedding_engine)

        ensure_registry_schema(embedding_engine)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def embedding_engine(self) -> Engine:
        """SQLAlchemy engine connected to the embedding store."""
        return self._embedding_engine
    
    @property
    def emb_session_factory(self) -> sessionmaker:
        """Session factory bound to ``embedding_engine``."""
        return self._embedding_sessionmaker

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_registered_models(
        self,
        *,
        model_name: Optional[str] = None,
        provider_type: Optional[ProviderType] = None,
    ) -> tuple[EmbeddingModelRecord, ...]:
        """Return all registered models matching the given filters.

        Parameters
        ----------
        model_name : str, optional
            Filter by canonical model name.
        provider_type : ProviderType, optional
            Provider that serves the model.
        Returns
        -------
        tuple[EmbeddingModelRecord, ...]
        """
        stmt = select(ModelRegistry)
        if model_name is not None:
            stmt = stmt.where(ModelRegistry.model_name == model_name)
        if provider_type is not None:
            stmt = stmt.where(ModelRegistry.provider_type == provider_type)

        with self.emb_session_factory(expire_on_commit=False) as session:
            rows = session.scalars(stmt).all()
        return tuple(self._row_to_record(r) for r in rows)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def register_model(
        self,
        *,
        model_name: str,
        index_config: IndexConfig,
        dimensions: int,
        provider_type: ProviderType,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> EmbeddingModelRecord:
        """Register a model or return the existing record if already registered.

        Parameters
        ----------
        model_name : str
            Canonical model name including tag.
        index_config : IndexConfig
            Initial index configuration.
        dimensions : int
            Embedding vector dimensionality.
        provider_type : ProviderType
            Provider that serves the model.
        metadata : Mapping[str, object], optional
            Free-form operational metadata.

        Returns
        -------
        EmbeddingModelRecord
            The newly created or already-existing record.

        Raises
        ------
        ModelRegistrationConflictError
            If the model is already registered with a different configuration.
        """
        _validate_metadata_keys(metadata)
        safe_model_name = self.safe_model_name(model_name)
        storage_identifier = self.storage_name(safe_model_name)

        with self.emb_session_factory(expire_on_commit=False) as session:
            existing = self._fetch_row(session, model_name)
            if existing is not None:
                if existing.dimensions != dimensions:
                    raise ModelRegistrationConflictError(
                        f"Model '{model_name}' already registered with "
                        f"dimensions={existing.dimensions}, got {dimensions}.",
                        conflict_field="dimensions",
                    )
                if existing.details != (metadata or {}):
                    raise ModelRegistrationConflictError(
                        f"Model '{model_name}' is already registered with different "
                        f"metadata. Reuse the existing model name or choose a new one.",
                        conflict_field="metadata"
                    )
                return self._row_to_record(existing)

        new_row = ModelRegistry(
            model_name=model_name,
            provider_type=provider_type,
            dimensions=dimensions,
            storage_identifier=storage_identifier,
            details=dict(metadata) if metadata else {},
            index_config=index_config,
        )
        with self.emb_session_factory(expire_on_commit=False) as session:
            session.add(new_row)
            session.commit()
        return self._row_to_record(new_row)

    def delete_model(self, *, model_name: str) -> None:
        """Delete a registry row. No-op if the row does not exist.

        Parameters
        ----------
        model_name : str
        """
        with self.emb_session_factory() as session:
            row = self._fetch_row(session, model_name)
            if row is not None:
                session.delete(row)
                session.commit()

    def update_index_config(
        self,
        *,
        model_name: str,
        index_config: IndexConfig,
    ) -> EmbeddingModelRecord:
        """Replace the index configuration of an existing registry row.

        Parameters
        ----------
        model_name : str
        index_config : IndexConfig
            New index configuration. The ``@validates`` hook syncs
            ``index_type`` and ``metric_type`` columns automatically.

        Returns
        -------
        EmbeddingModelRecord
            Updated record.

        Raises
        ------
        ValueError
            If the model is not registered.
        """
        with self.emb_session_factory(expire_on_commit=False) as session:
            row = self._fetch_row(session, model_name)
            if row is None:
                raise ValueError(f"Model '{model_name}' is not registered.")
            row.index_config = index_config
            session.commit()
            return self._row_to_record(row)

    def update_metadata(
        self,
        *,
        model_name: str,
        metadata: Mapping[str, object],
    ) -> EmbeddingModelRecord:
        """Replace the free-form metadata of an existing registry row.

        Parameters
        ----------
        model_name : str
        metadata : Mapping[str, object]
            New metadata dict. Replaces the existing value entirely.

        Returns
        -------
        EmbeddingModelRecord
            Updated record.

        Raises
        ------
        ValueError
            If the model is not registered.
        """
        _validate_metadata_keys(metadata)
        with self.emb_session_factory(expire_on_commit=False) as session:
            row = self._fetch_row(session, model_name)
            if row is None:
                raise ValueError(f"Model '{model_name}' is not registered.")
            row.details = dict(metadata)
            session.commit()
            return self._row_to_record(row)

    # ------------------------------------------------------------------
    # Naming helpers
    # ------------------------------------------------------------------

    @staticmethod
    def safe_model_name(model_name: str) -> str:
        """Normalise a model name for use in table identifiers.

        Lowercases and replaces any run of non-word characters with a single
        underscore, then strips leading/trailing underscores.

        Parameters
        ----------
        model_name : str
            Raw model name (e.g. ``'nomic-embed-text:v1.5'``).

        Returns
        -------
        str
            Normalised name (e.g. ``'nomic_embed_text_v1_5'``).
        """
        name = model_name.lower().strip()
        sanitized = re.sub(r"[^\w]+", "_", name)
        return re.sub(r"_+", "_", sanitized).strip("_")

    @staticmethod
    def storage_name(
        safe_model_name: str,
        embedding_table_prefix: str = "emb"
    ) -> str:
        """Return the physical table name for a model.

        Parameters
        ----------
        safe_model_name : str
            Output of :meth:`safe_model_name`.
        embedding_table_prefix : str, default "emb"
            Prefix for the embedding table name.

        Returns
        -------
        str
            Table name of the form ``<embedding_table_prefix>_<model>``.
        """
        return f"{embedding_table_prefix}_{safe_model_name}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_row(session: Session, model_name: str) -> Optional[ModelRegistry]:
        return session.scalar(
            select(ModelRegistry).where(ModelRegistry.model_name == model_name)
        )

    @staticmethod
    def _row_to_record(row: ModelRegistry) -> EmbeddingModelRecord:
        index_config = index_config_from_orm_row(row.index_type, row.index_config)
        return EmbeddingModelRecord(
            model_name=row.model_name,
            provider_type=row.provider_type,
            index_config=index_config,
            dimensions=row.dimensions,
            storage_identifier=row.storage_identifier,
            metadata=dict(row.details or {}),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _validate_metadata_keys(metadata: Optional[Mapping[str, object]]) -> None:
    """Raise ``ValueError`` if ``metadata`` contains a reserved key.

    Parameters
    ----------
    metadata : Mapping[str, object] or None
        Caller-supplied metadata dict to validate.

    Raises
    ------
    ValueError
        If any key in ``metadata`` is in ``RESERVED_METADATA_KEYS``.
    """
    if not metadata:
        return
    protected_keys_in_metadata = set(metadata.keys()) & RESERVED_METADATA_KEYS
    if protected_keys_in_metadata:
        raise ValueError(
            f"Metadata must not contain reserved keys: {sorted(protected_keys_in_metadata)}. "
            "These are managed internally by the registry."
        )
