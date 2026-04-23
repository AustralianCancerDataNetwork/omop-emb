from __future__ import annotations

from dataclasses import dataclass, asdict, fields, is_dataclass
from typing import Mapping, Optional, Any, Self, get_type_hints
from enum import Enum

from omop_emb.config import IndexType

INDEX_CONFIG_METADATA_KEY = "index_config"


class IndexConfig:
    """Base Provider for index configurations."""
    index_type: IndexType

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to JSON-compatible dict (Enums become strings)."""
        if not is_dataclass(self):
            raise TypeError(f"to_dict() should be called on dataclass instances, not {type(self).__name__}")
        return asdict(self)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> Self:
        """
        Strictly instantiates the config from a JSON metadata dictionary.
        Handles coercion of strings back into Enums.
        """
        if not is_dataclass(cls):
            raise TypeError(f"Must be called on a dataclass, not {cls.__name__}")

        config_data = metadata.get(INDEX_CONFIG_METADATA_KEY)
        if config_data is None:
            raise ValueError("Metadata is missing 'index_config'.")

        type_hints = get_type_hints(cls)
        init_params = {}

        for field in fields(cls):
            if field.name not in config_data:
                raise ValueError(f"Strict Check Failed: '{field.name}' missing in JSON.")

            value = config_data[field.name]
            field_type = type_hints[field.name]

            if isinstance(field_type, type) and issubclass(field_type, Enum):
                try:
                    value = field_type(value)
                except ValueError:
                    raise ValueError(f"Invalid Enum value '{value}' for {field_type.__name__}")
            
            init_params[field.name] = value

        return cls(**init_params)

@dataclass(frozen=True)
class FlatIndexConfig(IndexConfig):
    index_type: IndexType = IndexType.FLAT

@dataclass(frozen=True)
class HNSWIndexConfig(IndexConfig):
    num_neighbors: int = 32
    ef_search: int = 16
    ef_construction: int = 64
    index_type: IndexType = IndexType.HNSW


def index_config_from_index_type_and_metadata(
    index_type: IndexType,
    metadata: Optional[Mapping[str, Any]] = None,
) -> IndexConfig:
    """Reconstruct an IndexConfig from a persisted index type and metadata dict.

    Falls back to the dataclass defaults for any missing metadata key, so this
    is also safe to call when no metadata was ever written (e.g. legacy FLAT records).
    """
    if index_type == IndexType.FLAT:
        return FlatIndexConfig()
    if index_type == IndexType.HNSW:
        meta = metadata or {}
        return HNSWIndexConfig.from_metadata(meta)
    raise ValueError(
        f"No IndexConfig defined for index type {index_type!r}. "
        f"Pass an explicit IndexConfig instead."
    )
