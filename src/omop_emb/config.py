"""Configuration for omop-emb via oa-configurator."""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Dict, Final, Tuple

from pydantic import Field
from sqlalchemy import Engine
from oa_configurator import PackageConfigBase, ResourceSpec, Resolver, load_stack_config
from oa_configurator import configure_logging as _configure_logging
from omop_alchemy.config import CDM_DB_RESOURCE

EMB_DB_RESOURCE: Final[str] = "emb_db"
TOOL_NAME: Final[str] = "omop_emb"


class OmopEmbConfig(PackageConfigBase):
    """oa-configurator config class for omop-emb.

    omop-emb owns the embedding database and requires the CDM database
    configured by omop-alchemy.
    """

    tool_name: ClassVar[str] = TOOL_NAME
    required_resources: ClassVar[tuple[str, ...]] = (CDM_DB_RESOURCE,)
    owned_resources: ClassVar[tuple[ResourceSpec, ...]] = (
        ResourceSpec(
            semantic_name=EMB_DB_RESOURCE,
            display_name="Embedding Database",
            description="pgvector database for storing OMOP concept embeddings.",
            connection_name_hint="emb",
            cdm_schema_default="public",
            is_cdm_database=False,
        ),
    )

    backend: str = Field(
        default="pgvector",
        description="Embedding storage backend: 'pgvector' or 'sqlitevec'.",
    )
    sqlite_path: str | None = Field(
        default=None,
        description="Path to the SQLite database file (sqlitevec backend only).",
    )
    document_embedding_prefix: str = Field(
        default="",
        description="Text prefix prepended to documents before embedding.",
    )
    query_embedding_prefix: str = Field(
        default="",
        description="Text prefix prepended to queries before embedding.",
    )
    faiss_cache_dir: str | None = Field(
        default=None,
        description="Default directory for FAISS index files.",
    )
    embedding_dim: int | None = Field(
        default=None,
        description="Embedding dimensionality hint (rarely needed; usually auto-discovered from the model API).",
    )
    ollama_api_base: str = Field(
        default="http://ollama:11434/v1",
        description="Base URL for the Ollama API (OpenAI-compatible).",
    )
    api_key: str = Field(
        default="ollama",
        description="API key for the model provider ('ollama' for local Ollama).",
    )


def get_resolver() -> Resolver:
    """Return a Resolver loaded from the active stack config."""
    return Resolver(load_stack_config())


def get_config() -> OmopEmbConfig:
    """Return the omop-emb typed config from the active stack config."""
    return OmopEmbConfig.from_stack(load_stack_config())

def resolve_omop_cdm_engine() -> Engine:
    """Resolve CDM engine via oa-configurator, used read-only."""
    return get_resolver().resolve_resource(CDM_DB_RESOURCE).create_engine()

def resolve_omop_emb_engine() -> Engine:
    """Resolve embedding database engine via oa-configurator."""
    return get_resolver().resolve_resource(EMB_DB_RESOURCE).create_engine()

def configure_logging(verbosity: int = 0) -> None:
    """Configure logging for omop-emb and its dependencies."""
    _configure_logging(verbosity=verbosity, extra_namespaces=["omop_alchemy", TOOL_NAME])


class ProviderType(StrEnum):
    """Embedding model provider.

    Members
    -------
    OLLAMA
        Self-hosted models served via the Ollama runtime.
    """

    OLLAMA = "ollama"


class BackendType(StrEnum):
    """Embedding storage backend.

    Members
    -------
    SQLITEVEC
        Default backend. Requires no external database server.
    PGVECTOR
        Optional backend. Requires a PostgreSQL instance with the pgvector
        extension (``pip install omop-emb[pgvector]``).
    """

    SQLITEVEC = "sqlitevec"
    PGVECTOR = "pgvector"


class VectorColumnType(StrEnum):
    """PostgreSQL column type used to store embedding vectors (pgvector only).

    Selected automatically by the pgvector backend based on dimensionality.
    Not applicable to the sqlite-vec backend.

    Members
    -------
    VECTOR
        ``pgvector`` ``vector`` type (float32, up to 2 000 dims).
    HALFVEC
        ``pgvector`` ``halfvec`` type (float16, up to 4 000 dims).

    Notes
    -----
    .. TODO: Include the storage options also for sqlite-vec
       https://alexgarcia.xyz/sqlite-vec/guides/scalar-quant.html
    """

    VECTOR = "vector"
    HALFVEC = "halfvec"


PGVECTOR_VECTOR_MAX_DIMENSIONS = 2_000
PGVECTOR_HALFVEC_MAX_DIMENSIONS = 4_000


class IndexType(StrEnum):
    """Index structure built on an embedding table for nearest-neighbor search.

    Members
    -------
    FLAT
        Exact sequential scan. Always correct, slower at scale.
    HNSW
        Approximate nearest-neighbor index. Supported by pgvector and FAISS
        only, not by sqlite-vec.
    """

    FLAT = "flat"
    HNSW = "hnsw"
    IVFFLAT = "ivfflat"  # faiss only
    IVFPQ = "ivfpq"  # faiss only


class MetricType(StrEnum):
    """Distance metric used for nearest-neighbor search.

    Members
    -------
    L2
        Euclidean (L2) distance.
    COSINE
        Cosine distance.
    L1
        Manhattan (L1) distance.
    HAMMING
        Hamming distance (bit vectors).
    JACCARD
        Jaccard distance (bit vectors).
    """

    L2 = "l2"
    COSINE = "cosine"
    L1 = "l1"
    HAMMING = "hamming"
    JACCARD = "jaccard"


def parse_backend_type(value: str | BackendType) -> BackendType:
    """Parse a string or ``BackendType`` into a ``BackendType``.

    Parameters
    ----------
    value : str | BackendType
        Backend identifier string or enum member.

    Returns
    -------
    BackendType

    Raises
    ------
    ValueError
        If ``value`` is not a recognised backend type.
    """
    if isinstance(value, BackendType):
        return value
    try:
        return BackendType(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid backend type {value!r}. Expected one of "
            f"{[member.value for member in BackendType]}."
        ) from exc


def parse_index_type(value: str | IndexType) -> IndexType:
    """Parse a string or ``IndexType`` into an ``IndexType``.

    Parameters
    ----------
    value : str | IndexType
        Index type identifier string or enum member.

    Returns
    -------
    IndexType

    Raises
    ------
    ValueError
        If ``value`` is not a recognised index type.
    """
    if isinstance(value, IndexType):
        return value
    try:
        return IndexType(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid index type {value!r}. Expected one of "
            f"{[member.value for member in IndexType]}."
        ) from exc


def parse_metric_type(value: str | MetricType) -> MetricType:
    """Parse a string or ``MetricType`` into a ``MetricType``.

    Parameters
    ----------
    value : str | MetricType
        Metric type identifier string or enum member.

    Returns
    -------
    MetricType

    Raises
    ------
    ValueError
        If ``value`` is not a recognised metric type.
    """
    if isinstance(value, MetricType):
        return value
    try:
        return MetricType(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid metric type {value!r}. Expected one of "
            f"{[member.value for member in MetricType]}."
        ) from exc


# Supported index + metric combinations per backend.
SUPPORTED_INDICES_AND_METRICS_PER_BACKEND: Dict[BackendType, Dict[IndexType, Tuple[MetricType, ...]]] = {
    # https://alexgarcia.xyz/sqlite-vec/features/vec0.html
    BackendType.SQLITEVEC: {
        IndexType.FLAT: (MetricType.L2, MetricType.COSINE, MetricType.L1),
    },
    # https://github.com/pgvector/pgvector?tab=readme-ov-file#querying
    BackendType.PGVECTOR: {
        IndexType.FLAT: (MetricType.L2, MetricType.COSINE, MetricType.L1),
        IndexType.HNSW: (MetricType.L2, MetricType.COSINE, MetricType.L1),
    },
}


def is_supported_index_metric_combination_for_backend(
    backend: BackendType, index: IndexType, metric: MetricType
) -> bool:
    """Return whether a backend supports a given index and metric combination.

    Parameters
    ----------
    backend : BackendType
    index : IndexType
    metric : MetricType

    Returns
    -------
    bool
    """
    supported = SUPPORTED_INDICES_AND_METRICS_PER_BACKEND.get(backend, {})
    return metric in supported.get(index, ())


def is_index_type_supported_for_backend(backend: BackendType, index: IndexType) -> bool:
    """Return whether a backend supports a given index type.

    Parameters
    ----------
    backend : BackendType
    index : IndexType

    Returns
    -------
    bool
    """
    return index in SUPPORTED_INDICES_AND_METRICS_PER_BACKEND.get(backend, {})


def get_supported_index_types_for_backend(backend: BackendType) -> Tuple[IndexType, ...]:
    """Return all index types supported by a backend.

    Parameters
    ----------
    backend : BackendType

    Returns
    -------
    tuple[IndexType, ...]
    """
    return tuple(SUPPORTED_INDICES_AND_METRICS_PER_BACKEND.get(backend, {}).keys())


def get_supported_metrics_for_backend(backend: BackendType) -> Tuple[MetricType, ...]:
    """Return all metrics supported by any index type for a backend.

    Parameters
    ----------
    backend : BackendType

    Returns
    -------
    tuple[MetricType, ...]
    """
    seen: set[MetricType] = set()
    for metrics in SUPPORTED_INDICES_AND_METRICS_PER_BACKEND.get(backend, {}).values():
        seen.update(metrics)
    return tuple(seen)
