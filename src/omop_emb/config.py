import os
from enum import StrEnum
from typing import Dict, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from sqlalchemy import URL

ENV_DOCUMENT_EMBEDDING_PREFIX = "OMOP_EMB_DOCUMENT_EMBEDDING_PREFIX"
ENV_QUERY_EMBEDDING_PREFIX = "OMOP_EMB_QUERY_EMBEDDING_PREFIX"
ENV_EMBEDDING_DIM = "OMOP_EMB_EMBEDDING_DIM"

ENV_OMOP_EMB_BACKEND = "OMOP_EMB_BACKEND"

# Database connection — individual components (used to compose URL at runtime)
ENV_OMOP_EMB_DB_USER = "OMOP_EMB_DB_USER"
ENV_OMOP_EMB_DB_PASSWORD = "OMOP_EMB_DB_PASSWORD"
ENV_OMOP_EMB_DB_HOST = "OMOP_EMB_DB_HOST"
ENV_OMOP_EMB_DB_PORT = "OMOP_EMB_DB_PORT"
ENV_OMOP_EMB_DB_NAME = "OMOP_EMB_DB_NAME"
# Override the SQLAlchemy driver string (e.g. "postgresql+psycopg2")
ENV_OMOP_EMB_DB_DRIVER = "OMOP_EMB_DB_DRIVER"
# Optional full connection string — overrides all individual components above
ENV_OMOP_EMB_DB_URL = "OMOP_EMB_DB_URL"

# sqlite-vec backend (default)
ENV_EMB_SQLITE_PATH = "OMOP_EMB_SQLITE_PATH"

# OMOP CDM (always required for concept ingestion in CLI)
ENV_CDM_DATABASE_URL = "OMOP_CDM_DB_URL"

# Optional FAISS sidecar cache directory (auto-activates FAISS in EmbeddingReaderInterface)
ENV_FAISS_CACHE_DIR = "OMOP_EMB_FAISS_CACHE_DIR"


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


def build_engine_string(backend: "BackendType") -> "URL":
    """Compose a SQLAlchemy ``URL`` for the given backend at runtime.

    Parameters
    ----------
    backend : BackendType
        The storage backend that determines which driver and which environment
        variables are required.

    Returns
    -------
    sqlalchemy.URL

    Notes
    -----
    If ``OMOP_EMB_DB_URL`` is set it is used as-is for any backend, allowing
    callers to supply a fully-qualified connection string without setting the
    individual component variables.

    For ``SQLITEVEC``: reads ``OMOP_EMB_SQLITE_PATH``. Use the special value
    ``:memory:`` for a transient in-memory database (useful in tests).
    For ``PGVECTOR``: reads ``OMOP_EMB_DB_USER``, ``OMOP_EMB_DB_PASSWORD``,
    ``OMOP_EMB_DB_HOST``, ``OMOP_EMB_DB_NAME``, and optionally
    ``OMOP_EMB_DB_PORT`` (default 5432) and ``OMOP_EMB_DB_DRIVER`` (driver,
    default ``postgresql+psycopg``).

    Raises
    ------
    RuntimeError
        If a required environment variable is missing.
    ValueError
        If ``backend`` does not support URL composition from environment
        variables (e.g. ``FAISS``).
    """
    from sqlalchemy import URL
    from sqlalchemy.engine import make_url

    optional_url = os.getenv(ENV_OMOP_EMB_DB_URL)
    if optional_url:
        return make_url(optional_url)

    if backend == BackendType.SQLITEVEC:
        path = _get_required_env_variable(ENV_EMB_SQLITE_PATH)
        return URL.create(drivername="sqlite", database=path)

    if backend == BackendType.PGVECTOR:
        driver = os.getenv(ENV_OMOP_EMB_DB_DRIVER, "postgresql+psycopg")
        user = _get_required_env_variable(ENV_OMOP_EMB_DB_USER)
        password = _get_required_env_variable(ENV_OMOP_EMB_DB_PASSWORD)
        host = _get_required_env_variable(ENV_OMOP_EMB_DB_HOST)
        database = _get_required_env_variable(ENV_OMOP_EMB_DB_NAME)
        port_str = os.getenv(ENV_OMOP_EMB_DB_PORT)
        port = int(port_str) if port_str else None
        return URL.create(
            drivername=driver,
            username=user,
            password=password,
            host=host,
            port=port,
            database=database,
        )

    raise ValueError(
        f"Cannot compose an engine URL for backend {backend!r} from environment variables. "
        f"Set {ENV_OMOP_EMB_DB_URL!r} to supply a full connection string. "
        "FAISS is not a backend — use ENV_FAISS_CACHE_DIR and EmbeddingReaderInterface(faiss_cache_dir=...) instead."
    )


def _get_required_env_variable(name: str) -> str:
    """Get the value of a required environment variable.

    Parameters
    ----------
    name : str
        Environment variable name.

    Returns
    -------
    str
        Environment variable value.

    Raises
    ------
    RuntimeError
        If the environment variable is not set.
    """
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Required environment variable {name!r} is not set.")
    return value