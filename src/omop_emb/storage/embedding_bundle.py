"""Generic, backend-to-backend embedding export/import bundle.

A bundle is a single HDF5 file holding the raw (never normalized)
embeddings for one model, plus enough metadata to re-register and
re-import them into any :class:`EmbeddingBackend`. It carries no metric:
stored vectors are metric-independent, and an HNSW index's metric is part
of ``index_config``.

Bundles move raw embeddings between backends or systems (backup/restore,
migration) and are unrelated to FAISS.
:class:`~omop_emb.storage.faiss.faiss_cache.FAISSCache` builds directly
from a store via :func:`stream_embedding_batches`.

Disk layout (single ``.h5`` file)
----------------------------------
Datasets (chunked along axis 0 so export/import stream batch by batch
instead of materialising the full array in memory)::

    concept_ids      int64    (n,)
    embeddings       float32  (n, dimensions)   raw, never normalized
    domain_ids       str      (n,)
    vocabulary_ids   str      (n,)
    is_standard      bool     (n,)
    is_valid         bool     (n,)

Root attributes::

    schema_version, omop_emb_version, model_name, dimensions,
    provider_type, index_config, row_count, exported_at

Schema version 1 additionally stored ``metric_type``: the HNSW metric, or a
``cosine`` placeholder for FLAT. Reading a version 1 bundle checks it
against ``index_config`` and then discards it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as _pkg_version
from itertools import batched
from pathlib import Path
from typing import Iterator, Sequence

import h5py
import numpy as np
from tqdm import tqdm

from omop_emb.backends.base_backend import EmbeddingBackend, EmbeddingStoreReader
from omop_emb.backends.embedding_table import ConceptEmbeddingRecord
from omop_emb.backends.index_config import IndexConfig, index_config_from_dict
from omop_emb.config import MetricType

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)

CONCEPT_IDS = "concept_ids"
EMBEDDINGS = "embeddings"
DOMAIN_IDS = "domain_ids"
VOCABULARY_IDS = "vocabulary_ids"
IS_STANDARD = "is_standard"
IS_VALID = "is_valid"

REQUIRED_DATASETS = (
    CONCEPT_IDS,
    EMBEDDINGS,
    DOMAIN_IDS,
    VOCABULARY_IDS,
    IS_STANDARD,
    IS_VALID,
)

ATTR_SCHEMA_VERSION = "schema_version"
ATTR_OMOP_EMB_VERSION = "omop_emb_version"
ATTR_MODEL_NAME = "model_name"
ATTR_DIMENSIONS = "dimensions"
ATTR_V1_METRIC_TYPE = "metric_type"
ATTR_PROVIDER_TYPE = "provider_type"
ATTR_INDEX_CONFIG = "index_config"
ATTR_ROW_COUNT = "row_count"
ATTR_EXPORTED_AT = "exported_at"

REQUIRED_ATTRS = (
    ATTR_SCHEMA_VERSION,
    ATTR_OMOP_EMB_VERSION,
    ATTR_MODEL_NAME,
    ATTR_DIMENSIONS,
    ATTR_PROVIDER_TYPE,
    ATTR_INDEX_CONFIG,
    ATTR_ROW_COUNT,
    ATTR_EXPORTED_AT,
)


class BundleCorruptionError(ValueError):
    """Raised when a bundle file is missing required fields or has inconsistent shapes."""


class UnsupportedBundleVersionError(BundleCorruptionError):
    """Raised when a bundle's schema_version is not one this omop-emb reads."""

def get_required_attribute(attrs: h5py.AttributeManager, attr_name: str) -> str:
    if attr_name not in REQUIRED_ATTRS:
        raise ValueError(f"Internal error: attribute '{attr_name}' is not in REQUIRED_ATTRS.")
    if attr_name not in attrs:
        raise BundleCorruptionError(
            f"Bundle is missing required attribute '{attr_name}'."
        )
    return str(attrs[attr_name])

def get_required_dataset(f: h5py.File, ds_name: str) -> h5py.Dataset:
    if ds_name not in REQUIRED_DATASETS:
        raise ValueError(f"Internal error: dataset '{ds_name}' is not in REQUIRED_DATASETS.")
    if ds_name not in f:
        raise BundleCorruptionError(
            f"Bundle is missing required dataset '{ds_name}'."
        )
    dataset = f[ds_name]
    if not isinstance(dataset, h5py.Dataset):
        raise BundleCorruptionError(
            f"Bundle item '{ds_name}' is not a dataset."
        )
    return dataset


@dataclass(frozen=True)
class BundleMetadata:
    """A bundle's root attributes.

    Attributes
    ----------
    model_name : str
    dimensions : int
    provider_type : str
    index_config : IndexConfig
        The source store's index, rebuilt by ``import_bundle(rebuild_index=True)``.
    row_count : int
    exported_at : str
        ISO timestamp of the export.
    """

    model_name: str
    dimensions: int
    provider_type: str
    index_config: IndexConfig
    row_count: int
    exported_at: str

    @classmethod
    def from_h5_attrs(cls, attrs: "h5py.AttributeManager") -> "BundleMetadata":
        """Read the attributes of a bundle of any supported schema version.

        Raises
        ------
        BundleCorruptionError
            If an attribute is missing, or a version 1 HNSW bundle's metric
            disagrees with its index_config.
        UnsupportedBundleVersionError
            If the schema version is not supported.
        """
        version = _schema_version(attrs)
        index_config_dict = json.loads(get_required_attribute(attrs, ATTR_INDEX_CONFIG))
        index_config = index_config_from_dict(index_config_dict.get("index_type"), index_config_dict)
        if version == 1:
            _check_v1_metric(attrs, index_config)
        return cls(
            model_name=get_required_attribute(attrs, ATTR_MODEL_NAME),
            dimensions=int(get_required_attribute(attrs, ATTR_DIMENSIONS)),
            provider_type=str(get_required_attribute(attrs, ATTR_PROVIDER_TYPE)),
            index_config=index_config,
            row_count=int(get_required_attribute(attrs, ATTR_ROW_COUNT)),
            exported_at=str(get_required_attribute(attrs, ATTR_EXPORTED_AT)),
        )


def _schema_version(attrs: "h5py.AttributeManager") -> int:
    """Return the bundle's schema version, raising if it is missing or unsupported."""
    version = int(get_required_attribute(attrs, ATTR_SCHEMA_VERSION))
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise UnsupportedBundleVersionError(
            f"Bundle schema version {version} is not supported; this omop-emb reads "
            f"versions {list(SUPPORTED_SCHEMA_VERSIONS)}."
        )
    return version


def _check_v1_metric(attrs: "h5py.AttributeManager", index_config: IndexConfig) -> None:
    """Check a version 1 bundle's metric_type against its index_config.

    Version 1 wrote the HNSW metric, or ``cosine`` as a placeholder for FLAT.
    """
    if ATTR_V1_METRIC_TYPE not in attrs:
        raise BundleCorruptionError(
            f"Version 1 bundle is missing required attribute '{ATTR_V1_METRIC_TYPE}'."
        )
    metric_type = MetricType(str(attrs[ATTR_V1_METRIC_TYPE]))
    if index_config.metric_type is not None and metric_type != index_config.metric_type:
        raise BundleCorruptionError(
            f"Version 1 bundle has metric_type '{metric_type.value}' but its index_config "
            f"is built for '{index_config.metric_type.value}'."
        )


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class EmbeddingBatch:
    """One bounded-memory batch yielded by :func:`stream_embedding_batches`."""

    concept_ids: np.ndarray
    embeddings: np.ndarray
    domain_ids: list[str]
    vocabulary_ids: list[str]
    is_standard: list[bool]
    is_valid: list[bool]


def stream_embedding_batches(
    backend: EmbeddingStoreReader,
    model_name: str,
    concept_ids: Sequence[int],
    batch_size: int,
) -> Iterator[EmbeddingBatch]:
    """Stream every row for concept_ids out of backend in bounded-memory batches.

    Shared by :func:`export_bundle` and
    :meth:`~omop_emb.storage.faiss.faiss_cache.FAISSCache.build_from_backend`.
    """
    for id_batch in batched(concept_ids, batch_size):
        id_batch_list = list(id_batch)
        emb_map = backend.get_embeddings_by_concept_ids(
            model_name=model_name,
            concept_ids=id_batch_list,
        )
        records = backend.get_concept_filter_metadata(
            model_name=model_name,
            concept_ids=id_batch_list,
        )
        batch_records = [records[cid] for cid in id_batch_list if cid in emb_map]
        if not batch_records:
            continue

        yield EmbeddingBatch(
            concept_ids=np.asarray([r.concept_id for r in batch_records], dtype=np.int64),
            embeddings=np.asarray([emb_map[r.concept_id] for r in batch_records], dtype=np.float32),
            domain_ids=[r.domain_id for r in batch_records],
            vocabulary_ids=[r.vocabulary_id for r in batch_records],
            is_standard=[r.is_standard for r in batch_records],
            is_valid=[r.is_valid for r in batch_records],
        )


def validate_bundle(f: "h5py.File") -> None:
    """Raise :class:`BundleCorruptionError` if *f* doesn't match the bundle schema."""
    _schema_version(f.attrs)
    missing_attrs = [a for a in REQUIRED_ATTRS if a not in f.attrs]
    missing_datasets = [d for d in REQUIRED_DATASETS if d not in f]
    if missing_attrs or missing_datasets:
        raise BundleCorruptionError(
            f"Bundle '{f.filename}' does not match the expected schema. "
            f"Missing datasets: {missing_datasets or 'none'}. "
            f"Missing attributes: {missing_attrs or 'none'}."
        )

    row_count = int(get_required_attribute(f.attrs, ATTR_ROW_COUNT))
    for ds_name in REQUIRED_DATASETS:
        dataset = get_required_dataset(f, ds_name)
        if dataset.shape[0] != row_count:
            raise BundleCorruptionError(
                f"Bundle '{f.filename}': dataset '{ds_name}' has "
                f"{dataset.shape[0]} rows but 'row_count' attribute says {row_count}."
            )

    dimensions = int(get_required_attribute(f.attrs, ATTR_DIMENSIONS))
    embeddings_dataset = get_required_dataset(f, EMBEDDINGS)
    if embeddings_dataset.shape[1] != dimensions:
        raise BundleCorruptionError(
            f"Bundle '{f.filename}': '{EMBEDDINGS}' dataset has dimensionality "
            f"{embeddings_dataset.shape[1]} but 'dimensions' attribute says {dimensions}."
        )


def export_bundle(
    backend: EmbeddingStoreReader,
    model_name: str,
    output_dir: "Path | str",
    batch_size: int = 100_000,
) -> "tuple[BundleMetadata, Path]":
    """Stream every embedding for *model_name* into one HDF5 bundle.

    Raw, never-normalized vectors are written directly into a chunked,
    disk-backed dataset batch by batch, so peak memory stays close to one
    batch's worth regardless of how many rows are exported.

    The output filename is derived from the model's
    :attr:`EmbeddingModelRecord.storage_identifier` (already unique per
    model) under *output_dir*: never a caller-supplied literal path.

    Returns
    -------
    tuple[BundleMetadata, Path]
        The written bundle's metadata and the path it was written to.

    Raises
    ------
    ValueError
        If the model is not registered, or has no stored embeddings.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    record = backend.get_registered_model(model_name=model_name)
    if record is None:
        raise ValueError(f"Model '{model_name}' is not registered in the backend.")

    h5_path = output_dir / f"{record.storage_identifier}.h5"

    all_ids = sorted(backend.get_stored_concept_ids(model_name=model_name))
    if not all_ids:
        raise ValueError(f"No embeddings found for '{model_name}'. Nothing to export.")

    n = len(all_ids)
    dimensions = record.dimensions

    with h5py.File(h5_path, "w") as f:
        chunk_rows = min(batch_size, n)
        emb_ds = f.create_dataset(
            EMBEDDINGS,
            shape=(n, dimensions),
            maxshape=(n, dimensions),
            dtype="float32",
            chunks=(chunk_rows, dimensions),
        )
        cid_ds = f.create_dataset(
            CONCEPT_IDS, shape=(n,), maxshape=(n,), dtype="int64", chunks=(chunk_rows,)
        )
        domain_ds = f.create_dataset(
            DOMAIN_IDS,
            shape=(n,),
            maxshape=(n,),
            dtype=h5py.string_dtype(),
            chunks=(chunk_rows,),
        )
        vocab_ds = f.create_dataset(
            VOCABULARY_IDS,
            shape=(n,),
            maxshape=(n,),
            dtype=h5py.string_dtype(),
            chunks=(chunk_rows,),
        )
        standard_ds = f.create_dataset(
            IS_STANDARD, shape=(n,), maxshape=(n,), dtype="bool", chunks=(chunk_rows,)
        )
        valid_ds = f.create_dataset(
            IS_VALID, shape=(n,), maxshape=(n,), dtype="bool", chunks=(chunk_rows,)
        )

        cursor = 0
        for batch in tqdm(
            stream_embedding_batches(backend, model_name, all_ids, batch_size),
            total=(n + batch_size - 1) // batch_size,
            desc="Streaming export to bundle",
        ):
            batch_n = len(batch.concept_ids)
            end = cursor + batch_n
            emb_ds[cursor:end] = batch.embeddings
            cid_ds[cursor:end] = batch.concept_ids
            domain_ds[cursor:end] = batch.domain_ids
            vocab_ds[cursor:end] = batch.vocabulary_ids
            standard_ds[cursor:end] = batch.is_standard
            valid_ds[cursor:end] = batch.is_valid
            cursor = end

        if cursor < n:
            logger.warning(
                "%d concept id(s) reported as stored had no embedding row; "
                "truncating bundle from %d to %d rows.",
                n - cursor,
                n,
                cursor,
            )
            for ds_name in REQUIRED_DATASETS:
                ds = get_required_dataset(f, ds_name)
                ds.resize((cursor,) + ds.shape[1:])
            n = cursor

        meta = BundleMetadata(
            model_name=model_name,
            dimensions=dimensions,
            provider_type=record.provider_type,
            index_config=record.index_config,
            row_count=n,
            exported_at=_now_iso(),
        )
        f.attrs[ATTR_SCHEMA_VERSION] = SCHEMA_VERSION
        f.attrs[ATTR_OMOP_EMB_VERSION] = _pkg_version("omop-emb")
        f.attrs[ATTR_MODEL_NAME] = meta.model_name
        f.attrs[ATTR_DIMENSIONS] = meta.dimensions
        f.attrs[ATTR_PROVIDER_TYPE] = meta.provider_type
        f.attrs[ATTR_INDEX_CONFIG] = json.dumps(meta.index_config.to_dict())
        f.attrs[ATTR_ROW_COUNT] = meta.row_count
        f.attrs[ATTR_EXPORTED_AT] = meta.exported_at

    logger.info(
        "Bundle export complete: %d vectors, file='%s'.",
        meta.row_count,
        h5_path,
    )
    return meta, h5_path


def import_bundle(
    backend: EmbeddingBackend,
    h5_path: "Path | str",
    force: bool = False,
    batch_size: int = 10_000,
    rebuild_index: bool = False,
) -> int:
    """Stream every row of a bundle produced by :func:`export_bundle` into *backend*.

    Registers the model from the bundle's attributes if it isn't already
    registered. Vectors are read straight from the bundle's ``embeddings``
    dataset: never reconstructed from a FAISS index: so magnitudes are
    preserved exactly regardless of metric.

    A brand-new registration is backdated to the bundle's own ``exported_at``
    (the moment the *source* data was snapshotted) rather than "now", so a
    FAISS cache built after that point on the source machine still validates
    as fresh once shipped alongside this bundle and imported elsewhere.

    Pass ``rebuild_index=True`` to build the index recorded in the bundle's
    ``index_config`` right after the vectors land (registration itself only
    ever creates a FLAT index).

    Raises
    ------
    FileNotFoundError
        If *h5_path* does not exist.
    BundleCorruptionError
        If the file is missing required datasets/attributes or has
        inconsistent shapes.
    UnsupportedBundleVersionError
        If the bundle's schema version is not supported.
    RuntimeError
        If the backend already has embeddings for this model and
        ``force`` is ``False``.
    """
    h5_path = Path(h5_path).expanduser().resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f"Bundle file not found at '{h5_path}'.")

    with h5py.File(h5_path, "r") as f:
        validate_bundle(f)
        meta = BundleMetadata.from_h5_attrs(f.attrs)

        was_already_registered = backend.get_registered_model(model_name=meta.model_name) is not None
        if not force and was_already_registered:
            existing = backend.get_embedding_count(model_name=meta.model_name)
            if existing > 0:
                raise RuntimeError(
                    f"Backend already has {existing} embeddings for '{meta.model_name}'. "
                    "Pass force=True to overwrite."
                )

        if not was_already_registered:
            backend.register_model(
                model_name=meta.model_name,
                provider_type=meta.provider_type,
                dimensions=meta.dimensions,
                registered_at=datetime.fromisoformat(meta.exported_at),
            )

        emb_ds = get_required_dataset(f, EMBEDDINGS)
        cid_ds = get_required_dataset(f, CONCEPT_IDS)
        domain_ds = get_required_dataset(f, DOMAIN_IDS).asstr()
        vocab_ds = get_required_dataset(f, VOCABULARY_IDS).asstr()
        standard_ds = get_required_dataset(f, IS_STANDARD)
        valid_ds = get_required_dataset(f, IS_VALID)
        n = meta.row_count

        def _batches():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                cids = cid_ds[start:end]
                domains = domain_ds[start:end]
                vocabs = vocab_ds[start:end]
                standards = standard_ds[start:end]
                valids = valid_ds[start:end]
                records = [
                    ConceptEmbeddingRecord(
                        concept_id=int(cids[i]),
                        domain_id=str(domains[i]),
                        vocabulary_id=str(vocabs[i]),
                        is_standard=bool(standards[i]),
                        is_valid=bool(valids[i]),
                    )
                    for i in range(end - start)
                ]
                yield records, np.asarray(emb_ds[start:end], dtype=np.float32)

        backend.bulk_upsert_embeddings(
            model_name=meta.model_name,
            batches=_batches(),
            total_n_batches=(n + batch_size - 1) // batch_size,
        )

        if was_already_registered:
            backend.refresh_model_updated_at_timestamp(model_name=meta.model_name)

    if rebuild_index:
        backend.rebuild_index(model_name=meta.model_name, index_config=meta.index_config)
        logger.info(
            "Rebuilt index (%s) for '%s'.",
            meta.index_config.index_type.value,
            meta.model_name,
        )

    logger.info(
        "Imported %d vectors for '%s' from bundle '%s'.",
        n,
        meta.model_name,
        h5_path,
    )
    return n
