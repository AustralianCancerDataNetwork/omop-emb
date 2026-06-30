"""Generic, backend-to-backend embedding export/import bundle.

A bundle is a single HDF5 file holding the raw (never normalized)
embeddings for one model, plus enough metadata to re-register and
re-import them into any :class:`EmbeddingBackend`. ``metric_type`` is not
a caller-facing export parameter -- the embeddings table has no
metric-specific columns, so it's derived internally from the registry
purely to label the bundle.

This exists purely for moving raw embeddings between backends or systems
(backup/restore, migration) -- the *bundle file* has no relationship to
FAISS. :class:`~omop_emb.storage.faiss.faiss_cache.FAISSCache` builds
directly from a live :class:`EmbeddingBackend` (see
:func:`stream_embedding_batches`, shared by both); it never reads or
writes a bundle. It does, however, share :class:`ExportMetadata` -- the
same small dataclass backs both the bundle's HDF5 attributes and the
FAISS cache's per-index JSON sidecar, since both are "facts about an
exported/built artifact" with the same shape.

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

    schema_version, omop_emb_version, model_name, dimensions, metric_type,
    provider_type, index_config, row_count, exported_at
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

from omop_emb.backends.base_backend import EmbeddingBackend
from omop_emb.backends.embedding_table import ConceptEmbeddingRecord
from omop_emb.backends.index_config import IndexConfig, index_config_from_index_type
from omop_emb.config import MetricType, ProviderType

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

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
ATTR_METRIC_TYPE = "metric_type"
ATTR_PROVIDER_TYPE = "provider_type"
ATTR_INDEX_CONFIG = "index_config"
ATTR_ROW_COUNT = "row_count"
ATTR_EXPORTED_AT = "exported_at"

REQUIRED_ATTRS = (
    ATTR_SCHEMA_VERSION,
    ATTR_OMOP_EMB_VERSION,
    ATTR_MODEL_NAME,
    ATTR_DIMENSIONS,
    ATTR_METRIC_TYPE,
    ATTR_PROVIDER_TYPE,
    ATTR_INDEX_CONFIG,
    ATTR_ROW_COUNT,
    ATTR_EXPORTED_AT,
)


class BundleCorruptionError(ValueError):
    """Raised when a bundle file is missing required fields or has inconsistent shapes."""

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
class ExportMetadata:
    """Facts about an exported/built artifact -- a bundle or a FAISS cache.

    Deliberately not :class:`EmbeddingModelRecord`: there is no live registry
    row to read when this is reconstructed from a bare file (no
    ``storage_identifier``/``created_at`` to draw on), so fabricating those
    would mean inventing placeholder values. Backs two different on-disk
    formats via two independent (de)serializers: :meth:`from_h5_attrs`
    (the bundle's HDF5 attributes) and :meth:`to_json`/:meth:`from_json`
    (the FAISS cache's per-index JSON sidecar). Each direction populates
    every field even though it only reads back some of them -- e.g. the
    bundle path never reads ``index_config`` back, the FAISS path never
    reads ``provider_type`` back -- both are free to obtain (already on
    the ``EmbeddingModelRecord`` fetched at write time), so it isn't worth
    two separate types over.
    """

    model_name: str
    dimensions: int
    metric_type: MetricType
    provider_type: ProviderType
    index_config: IndexConfig
    row_count: int
    exported_at: str

    @classmethod
    def from_h5_attrs(cls, attrs: "h5py.AttributeManager") -> "ExportMetadata":
        return cls(
            model_name=get_required_attribute(attrs, ATTR_MODEL_NAME),
            dimensions=int(get_required_attribute(attrs, ATTR_DIMENSIONS)),
            metric_type=MetricType(get_required_attribute(attrs, ATTR_METRIC_TYPE)),
            provider_type=ProviderType(get_required_attribute(attrs, ATTR_PROVIDER_TYPE)),
            index_config=index_config_from_index_type(
                **json.loads(get_required_attribute(attrs, ATTR_INDEX_CONFIG))
            ),
            row_count=int(get_required_attribute(attrs, ATTR_ROW_COUNT)),
            exported_at=str(get_required_attribute(attrs, ATTR_EXPORTED_AT)),
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "model_name": self.model_name,
                "dimensions": self.dimensions,
                "metric_type": self.metric_type.value,
                "provider_type": self.provider_type.value,
                "index_config": self.index_config.to_dict(),
                "row_count": self.row_count,
                "exported_at": self.exported_at,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "ExportMetadata":
        """Deserialise from a per-index ``.json`` sidecar string.

        Raises
        ------
        ValueError
            If the JSON is malformed or contains an unknown enum value.
        """
        d = json.loads(text)
        if "index_config" not in d:
            raise ValueError("Missing 'index_config' field in cache metadata JSON.")

        return cls(
            model_name=d.get("model_name", ""),
            dimensions=int(d.get("dimensions", 0)),
            metric_type=MetricType(d["metric_type"]),
            provider_type=ProviderType(d["provider_type"]),
            index_config=index_config_from_index_type(**d["index_config"]),
            row_count=int(d.get("row_count", -1)),
            exported_at=d.get("exported_at", ""),
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
    backend: EmbeddingBackend,
    model_name: str,
    metric_type: MetricType,
    concept_ids: Sequence[int],
    batch_size: int,
) -> Iterator[EmbeddingBatch]:
    """Stream every row for *concept_ids* out of *backend* in bounded-memory batches.

    Shared by :func:`export_bundle` (writes batches to an HDF5 bundle) and
    :meth:`~omop_emb.storage.faiss.faiss_cache.FAISSCache.build_from_backend`
    (feeds batches straight into a FAISS index) -- the only two places that
    need every embedding for a model out of the backend without loading it
    all into memory at once.
    """
    for id_batch in batched(concept_ids, batch_size):
        id_batch_list = list(id_batch)
        emb_map = backend.get_embeddings_by_concept_ids(
            model_name=model_name,
            metric_type=metric_type,
            concept_ids=id_batch_list,
        )
        filter_meta = backend.get_concept_filter_metadata(
            model_name=model_name,
            metric_type=metric_type,
            concept_ids=id_batch_list,
        )

        batch_cids = []
        batch_vecs = []
        batch_domains = []
        batch_vocabs = []
        batch_standard = []
        batch_valid = []
        for cid in id_batch_list:
            if cid not in emb_map:
                continue
            batch_cids.append(cid)
            batch_vecs.append(emb_map[cid])
            m = filter_meta.get(cid, {})
            batch_domains.append(str(m.get("domain_id", "")))
            batch_vocabs.append(str(m.get("vocabulary_id", "")))
            batch_standard.append(bool(m.get("is_standard", False)))
            batch_valid.append(bool(m.get("is_valid", True)))

        if not batch_cids:
            continue

        yield EmbeddingBatch(
            concept_ids=np.asarray(batch_cids, dtype=np.int64),
            embeddings=np.asarray(batch_vecs, dtype=np.float32),
            domain_ids=batch_domains,
            vocabulary_ids=batch_vocabs,
            is_standard=batch_standard,
            is_valid=batch_valid,
        )


def validate_bundle(f: "h5py.File") -> None:
    """Raise :class:`BundleCorruptionError` if *f* doesn't match the bundle schema."""
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
    backend: EmbeddingBackend,
    model_name: str,
    output_dir: "Path | str",
    batch_size: int = 100_000,
) -> "tuple[ExportMetadata, Path]":
    """Stream every embedding for *model_name* into one HDF5 bundle.

    Raw, never-normalized vectors are written directly into a chunked,
    disk-backed dataset batch by batch, so peak memory stays close to one
    batch's worth regardless of how many rows are exported.

    There is no ``metric_type`` parameter: the embeddings table has no
    metric-specific columns (metric only ever selects a distance operator
    at *query* time, which this function never does), so it isn't a
    caller-facing concept here. Internally, whatever the registry already
    locks the model to (or COSINE, if the model is FLAT/unconstrained) is
    used purely to satisfy the existing backend method signatures and to
    label the bundle's own metadata.

    The output filename is derived from the model's
    :attr:`EmbeddingModelRecord.storage_identifier` (already unique per
    model) under *output_dir* -- never a caller-supplied literal path.

    Returns
    -------
    tuple[ExportMetadata, Path]
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

    metric_type = record.metric_type or MetricType.COSINE
    h5_path = output_dir / f"{record.storage_identifier}.h5"

    all_ids = sorted(
        backend.get_all_stored_concept_ids(model_name=model_name, metric_type=metric_type)
    )
    if not all_ids:
        raise ValueError(
            f"No embeddings found for '{model_name}' (metric={metric_type.value}). "
            "Nothing to export."
        )

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
            stream_embedding_batches(backend, model_name, metric_type, all_ids, batch_size),
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

        meta = ExportMetadata(
            model_name=model_name,
            dimensions=dimensions,
            metric_type=metric_type,
            provider_type=record.provider_type,
            index_config=record.index_config,
            row_count=n,
            exported_at=_now_iso(),
        )
        f.attrs[ATTR_SCHEMA_VERSION] = SCHEMA_VERSION
        f.attrs[ATTR_OMOP_EMB_VERSION] = _pkg_version("omop-emb")
        f.attrs[ATTR_MODEL_NAME] = meta.model_name
        f.attrs[ATTR_DIMENSIONS] = meta.dimensions
        f.attrs[ATTR_METRIC_TYPE] = meta.metric_type.value
        f.attrs[ATTR_PROVIDER_TYPE] = meta.provider_type.value
        f.attrs[ATTR_INDEX_CONFIG] = json.dumps(meta.index_config.to_dict())
        f.attrs[ATTR_ROW_COUNT] = meta.row_count
        f.attrs[ATTR_EXPORTED_AT] = meta.exported_at

    logger.info(
        "Bundle export complete: %d vectors, metric=%s, file='%s'.",
        meta.row_count,
        metric_type.value,
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
    dataset -- never reconstructed from a FAISS index -- so magnitudes are
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
    RuntimeError
        If the backend already has embeddings for this model and
        ``force`` is ``False``.
    """
    h5_path = Path(h5_path).expanduser().resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f"Bundle file not found at '{h5_path}'.")

    with h5py.File(h5_path, "r") as f:
        validate_bundle(f)
        meta = ExportMetadata.from_h5_attrs(f.attrs)

        if not force and backend.is_model_registered(model_name=meta.model_name):
            existing = backend.get_embedding_count(
                model_name=meta.model_name, metric_type=meta.metric_type
            )
            if existing > 0:
                raise RuntimeError(
                    f"Backend already has {existing} embeddings for '{meta.model_name}'. "
                    "Pass force=True to overwrite."
                )

        was_already_registered = backend.is_model_registered(model_name=meta.model_name)
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
            metric_type=meta.metric_type,
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
        "Imported %d vectors for '%s' (metric=%s) from bundle '%s'.",
        n,
        meta.model_name,
        meta.metric_type.value,
        h5_path,
    )
    return n
