"""Legacy CLI commands for backward compatibility. These are just to support older embeddings."""

import json
import logging
import tempfile
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import h5py
import numpy as np
import typer
from tqdm import tqdm

from omop_emb.backends import resolve_backend
from omop_emb.backends.index_config import IndexConfig, index_config_from_index_type
from omop_emb.config import IndexType, MetricType, ProviderType
from omop_emb.storage import embedding_bundle

if TYPE_CHECKING:
    from omop_emb.storage.faiss import FAISSCache

logger = logging.getLogger(__name__)
app = typer.Typer(help="Legacy commands for importing pre-built embeddings.")


def _reconstruct_bundle_from_faiss_cache(
    cache: "FAISSCache",
    model_name: str,
    out_h5_path: "Path | str",
    metric_type: MetricType,
    index_config: IndexConfig,
    provider_type: ProviderType,
    batch_size: int = 100_000,
) -> embedding_bundle.ExportMetadata:
    """Rebuild an embedding bundle from a FAISS cache's on-disk index.

    Escape hatch for caches built before the bundle format existed, or for
    a standalone ``.faiss``/``metadata.npz`` pair with no backend access.
    Only ever used by ``import-legacy-faiss-cache`` -- prefer exporting
    directly from the backend via
    :func:`omop_emb.storage.embedding_bundle.export_bundle` whenever the
    backend is still reachable; it never goes through this path.

    Only ``FlatIndexConfig`` and ``HNSWIndexConfig`` support exact vector
    reconstruction; IVF/PQ indices are lossy and will raise.

    **Important**: for ``metric_type=COSINE``, the vectors reconstructed
    here are the index's L2-normalized copies, not the original raw
    embeddings -- that magnitude information was already lost when this
    FAISS cache was built. This function cannot recover it.

    Raises
    ------
    FileNotFoundError
        If the ``.faiss``/``.json``/``metadata.npz`` files do not exist.
    RuntimeError
        If the FAISS index does not support reconstruction (e.g. IVF-PQ).
    ValueError
        If the row count in ``metadata.npz`` does not match the FAISS
        index's ``ntotal``.
    """
    import faiss  # this command requires faiss-cpu, same as FAISSCache above

    json_path = cache.json_path(metric_type, index_config)
    if not json_path.exists():
        raise FileNotFoundError(
            f"FAISS index metadata not found at '{json_path}'. "
            "Run FAISSCache.build_from_backend() first."
        )
    meta = embedding_bundle.ExportMetadata.from_json(json_path.read_text())

    meta_path = cache.metadata_path()
    if not meta_path.exists():
        raise FileNotFoundError(
            f"FAISS metadata not found at '{meta_path}'. "
            "Run FAISSCache.build_from_backend() first."
        )
    npz = np.load(meta_path, allow_pickle=True)
    domain_ids_arr = npz["domain_ids"]
    vocabulary_ids_arr = npz["vocabulary_ids"]
    is_standard_arr = npz["is_standard"]
    is_valid_arr = npz["is_valid"]
    n = len(npz["concept_ids"])

    faiss_path = cache.faiss_path(metric_type, index_config)
    if not faiss_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{faiss_path}'. "
            "Run FAISSCache.build_from_backend() first."
        )
    index = faiss.read_index(str(faiss_path))
    assert isinstance(index, faiss.IndexIDMap), (
        f"FAISS index at '{faiss_path}' is not an IndexIDMap; "
        "this codebase only ever writes IndexIDMap-wrapped indices."
    )

    if index.ntotal != n:
        raise ValueError(
            f"Row count mismatch: metadata.npz has {n} concepts but "
            f"FAISS index '{faiss_path.name}' has {index.ntotal} vectors. "
            "Re-build to fix."
        )

    out_h5_path = Path(out_h5_path).expanduser().resolve()
    out_h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_h5_path, "w") as f:
        chunk_rows = min(batch_size, n) if n else 1
        emb_ds = f.create_dataset(
            embedding_bundle.EMBEDDINGS,
            shape=(n, meta.dimensions),
            dtype="float32",
            chunks=(chunk_rows, meta.dimensions),
        )
        f.create_dataset(
            embedding_bundle.CONCEPT_IDS, data=npz["concept_ids"].astype(np.int64)
        )
        f.create_dataset(
            embedding_bundle.DOMAIN_IDS,
            data=domain_ids_arr.tolist(),
            dtype=h5py.string_dtype(),
        )
        f.create_dataset(
            embedding_bundle.VOCABULARY_IDS,
            data=vocabulary_ids_arr.tolist(),
            dtype=h5py.string_dtype(),
        )
        f.create_dataset(embedding_bundle.IS_STANDARD, data=is_standard_arr.astype(bool))
        f.create_dataset(embedding_bundle.IS_VALID, data=is_valid_arr.astype(bool))

        chunk_buf = np.empty((chunk_rows, meta.dimensions), dtype=np.float32)
        for start in tqdm(
            range(0, n, chunk_rows),
            total=(n + chunk_rows - 1) // chunk_rows,
            desc="Reconstructing vectors from FAISS",
        ):
            end = min(start + chunk_rows, n)
            count = end - start
            try:
                index.index.reconstruct_n(start, count, chunk_buf[:count])
            except RuntimeError as exc:
                raise RuntimeError(
                    f"FAISS index at '{faiss_path}' does not support exact "
                    "reconstruction. Only FLAT and HNSW indices can be converted "
                    f"back into a bundle. Error: {exc}"
                ) from exc
            emb_ds[start:end] = chunk_buf[:count]

        bundle_meta = embedding_bundle.ExportMetadata(
            model_name=model_name,
            dimensions=meta.dimensions,
            metric_type=metric_type,
            provider_type=provider_type,
            index_config=index_config,
            row_count=n,
            exported_at=embedding_bundle._now_iso(),
        )
        f.attrs[embedding_bundle.ATTR_SCHEMA_VERSION] = embedding_bundle.SCHEMA_VERSION
        f.attrs[embedding_bundle.ATTR_OMOP_EMB_VERSION] = _pkg_version("omop-emb")
        f.attrs[embedding_bundle.ATTR_MODEL_NAME] = bundle_meta.model_name
        f.attrs[embedding_bundle.ATTR_DIMENSIONS] = bundle_meta.dimensions
        f.attrs[embedding_bundle.ATTR_METRIC_TYPE] = bundle_meta.metric_type.value
        f.attrs[embedding_bundle.ATTR_PROVIDER_TYPE] = bundle_meta.provider_type.value
        f.attrs[embedding_bundle.ATTR_INDEX_CONFIG] = json.dumps(bundle_meta.index_config.to_dict())
        f.attrs[embedding_bundle.ATTR_ROW_COUNT] = bundle_meta.row_count
        f.attrs[embedding_bundle.ATTR_EXPORTED_AT] = bundle_meta.exported_at

    logger.warning(
        "Reconstructed bundle '%s' from FAISS index '%s'. For metric=COSINE, "
        "the embeddings in this bundle are L2-normalized copies, not the "
        "original raw vectors -- magnitude was already lost when this FAISS "
        "cache was built.",
        out_h5_path,
        faiss_path,
    )
    return bundle_meta


@app.command(
    name="import-legacy-faiss-cache",
    help="Migrate an old-style FAISS cache (pre-bundle) into the backend.",
)
def import_legacy_faiss_cache(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Canonical model name the FAISS cache was built for.",
        ),
    ],
    cache_dir: Annotated[
        str,
        typer.Option(
            "--cache-dir",
            help="Root cache directory containing the FAISS index files.",
        ),
    ],
    provider_type: Annotated[
        ProviderType,
        typer.Option(
            "--provider-type",
            help="Embedding provider. Required to register the model if it is not already in the backend.",
        ),
    ],
    metric_type: Annotated[
        MetricType,
        typer.Option(
            "--metric-type",
            help="Distance metric of the cache to import from.",
            rich_help_panel="Index Options",
        ),
    ] = MetricType.COSINE,
    index_type: Annotated[
        IndexType,
        typer.Option(
            "--index-type",
            help="Index type to import from (FLAT or HNSW).",
            rich_help_panel="Index Options",
        ),
    ] = IndexType.FLAT,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite existing embeddings without prompting.",
        ),
    ] = False,
):
    """Migrate embeddings out of a FAISS cache built by an old version of
    ``export-faiss-cache``, for people who already have one sitting on disk.

    New exports should use ``maintenance export`` / ``maintenance import``
    instead, which carry a self-describing bundle and never lose data. This
    command reconstructs vectors from the FAISS index itself as a one-time
    migration path.
    """

    typer.secho(
        "Deprecated: this command reconstructs vectors from a FAISS index "
        "file. For metric=COSINE caches, the reconstructed vectors are "
        "L2-normalized -- they are NOT the original raw embeddings that "
        "were in the backend when this cache was built; that magnitude is "
        "already lost and cannot be recovered. New exports should use "
        "'maintenance export' / 'maintenance import', which round-trip "
        "losslessly via a bundle.",
        fg=typer.colors.YELLOW,
        err=True,
    )

    backend = resolve_backend()
    try:
        from omop_emb.storage.faiss import FAISSCache
    except ImportError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    index_config = index_config_from_index_type(index_type, metric_type=metric_type)
    cache = FAISSCache(model_name=model, cache_dir=cache_dir)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_bundle = Path(tmp_dir) / "legacy_migration.h5"
            _reconstruct_bundle_from_faiss_cache(
                cache,
                model_name=model,
                out_h5_path=tmp_bundle,
                metric_type=metric_type,
                index_config=index_config,
                provider_type=provider_type,
            )
            imported = embedding_bundle.import_bundle(
                backend=backend,
                h5_path=tmp_bundle,
                force=force,
            )
    except (RuntimeError, embedding_bundle.BundleCorruptionError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    typer.echo(
        f"Imported {imported} concepts for '{model}' "
        f"(metric={metric_type.value}, index={index_type.value})."
    )
