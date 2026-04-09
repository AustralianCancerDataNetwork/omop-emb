from __future__ import annotations

import json
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from omop_emb.cli import (
    export_pgvector,
    import_pgvector,
    migrate_legacy_pgvector_registry,
)
from omop_emb.config import BackendType, IndexType
from omop_emb.interface import EmbeddingInterface

from .conftest import CONCEPTS, EMBEDDING_DIM, MODEL_NAME


@pytest.mark.pgvector
@pytest.mark.unit
def test_pgvector_export_import_roundtrip(
    session: Session,
    mock_llm_client,
    temp_storage_dir: Path,
) -> None:
    engine = session.get_bind()
    assert isinstance(engine, Engine)

    source_storage = temp_storage_dir / "source_registry"
    source_storage.mkdir(parents=True, exist_ok=True)

    interface = EmbeddingInterface.from_backend_name(
        backend_name=BackendType.PGVECTOR,
        storage_base_dir=str(source_storage),
        embedding_client=mock_llm_client,
    )
    interface.initialise_store(engine)

    interface.register_model(
        engine=engine,
        model_name=MODEL_NAME,
        dimensions=EMBEDDING_DIM,
        index_type=IndexType.FLAT,
        metadata={"origin": "roundtrip-test"},
    )

    concept_names = ["Hypertension", "Diabetes"]
    concept_ids = tuple(CONCEPTS[name].concept_id for name in concept_names)
    embeddings = mock_llm_client.embeddings(concept_names)

    interface.add_to_db(
        session=session,
        index_type=IndexType.FLAT,
        concept_ids=concept_ids,
        embeddings=embeddings,
        model=MODEL_NAME,
    )

    engine_url = engine.url.render_as_string(hide_password=False)

    snapshot_dir = temp_storage_dir / "snapshot"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OMOP_DATABASE_URL", engine_url)
        export_pgvector(
            output_dir=str(snapshot_dir),
            storage_base_dir=str(source_storage),
            model=[MODEL_NAME],
            index_type=IndexType.FLAT,
        )

    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["backend"] == BackendType.PGVECTOR.value
    assert len(manifest["tables"]) == 1

    storage_identifier = manifest["tables"][0]["storage_identifier"]
    table_name = f'"{storage_identifier}"'

    with engine.begin() as conn:
        conn.execute(sa.text(f"TRUNCATE TABLE {table_name}"))

    target_storage = temp_storage_dir / "target_registry"
    target_storage.mkdir(parents=True, exist_ok=True)

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OMOP_DATABASE_URL", engine_url)
        import_pgvector(
            input_dir=str(snapshot_dir),
            storage_base_dir=str(target_storage),
            replace=True,
            batch_size=2,
        )

    with engine.connect() as conn:
        count = int(conn.execute(sa.text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())
    assert count == len(concept_ids)


@pytest.mark.pgvector
@pytest.mark.unit
def test_migrate_legacy_pgvector_registry(
    session: Session,
    temp_storage_dir: Path,
) -> None:
    engine = session.get_bind()
    assert isinstance(engine, Engine)

    legacy_table = "model_registry_legacy"

    with engine.begin() as conn:
        conn.execute(
            sa.text(
                f"""
                CREATE TABLE {legacy_table} (
                    model_name TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    index_type TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    details JSONB
                )
                """
            )
        )
        conn.execute(
            sa.text(
                f"""
                INSERT INTO {legacy_table} (model_name, dimensions, index_type, table_name, details)
                VALUES (:model_name, :dimensions, :index_type, :table_name, CAST(:details AS JSONB))
                """
            ),
            {
                "model_name": "legacy-model",
                "dimensions": 1,
                "index_type": "flat",
                "table_name": "legacy_table_flat",
                "details": '{"migrated": true}',
            },
        )

    source_url = engine.url.render_as_string(hide_password=False)
    storage_dir = temp_storage_dir / "migrated_registry"
    storage_dir.mkdir(parents=True, exist_ok=True)

    migrate_legacy_pgvector_registry(
        storage_base_dir=str(storage_dir),
        source_database_url=source_url,
        legacy_table=legacy_table,
        dry_run=True,
        drop_legacy_registry=False,
    )

    interface = EmbeddingInterface.from_backend_name(
        backend_name=BackendType.PGVECTOR,
        storage_base_dir=str(storage_dir),
    )
    migrated_before = interface.backend.embedding_model_registry.get_registered_models_from_db(
        backend_type=BackendType.PGVECTOR,
        model_name="legacy-model",
        index_type=IndexType.FLAT,
    )
    assert migrated_before is None

    migrate_legacy_pgvector_registry(
        storage_base_dir=str(storage_dir),
        source_database_url=source_url,
        legacy_table=legacy_table,
        dry_run=False,
        drop_legacy_registry=True,
    )

    migrated = interface.backend.embedding_model_registry.get_registered_models_from_db(
        backend_type=BackendType.PGVECTOR,
        model_name="legacy-model",
        index_type=IndexType.FLAT,
    )
    assert migrated is not None
    assert len(migrated) == 1
    assert migrated[0].storage_identifier == "legacy_table_flat"

    inspector = sa.inspect(engine)
    assert not inspector.has_table(legacy_table)
