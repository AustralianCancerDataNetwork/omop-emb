from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from omop_emb.cli.cli_maintenance import migrate_legacy_pgvector_registry
from omop_emb.config import BackendType, IndexType, ProviderType
from omop_emb.interface import list_registered_models

from .conftest import EMBEDDING_DIM, MODEL_NAME, PROVIDER_TYPE


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
                "model_name": "legacy-model:v1",
                "dimensions": 1,
                "index_type": "flat",
                "table_name": "legacy_table_flat",
                "details": '{"migrated": true}',
            },
        )

    source_url = engine.url.render_as_string(hide_password=False)
    storage_dir = temp_storage_dir / "migrated_registry"
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Dry run — nothing should be written to the local registry
    migrate_legacy_pgvector_registry(
        storage_base_dir=str(storage_dir),
        source_database_url=source_url,
        legacy_table=legacy_table,
        dry_run=True,
        drop_legacy_registry=False,
    )

    # Verify dry run wrote nothing — use standalone list function
    migrated_before = list_registered_models(
        backend_name_or_type=BackendType.PGVECTOR,
        provider_type=PROVIDER_TYPE,
        model_name="legacy-model:v1",
        index_type=IndexType.FLAT,
        storage_base_dir=str(storage_dir),
    )
    assert len(migrated_before) == 0

    # Real migration
    migrate_legacy_pgvector_registry(
        storage_base_dir=str(storage_dir),
        source_database_url=source_url,
        legacy_table=legacy_table,
        dry_run=False,
        drop_legacy_registry=True,
    )

    # Re-read the registry
    migrated = list_registered_models(
        backend_name_or_type=BackendType.PGVECTOR,
        provider_type=PROVIDER_TYPE,
        model_name="legacy-model:v1",
        index_type=IndexType.FLAT,
        storage_base_dir=str(storage_dir),
    )
    assert len(migrated) == 1
    assert migrated[0].storage_identifier == "legacy_table_flat"

    inspector = sa.inspect(engine)
    assert not inspector.has_table(legacy_table)
