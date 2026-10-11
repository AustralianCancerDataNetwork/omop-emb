from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("pgvector", reason="omop-emb[pgvector] not installed")

from omop_emb.backends.index_config import FlatIndexConfig
from omop_emb.backends.pgvector.pg_sql import pg_embedding_table_descriptor
from omop_emb.model_registry import EmbeddingModelRecord


@pytest.mark.unit
def test_pgvector_descriptor_metadata_isolated_by_dimensions():
    record = EmbeddingModelRecord(
        model_name="descriptor-768",
        provider_type="ollama",
        index_config=FlatIndexConfig(),
        dimensions=768,
        storage_identifier="emb_descriptor",
    )
    descriptor_768 = pg_embedding_table_descriptor(record)
    descriptor_1024 = pg_embedding_table_descriptor(replace(record, dimensions=1024))

    assert descriptor_768.__table__.metadata is not descriptor_1024.__table__.metadata
    assert "VECTOR(768)" in str(descriptor_768.__table__.c.embedding.type).upper()
    assert "VECTOR(1024)" in str(descriptor_1024.__table__.c.embedding.type).upper()
