"""Missing-embedding lookups through the reader and its storage backends."""

from datetime import date
from unittest.mock import Mock

import pytest
from omop_alchemy.cdm.model.vocabulary import Concept
from omop_alchemy.cdm.query import ConceptFilter
from omop_alchemy.cross_database import cdm_sessionmaker
from sqlalchemy import insert

from omop_emb.backends.db_utils import table_for
from omop_emb.config import MetricType
from omop_emb.interface import EmbeddingReaderInterface

from .conftest import CONCEPT_EMBEDDINGS, CONCEPT_RECORDS, MODEL_NAME, populate, sqlite_cdm_database


@pytest.fixture(
    params=[
        pytest.param("svec_backend", marks=pytest.mark.unit, id="sqlitevec"),
        pytest.param(
            "pg_backend",
            marks=[
                pytest.mark.integration,
                pytest.mark.pgvector,
                pytest.mark.postgresql,
                pytest.mark.db_dialect,
            ],
            id="pgvector",
        ),
        pytest.param(
            "duckdbvss_backend",
            marks=[pytest.mark.duckdbvss, pytest.mark.db_dialect],
            id="duckdbvss",
        ),
    ],
)
def reader(request):
    backend = request.getfixturevalue(request.param)
    populate(backend, CONCEPT_RECORDS[:2], CONCEPT_EMBEDDINGS[:2])

    cdm = sqlite_cdm_database()
    engine, _ = cdm.create_engines()
    try:
        table_for(Concept).create(engine)
        with engine.begin() as connection:
            connection.execute(
                insert(Concept),
                [
                    {
                        "concept_id": concept_id,
                        "concept_name": f"Concept {concept_id}",
                        "domain_id": "Drug" if concept_id in (2, 4) else "Condition",
                        "vocabulary_id": "SNOMED",
                        "concept_class_id": "Clinical Finding",
                        "standard_concept": "S",
                        "concept_code": str(concept_id),
                        "valid_start_date": date(2020, 1, 1),
                        "valid_end_date": date(2099, 12, 31),
                    }
                    for concept_id in (1, 2, 3, 4)
                ],
            )
        yield EmbeddingReaderInterface(
            MODEL_NAME,
            backend,
            MetricType.COSINE,
            cdm_session_factory=cdm_sessionmaker(cdm, primary=engine, vocab=engine),
        )
    finally:
        engine.dispose()


def test_filtered_lookup_and_count_avoid_loading_all_stored_ids(reader, monkeypatch):
    monkeypatch.setattr(
        reader.backend,
        "_get_all_stored_concept_ids_impl",
        Mock(side_effect=AssertionError("filtered lookup loaded all stored IDs")),
    )
    concept_filter = ConceptFilter(concept_ids=(1, 2, 3, 999), domains=("Condition",))

    missing = reader.get_concepts_without_embedding(concept_filter=concept_filter)

    assert set(missing) == {3}
    assert missing[3].concept_name == "Concept 3"
    assert reader.count_concepts_without_embedding(concept_filter=concept_filter) == 1


@pytest.mark.parametrize(
    ("concept_filter", "missing_ids"),
    [
        pytest.param(None, {3, 4}, id="none"),
        pytest.param(ConceptFilter(), {3, 4}, id="empty"),
        pytest.param(ConceptFilter(domains=("Condition",)), {3}, id="domain-only"),
    ],
)
def test_lookup_and_count_without_explicit_ids_load_all_stored_ids(
    reader, monkeypatch, concept_filter, missing_ids
):
    load_all = Mock(wraps=reader.backend._get_all_stored_concept_ids_impl)
    monkeypatch.setattr(reader.backend, "_get_all_stored_concept_ids_impl", load_all)

    assert set(reader.get_concepts_without_embedding(concept_filter=concept_filter)) == missing_ids
    load_all.assert_called_once()
    assert reader.count_concepts_without_embedding(concept_filter=concept_filter) == len(
        missing_ids
    )
    assert load_all.call_count == 2


def test_no_cdm_matches_skips_stored_id_lookup(reader, monkeypatch):
    monkeypatch.setattr(
        reader.backend,
        "get_stored_concept_ids",
        Mock(side_effect=AssertionError("no CDM matches, but the store was queried")),
    )

    assert (
        reader.get_concepts_without_embedding(concept_filter=ConceptFilter(concept_ids=(999,)))
        == {}
    )
