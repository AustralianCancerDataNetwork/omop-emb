"""Tests for CDMConceptFilter.apply() — the CDM-only WHERE/LIMIT builder."""

import pytest
from sqlalchemy import select

from omop_alchemy.cdm.model.vocabulary import Concept
from omop_emb.utils.embedding_utils import CDMConceptFilter


@pytest.mark.unit
class TestCDMConceptFilterApply:
    def test_empty_filter_adds_no_clauses(self):
        query = select(Concept.concept_id)
        result = CDMConceptFilter().apply(query, Concept)

        assert str(result) == str(query)

    def test_concept_ids_adds_in_clause(self):
        query = select(Concept.concept_id)
        result = CDMConceptFilter(concept_ids=(1, 2, 3)).apply(query, Concept)

        compiled = str(result)
        assert "WHERE" in compiled
        assert "concept_id IN" in compiled

    def test_limit_is_applied(self):
        query = select(Concept.concept_id)
        result = CDMConceptFilter(limit=5).apply(query, Concept)

        assert "LIMIT" in str(result)

    def test_negative_limit_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            CDMConceptFilter(limit=0)

    def test_is_empty(self):
        assert CDMConceptFilter().is_empty()
        assert not CDMConceptFilter(limit=5).is_empty()
        assert not CDMConceptFilter(domains=("Drug",)).is_empty()
