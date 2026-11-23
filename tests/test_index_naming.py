"""Physical index names."""

from __future__ import annotations

import pytest

from omop_emb.backends.index_naming import IndexNames
from omop_emb.config import MetricType
from omop_emb.model_registry import RegistryManager


class TestHNSW:
    def test_name_follows_table_and_metric(self):
        assert IndexNames.hnsw("my_table", MetricType.L2) == "idx_my_table_l2"
        assert IndexNames.hnsw("my_table", MetricType.COSINE) == "idx_my_table_cosine"

    def test_name_fits_the_identifier_limit_for_a_long_table_name(self):
        storage_id = RegistryManager.storage_name("store_a", "x" * 120)

        for metric in MetricType:
            assert len(IndexNames.hnsw(storage_id, metric).encode("ascii")) <= IndexNames.MAX_BYTES


class TestPartial:
    def test_name_is_readable_and_unique_per_table_and_domain(self):
        name = IndexNames.partial("my_table", "Spec Anatomic Site")

        assert name.startswith(IndexNames.PREFIX)
        assert name.endswith("_spec_anatomic_site")
        assert name == IndexNames.partial("my_table", "Spec Anatomic Site")
        assert name != IndexNames.partial("other_table", "Spec Anatomic Site")
        assert name != IndexNames.partial("my_table", "Drug")

    def test_name_fits_the_identifier_limit_for_a_long_table_name(self):
        storage_id = RegistryManager.storage_name("store_a", "x" * 120)

        name = IndexNames.partial(storage_id, "Spec Anatomic Site")
        assert len(name.encode("ascii")) <= IndexNames.MAX_BYTES

    @pytest.mark.parametrize("domain", ["", "***", "x" * 60])
    def test_rejects_a_domain_that_cannot_be_named(self, domain):
        with pytest.raises(ValueError, match="partial index"):
            IndexNames.partial("my_table", domain)


class TestOwns:
    def test_matches_the_indexes_created_for_the_table(self):
        assert IndexNames.owns("my_table", IndexNames.hnsw("my_table", MetricType.L2))
        assert IndexNames.owns("my_table", IndexNames.partial("my_table", "Drug"))

    @pytest.mark.parametrize(
        "name", ["my_table_pkey", "ix_my_table_concept_id", "idx_domain", "idx_my_tablex_l2"]
    )
    def test_ignores_other_indexes_on_the_table(self, name):
        assert not IndexNames.owns("my_table", name)

    def test_ignores_the_indexes_of_other_tables(self):
        assert not IndexNames.owns("my_table", IndexNames.hnsw("other_table", MetricType.L2))
        assert not IndexNames.owns("my_table", IndexNames.partial("other_table", "Drug"))
