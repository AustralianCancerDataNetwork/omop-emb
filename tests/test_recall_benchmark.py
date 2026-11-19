"""Scoring, plan parsing, scenarios and the read-only guard of the recall benchmark."""

from __future__ import annotations

from typing import ClassVar

import pytest
import sqlalchemy as sa

from benchmarks.recall.plans import label_index, scanned_indexes, sql_literal
from benchmarks.recall.report import weighted_recall
from benchmarks.recall.scenarios import (
    PROFILES,
    Query,
    Scenario,
    build_scenarios,
    parse_pair,
    scenario_queries,
)
from benchmarks.recall.scoring import (
    estimate_index_bytes,
    merge_top_k,
    parse_setting,
    recall_at_k,
    weighted_mean,
)
from benchmarks.recall.store import guard_read_only
from omop_emb.backends.index_naming import IndexNames
from omop_emb.config import MetricType

pytestmark = pytest.mark.unit


class TestRecallByDistance:
    TRUTH: ClassVar[list[tuple[int, float]]] = [(1, 0.10), (2, 0.20), (3, 0.30), (4, 0.40)]

    def test_all_found(self):
        assert recall_at_k(self.TRUTH, self.TRUTH, 4) == 1.0

    def test_counts_the_share_found(self):
        returned = [(1, 0.10), (9, 0.90), (3, 0.30), (8, 0.80)]
        assert recall_at_k(returned, self.TRUTH, 4) == 0.5

    def test_a_tied_concept_counts_as_found(self):
        returned = [(1, 0.10), (2, 0.20), (3, 0.30), (7, 0.40 + 1e-9)]
        assert recall_at_k(returned, self.TRUTH, 4) == 1.0

    def test_only_the_first_k_returned_count(self):
        returned = [(9, 0.90), (1, 0.10), (2, 0.20)]
        assert recall_at_k(returned, self.TRUTH, 1) == 0.0

    def test_a_filter_with_fewer_than_k_matches_is_scored_on_what_exists(self):
        assert recall_at_k([(1, 0.10), (2, 0.20)], self.TRUTH[:2], 10) == 1.0

    def test_nothing_to_find_has_no_recall(self):
        assert recall_at_k([], [], 10) is None

    def test_a_missing_result_lowers_recall(self):
        assert recall_at_k([], self.TRUTH, 4) == 0.0


def test_merging_keeps_the_closest_distinct_concepts():
    merged = merge_top_k([[(1, 0.1), (3, 0.3)], [(2, 0.2), (3, 0.25)]], 3)
    assert merged == [(1, 0.1), (2, 0.2), (3, 0.25)]


def test_weighted_mean_ignores_missing_values_and_renormalises():
    assert weighted_mean({"a": 1.0, "b": None}, {"a": 0.2, "b": 0.8}) == 1.0
    assert weighted_mean({"a": 1.0, "b": 0.0}, {"a": 0.75, "b": 0.25}) == 0.75
    assert weighted_mean({"a": None}, {"a": 1.0}) is None


def test_index_estimate_matches_the_measured_size_of_a_full_precision_index():
    measured = 3.30 * 2**30
    assert estimate_index_bytes(432_583, 1024, 4, 32) == pytest.approx(measured, rel=0.02)


@pytest.mark.parametrize(
    ("text", "expected"), [("200", (200, None)), ("200:100000", (200, 100000))]
)
def test_setting_parses_ef_search_and_scan_cap(text, expected):
    assert parse_setting(text) == expected


class TestPlans:
    PLAN: ClassVar[str] = (
        "Limit  (cost=1.00..2.00 rows=10 width=12)\n"
        "  ->  Index Scan using idx_ab12cd34_drug on emb_x  (cost=1.00..2.00 rows=5 width=12)\n"
        "  ->  Seq Scan on other (cost=0.00..1.00 rows=1 width=4)"
    )

    def test_names_the_scanned_indexes(self):
        assert scanned_indexes(self.PLAN) == ["idx_ab12cd34_drug", "seq scan"]

    def test_labels_global_partial_and_other_indexes(self):
        table, metric = "emb_x", MetricType.COSINE
        domains = ["Drug", "Condition"]

        def label(name: str) -> str:
            return label_index(name, table, metric, domains)

        assert label(IndexNames.hnsw(table, metric)) == "global"
        assert label(IndexNames.partial(table, "Drug")) == "partial:Drug"
        assert label("seq scan") == "seq scan"
        assert label("emb_x_pkey") == "other:emb_x_pkey"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (10, "10"),
            (True, "true"),
            ("[0.5,0.5]", "'[0.5,0.5]'"),
            ("O'Neil", "'O''Neil'"),
            (["Drug", "Condition"], '\'{"Drug","Condition"}\''),
        ],
    )
    def test_renders_a_bound_parameter_as_a_literal(self, value, expected):
        assert sql_literal(value) == expected


class TestScenarios:
    def test_every_scope_runs_under_every_profile(self):
        scenarios = build_scenarios(["Drug", "Condition"], [("Drug", "Condition")], list(PROFILES))
        assert len(scenarios) == (1 + 2 + 1) * len(PROFILES)

    def test_unknown_profile_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown profiles"):
            build_scenarios(["Drug"], [], ["nope"])

    @pytest.mark.parametrize("text", ["Drug", "Drug+", "A+B+C"])
    def test_a_pair_is_two_domains(self, text):
        with pytest.raises(ValueError, match="two domains"):
            parse_pair(text)

    def test_filters_follow_the_scope_and_profile(self):
        assert Scenario((), "any").concept_filter() is None
        one = Scenario(("Drug",), "active+standard").concept_filter()
        assert one is not None and one.domains == ("Drug",)
        assert one.require_active and one.require_standard
        leg = Scenario(("Drug", "Condition"), "any").concept_filter(("Drug",))
        assert leg is not None and leg.domains == ("Drug",)

    def test_a_search_without_a_domain_takes_a_few_queries_per_domain(self):
        queries = [Query(i, f"q{i}", "Drug" if i < 4 else "Condition") for i in range(8)]
        chosen = scenario_queries(Scenario((), "any"), queries, 2)
        assert [q.id for q in chosen] == [0, 1, 4, 5]
        assert len(scenario_queries(Scenario(("Drug",), "any"), queries, 2)) == 4


def _cell(scope: str, recall: float, arm: str = "combined") -> dict:
    return {
        "setting": "s",
        "profile": "any",
        "scope": scope,
        "arm": arm,
        "recall@10": {"mean": recall},
    }


def test_workload_recall_weights_single_domain_and_no_domain_searches():
    cells = [
        _cell("Drug", 0.8),
        _cell("Condition", 1.0),
        _cell("none", 0.5),
        _cell("Drug+Condition", 0.1),
    ]

    value = weighted_recall(cells, "s", "any", {"single": 0.8, "none": 0.2})

    assert value == pytest.approx(0.8 * 0.9 + 0.2 * 0.5)


def test_read_only_guard_allows_reads_and_blocks_writes():
    engine = sa.create_engine("sqlite://")
    guard_read_only(engine)
    with engine.connect() as conn:
        assert conn.execute(sa.text("SELECT 1")).scalar_one() == 1
        with pytest.raises(RuntimeError, match="Blocked non-read-only SQL"):
            conn.execute(sa.text("CREATE TABLE t (x INTEGER)"))
