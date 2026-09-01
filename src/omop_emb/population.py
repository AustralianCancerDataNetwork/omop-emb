"""Identity-aware, read-only embedding population planning."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import chain
from typing import Literal

from sqlalchemy import Engine, Row, select
from sqlalchemy.orm import Session

from omop_alchemy.cdm.model.vocabulary import Concept
from omop_alchemy.cdm.query import ConceptFilter

from omop_emb.backends.read_only import ReadOnlyEmbeddingStore, StoredEmbedding
from omop_emb.utils.cdm import streamed


@dataclass(frozen=True)
class PopulationScope:
    """The exact CDM predicate used by a population plan."""

    standard_only: bool = False
    valid_only: bool = False
    vocabularies: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()

    def concept_filter(self) -> ConceptFilter:
        """Return the canonical omop-alchemy filter for this scope."""

        return ConceptFilter(
            domains=self.domains or None,
            vocabularies=self.vocabularies or None,
            require_standard=self.standard_only,
            require_active=self.valid_only,
        )


@dataclass(frozen=True)
class VocabularyPopulationPlan:
    """Identity comparison for one vocabulary under a population scope."""

    vocabulary: str
    eligible_ids: frozenset[int]
    compatible_ids: frozenset[int]
    missing_ids: frozenset[int]
    stale_ids: frozenset[int]
    metadata_changed_ids: frozenset[int]
    changed_source_text_ids: frozenset[int] = frozenset()

    @property
    def pending_ids(self) -> frozenset[int]:
        return (
            self.missing_ids | self.metadata_changed_ids | self.changed_source_text_ids
        )


@dataclass(frozen=True)
class EmbeddingPopulationPlan:
    """A repeatable identity-aware plan shared by coverage and execution."""

    model_name: str
    scope: PopulationScope
    rows: tuple[VocabularyPopulationPlan, ...]
    store_initialized: bool
    source_text_provenance: Literal["not_recorded"] = "not_recorded"

    @property
    def eligible_ids(self) -> frozenset[int]:
        return _union(row.eligible_ids for row in self.rows)

    @property
    def compatible_ids(self) -> frozenset[int]:
        return _union(row.compatible_ids for row in self.rows)

    @property
    def missing_ids(self) -> frozenset[int]:
        return _union(row.missing_ids for row in self.rows)

    @property
    def stale_ids(self) -> frozenset[int]:
        return _union(row.stale_ids for row in self.rows)

    @property
    def metadata_changed_ids(self) -> frozenset[int]:
        return _union(row.metadata_changed_ids for row in self.rows)

    @property
    def changed_source_text_ids(self) -> frozenset[int]:
        return _union(row.changed_source_text_ids for row in self.rows)

    @property
    def pending_ids(self) -> frozenset[int]:
        return (
            self.missing_ids | self.metadata_changed_ids | self.changed_source_text_ids
        )


@dataclass
class _VocabularyAccumulator:
    eligible: set[int] = field(default_factory=set)
    compatible: set[int] = field(default_factory=set)
    missing: set[int] = field(default_factory=set)
    stale: set[int] = field(default_factory=set)
    metadata_changed: set[int] = field(default_factory=set)

    def freeze(self, vocabulary: str) -> VocabularyPopulationPlan:
        return VocabularyPopulationPlan(
            vocabulary=vocabulary,
            eligible_ids=frozenset(self.eligible),
            compatible_ids=frozenset(self.compatible),
            missing_ids=frozenset(self.missing),
            stale_ids=frozenset(self.stale),
            metadata_changed_ids=frozenset(self.metadata_changed),
        )


def plan_population(
    cdm_engine: Engine,
    store: ReadOnlyEmbeddingStore,
    *,
    model_name: str,
    scope: PopulationScope = PopulationScope(),
    batch_size: int = 10_000,
) -> EmbeddingPopulationPlan:
    """Compare current CDM concept identities with stored vector identities.

    CDM and vector rows are streamed. Only the stored identity map and the ID
    sets returned in the final plan are retained. Stored source text is not
    retained by the current embedding schema, so source-text changes are
    explicitly reported as unavailable.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    stored = {
        item.concept_id: item
        for item in _iter_stored_embeddings(store, model_name, batch_size=batch_size)
    }
    accumulators: dict[str, _VocabularyAccumulator] = {}

    for row in _iter_current_concepts(cdm_engine, scope, batch_size=batch_size):
        concept_id = int(row.concept_id)
        vocabulary = str(row.vocabulary_id)
        accumulator = accumulators.setdefault(vocabulary, _VocabularyAccumulator())
        accumulator.eligible.add(concept_id)
        stored_item = stored.pop(concept_id, None)
        if stored_item is None:
            accumulator.missing.add(concept_id)
        elif _metadata_matches(row, stored_item):
            accumulator.compatible.add(concept_id)
        else:
            accumulator.metadata_changed.add(concept_id)

    for concept_id, item in stored.items():
        if not _stored_matches_scope(item, scope):
            continue
        accumulator = accumulators.setdefault(
            item.vocabulary_id,
            _VocabularyAccumulator(),
        )
        accumulator.stale.add(concept_id)

    return EmbeddingPopulationPlan(
        model_name=model_name,
        scope=scope,
        rows=tuple(
            accumulators[vocabulary].freeze(vocabulary)
            for vocabulary in sorted(accumulators)
        ),
        store_initialized=store.initialized,
    )


def _iter_current_concepts(
    cdm_engine: Engine,
    scope: PopulationScope,
    *,
    batch_size: int,
) -> Iterator[Row]:
    statement = (
        scope.concept_filter()
        .apply(
            select(
                Concept.concept_id,
                Concept.domain_id,
                Concept.vocabulary_id,
                Concept.is_standard_expr().label("is_standard"),
                Concept.is_valid_expr().label("is_valid"),
            )
        )
    )
    with Session(cdm_engine) as session:
        yield from session.execute(streamed(statement, batch_size))


def _iter_stored_embeddings(
    store: ReadOnlyEmbeddingStore,
    model_name: str,
    *,
    batch_size: int,
) -> Iterator[StoredEmbedding]:
    iterator = getattr(store, "iter_stored_embeddings", None)
    if iterator is not None:
        yield from iterator(model_name, batch_size=batch_size)
        return
    yield from store.stored_embeddings(model_name)


def _stored_matches_scope(item: StoredEmbedding, scope: PopulationScope) -> bool:
    return (
        (not scope.vocabularies or item.vocabulary_id in scope.vocabularies)
        and (not scope.domains or item.domain_id in scope.domains)
        and (not scope.standard_only or item.is_standard)
        and (not scope.valid_only or item.is_valid)
    )


def _metadata_matches(row: Row, stored: StoredEmbedding) -> bool:
    return (
        str(row.domain_id) == stored.domain_id
        and str(row.vocabulary_id) == stored.vocabulary_id
        and bool(row.is_standard) == stored.is_standard
        and bool(row.is_valid) == stored.is_valid
    )


def _union(groups: Iterable[frozenset[int]]) -> frozenset[int]:
    return frozenset(chain.from_iterable(groups))


__all__ = [
    "EmbeddingPopulationPlan",
    "PopulationScope",
    "VocabularyPopulationPlan",
    "plan_population",
]
