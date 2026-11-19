"""The query scenarios of a benchmark: a domain scope and a set of concept filters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omop_emb.utils.embedding_utils import EmbeddingConceptFilter

PROFILES: dict[str, tuple[bool, bool]] = {
    "any": (False, False),
    "active": (True, False),
    "standard": (False, True),
    "active+standard": (True, True),
}
"""Filter profiles as ``(require_active, require_standard)``."""

NONE_SCOPE = "none"


@dataclass(frozen=True)
class Query:
    """A query text with the domain it was sampled from."""

    id: int
    text: str
    domain: str


@dataclass(frozen=True)
class Scenario:
    """A domain scope under one filter profile.

    Attributes
    ----------
    domains : tuple[str, ...]
        Domains the search names. Empty for no domain.
    profile : str
        Key of ``PROFILES``.
    """

    domains: tuple[str, ...]
    profile: str

    @property
    def scope(self) -> str:
        """``none``, a domain, or domains joined with ``+``."""
        return "+".join(self.domains) if self.domains else NONE_SCOPE

    @property
    def name(self) -> str:
        return f"{self.scope}|{self.profile}"

    def concept_filter(
        self, domains: tuple[str, ...] | None = None
    ) -> EmbeddingConceptFilter | None:
        """The filter this scenario sends, optionally naming other domains."""
        active, standard = PROFILES[self.profile]
        domains = self.domains if domains is None else domains
        if not domains and not active and not standard:
            return None
        return EmbeddingConceptFilter(
            domains=domains or None, require_active=active, require_standard=standard
        )


def parse_pair(text: str) -> tuple[str, str]:
    """Parse ``A+B`` into two domains."""
    first, _, second = text.partition("+")
    if not first or not second or "+" in second:
        raise ValueError(f"A pair is two domains joined by '+', got '{text}'.")
    return first, second


def build_scenarios(
    domains: Sequence[str], pairs: Sequence[tuple[str, str]], profiles: Sequence[str]
) -> list[Scenario]:
    """Scenarios for no domain, each domain and each pair, under every profile."""
    unknown = [profile for profile in profiles if profile not in PROFILES]
    if unknown:
        raise ValueError(f"Unknown profiles {unknown}. Choose from {list(PROFILES)}.")
    scopes: list[tuple[str, ...]] = [(), *((d,) for d in domains), *pairs]
    return [Scenario(scope, profile) for scope in scopes for profile in profiles]


def scenario_queries(
    scenario: Scenario, queries: Sequence[Query], none_per_domain: int
) -> list[Query]:
    """The queries a scenario runs.

    A domain or pair runs the queries sampled from those domains. No domain
    runs the first ``none_per_domain`` queries of every domain, because exact
    search without a domain is the slowest part of the benchmark.
    """
    if scenario.domains:
        return [q for q in queries if q.domain in scenario.domains]
    taken: dict[str, int] = {}
    chosen = []
    for query in queries:
        if taken.get(query.domain, 0) < none_per_domain:
            taken[query.domain] = taken.get(query.domain, 0) + 1
            chosen.append(query)
    return chosen
