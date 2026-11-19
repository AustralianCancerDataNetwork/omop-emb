"""The files a benchmark keeps in its work directory."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.recall.scenarios import Query, Scenario, build_scenarios, parse_pair


def write_json(path: Path, content: Any) -> None:
    """Write content as JSON, replacing path in one step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@dataclass
class Workspace:
    """The queries, scenarios, ground truth and results of one benchmark.

    Attributes
    ----------
    root : Path
        Work directory.
    """

    root: Path

    @property
    def plan_path(self) -> Path:
        return self.root / "plan.json"

    @property
    def vectors_path(self) -> Path:
        return self.root / "vectors.npy"

    @property
    def truth_path(self) -> Path:
        return self.root / "truth.json"

    @property
    def builds_path(self) -> Path:
        return self.root / "builds.json"

    def run_path(self, label: str) -> Path:
        return self.root / "runs" / f"{label}.json"

    def save_plan(
        self,
        *,
        queries: list[Query],
        vectors: np.ndarray,
        seed: str,
        per_domain: int,
        domains: list[str],
        pairs: list[str],
        profiles: list[str],
        none_per_domain: int,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        np.save(self.vectors_path, vectors)
        write_json(
            self.plan_path,
            {
                "seed": seed,
                "per_domain": per_domain,
                "domains": domains,
                "pairs": pairs,
                "profiles": profiles,
                "none_per_domain": none_per_domain,
                "queries": [{"id": q.id, "text": q.text, "domain": q.domain} for q in queries],
            },
        )

    def load_plan(self) -> dict[str, Any]:
        if not self.plan_path.exists():
            raise RuntimeError(f"No plan in {self.root}. Run the 'queries' command first.")
        return json.loads(self.plan_path.read_text(encoding="utf-8"))

    def queries(self) -> list[Query]:
        return [Query(q["id"], q["text"], q["domain"]) for q in self.load_plan()["queries"]]

    def vectors(self) -> np.ndarray:
        return np.load(self.vectors_path)

    def scenarios(self) -> list[Scenario]:
        plan = self.load_plan()
        return build_scenarios(
            plan["domains"], [parse_pair(p) for p in plan["pairs"]], plan["profiles"]
        )

    def read_builds(self) -> list[dict[str, Any]]:
        if not self.builds_path.exists():
            return []
        return json.loads(self.builds_path.read_text(encoding="utf-8"))

    def append_build(self, entry: dict[str, Any]) -> None:
        write_json(self.builds_path, [*self.read_builds(), entry])
