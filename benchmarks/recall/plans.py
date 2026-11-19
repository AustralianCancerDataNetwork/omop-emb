"""Which index the planner uses for a statement, as a one-off plan and as a generic plan."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from sqlalchemy import Engine, text

from omop_emb.backends.index_naming import IndexNames
from omop_emb.config import MetricType

_SCAN = re.compile(r"(Index Only Scan|Index Scan|Bitmap Index Scan|Seq Scan)(?: using (\w+))?")
_PLACEHOLDER = re.compile(r"%\((\w+)\)s")


def scanned_indexes(plan: str) -> list[str]:
    """Names of the indexes a plan scans, with ``seq scan`` for a sequential scan."""
    return [match.group(2) or "seq scan" for match in _SCAN.finditer(plan)]


def label_index(name: str, table: str, metric: MetricType, domains: Iterable[str]) -> str:
    """``global``, ``partial:<domain>``, ``seq scan`` or ``other:<name>`` for an index name."""
    if name == "seq scan":
        return name
    if name == IndexNames.hnsw(table, metric):
        return "global"
    for domain in domains:
        try:
            if name == IndexNames.partial(table, domain):
                return f"partial:{domain}"
        except ValueError:
            continue
    return f"other:{name}"


def sql_literal(value: Any) -> str:
    """SQL literal for a bound parameter: a number, a boolean, a string or a list of strings."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "'{" + ",".join('"' + str(item).replace('"', '\\"') + '"' for item in value) + "}'"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def _explain_custom(engine: Engine, statement: str, parameters: Mapping[str, Any]) -> str:
    with engine.connect() as conn:
        conn.execute(text("SET LOCAL transaction_read_only = on"))
        rows = conn.exec_driver_sql("EXPLAIN " + statement, dict(parameters))
        return "\n".join(str(row[0]) for row in rows)


def _explain_generic(engine: Engine, statement: str, parameters: Mapping[str, Any]) -> str:
    names = list(dict.fromkeys(_PLACEHOLDER.findall(statement)))
    prepared = _PLACEHOLDER.sub(lambda m: f"${names.index(m.group(1)) + 1}", statement)
    call = ", ".join(sql_literal(parameters[name]) for name in names)
    with engine.connect() as conn:
        conn.execute(text("SET LOCAL transaction_read_only = on"))
        conn.execute(text("SET LOCAL plan_cache_mode = force_generic_plan"))
        conn.exec_driver_sql(f"PREPARE benchmark_generic AS {prepared}")
        try:
            rows = conn.exec_driver_sql(f"EXPLAIN EXECUTE benchmark_generic({call})")
            return "\n".join(str(row[0]) for row in rows)
        finally:
            conn.exec_driver_sql("DEALLOCATE benchmark_generic")


def plan_labels(
    engine: Engine,
    statements: Iterable[tuple[str, Mapping[str, Any]]],
    *,
    table: str,
    metric: MetricType,
    domains: Iterable[str],
) -> dict[str, Counter[str]]:
    """Count the index labels of statements under a one-off plan (``custom``) and a generic plan.

    Parameters
    ----------
    engine : Engine
        Engine the statements ran on.
    statements : Iterable[tuple[str, Mapping[str, Any]]]
        Statement texts as sent to the driver, with their parameters.
    table : str
        Storage identifier of the embedding table.
    metric : MetricType
        Metric of the global index.
    domains : Iterable[str]
        Domains whose partial index names are recognised.
    """
    domains = tuple(domains)
    counts: dict[str, Counter[str]] = {"custom": Counter(), "generic": Counter()}
    for statement, parameters in statements:
        for kind, explain in (("custom", _explain_custom), ("generic", _explain_generic)):
            plan = explain(engine, statement, parameters)
            for name in scanned_indexes(plan):
                counts[kind][label_index(name, table, metric, domains)] += 1
    return counts
