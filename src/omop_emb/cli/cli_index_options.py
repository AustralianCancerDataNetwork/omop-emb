"""Index options shared by the commands that build an index."""

from __future__ import annotations

import typer

from omop_emb.backends.index_config import IndexConfig, index_config_from_index_type
from omop_emb.config import IndexType, MetricType, VectorPrecision


def build_index_config(
    index_type: IndexType,
    *,
    metric_type: MetricType,
    num_neighbors: int | None,
    ef_search: int | None,
    ef_construction: int | None,
    index_precision: VectorPrecision | None,
    partial_domain_indexes: bool,
    partial_valid_only: bool,
) -> IndexConfig:
    """Build the index configuration the command line options describe.

    Raises
    ------
    typer.BadParameter
        If an option value is invalid or a partial index option is given for an index
        type other than HNSW.
    """
    if index_type != IndexType.HNSW and (partial_domain_indexes or partial_valid_only):
        raise typer.BadParameter("Partial index options apply to HNSW indexes only.")
    try:
        return index_config_from_index_type(
            index_type,
            num_neighbors=num_neighbors,
            ef_search=ef_search,
            ef_construction=ef_construction,
            index_precision=index_precision,
            metric_type=metric_type,
            partial_domain_indexes=partial_domain_indexes,
            partial_valid_only=partial_valid_only,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
