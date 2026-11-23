from __future__ import annotations

import hashlib
import re
from typing import ClassVar

from omop_emb.config import MetricType


class IndexNames:
    """Physical names of the indexes omop-emb creates.

    The indexes omop-emb creates on a table are named after that table, so
    ``owns`` tells them apart from every other index on it, such as its
    primary key or an index someone added by hand.

    Attributes
    ----------
    PREFIX : str
        Start of every index name.
    MAX_BYTES : int
        PostgreSQL's identifier limit, which every name fits.
    """

    PREFIX: ClassVar[str] = "idx_"
    MAX_BYTES: ClassVar[int] = 63

    @classmethod
    def hnsw(cls, tablename: str, metric_type: MetricType) -> str:
        """Name of tablename's global HNSW index for metric_type."""
        return f"{cls.PREFIX}{tablename}_{metric_type.value}"

    @classmethod
    def partial(cls, tablename: str, domain: str) -> str:
        """Name of tablename's partial HNSW index for domain.

        The name is ``idx_<hash>_<slug>``. ``hash`` is eight characters of a
        hash of the table name, which keeps the name unique within the schema.
        ``slug`` is the domain in lowercase with runs of other characters
        replaced by ``_``.

        Raises
        ------
        ValueError
            If the slug is empty or the name exceeds ``MAX_BYTES``.
        """
        prefix = cls._partial_prefix(tablename)
        slug = re.sub(r"[^a-z0-9]+", "_", domain.lower()).strip("_")
        if not slug or len(prefix) + len(slug) > cls.MAX_BYTES:
            raise ValueError(f"Cannot name a partial index for domain '{domain}'.")
        return prefix + slug

    @classmethod
    def owns(cls, tablename: str, name: str) -> bool:
        """Whether name is the name of an index omop-emb creates on tablename."""
        return name.startswith((f"{cls.PREFIX}{tablename}_", cls._partial_prefix(tablename)))

    @classmethod
    def _partial_prefix(cls, tablename: str) -> str:
        return f"{cls.PREFIX}{hashlib.sha1(tablename.encode('utf-8')).hexdigest()[:8]}_"
