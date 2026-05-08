"""Tests for optional dependency import guards.

Verifies that missing optional packages produce clear, actionable error
messages rather than bare ImportErrors.
"""
from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest


def _evict_pgvector_modules() -> dict:
    """Remove pgvector and pgvector-backend modules from sys.modules.

    Returns the evicted entries so the caller can restore them in a
    finally block.
    """
    keys = [
        k for k in sys.modules
        if k.startswith("pgvector") or "backends.pgvector" in k
    ]
    return {k: sys.modules.pop(k) for k in keys}


def test_pg_backend_missing_pgvector_install_hint():
    """Importing pg_backend without pgvector installed surfaces the install hint."""
    saved = _evict_pgvector_modules()
    try:
        with patch.dict(sys.modules, {"pgvector": None, "pgvector.sqlalchemy": None}):
            with pytest.raises(ImportError, match=r"pip install omop-emb\[pgvector\]"):
                importlib.import_module("omop_emb.backends.pgvector.pg_backend")
    finally:
        sys.modules.update(saved)


def test_pgvector_package_init_missing_pgvector_install_hint():
    """Importing the pgvector package without pgvector installed surfaces the install hint."""
    saved = _evict_pgvector_modules()
    try:
        with patch.dict(sys.modules, {"pgvector": None, "pgvector.sqlalchemy": None}):
            with pytest.raises(ImportError, match=r"pip install omop-emb\[pgvector\]"):
                importlib.import_module("omop_emb.backends.pgvector")
    finally:
        sys.modules.update(saved)


def test_omop_emb_importable_without_pgvector():
    """The top-level omop_emb package is importable even without pgvector."""
    with patch.dict(sys.modules, {"pgvector": None, "pgvector.sqlalchemy": None}):
        # Should not raise — pgvector is an optional, lazily-imported dep
        import omop_emb  # noqa: F401


def test_faiss_package_missing_faiss_install_hint():
    """Importing omop_emb.storage.faiss without faiss installed surfaces the install hint."""
    faiss_keys = [k for k in sys.modules if k.startswith("faiss") or "storage.faiss" in k]
    saved = {k: sys.modules.pop(k) for k in faiss_keys}
    try:
        with patch.dict(sys.modules, {"faiss": None}):
            with pytest.raises(ImportError, match=r"pip install omop-emb\[faiss\]"):
                importlib.import_module("omop_emb.storage.faiss")
    finally:
        sys.modules.update(saved)
