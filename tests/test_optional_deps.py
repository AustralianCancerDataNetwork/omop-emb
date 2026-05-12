"""Tests for optional dependency import guards.

Verifies that:
- optional packages can be omitted without breaking top-level and package imports
- clear, actionable error messages are raised when optional symbols are
  accessed without the backing package installed

Each "importable without X" test exercises the lazy-load shield.
Each "missing install hint" test exercises that the error is still surfaced
at point of use rather than being silently swallowed.
"""
from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evict_pgvector_modules() -> dict:
    """Remove pgvector and related modules from sys.modules.

    Also evicts omop_emb.backends so the __getattr__ lazy cache is cleared
    and attribute-access tests get a fresh module.
    """
    keys = [
        k for k in sys.modules
        if k.startswith("pgvector")
        or "backends.pgvector" in k
        or k == "omop_emb.backends"
    ]
    return {k: sys.modules.pop(k) for k in keys}


def _evict_faiss_modules() -> dict:
    """Remove faiss and related modules from sys.modules.

    Also evicts omop_emb.storage.faiss so the __getattr__ lazy cache is
    cleared and attribute-access tests get a fresh module.
    """
    keys = [
        k for k in sys.modules
        if k.startswith("faiss") or "storage.faiss" in k
    ]
    return {k: sys.modules.pop(k) for k in keys}


# ---------------------------------------------------------------------------
# pgvector — direct module imports still raise immediately
# ---------------------------------------------------------------------------

def test_pg_backend_missing_pgvector_install_hint():
    """Directly importing pg_backend without pgvector raises with the install hint."""
    saved = _evict_pgvector_modules()
    try:
        with patch.dict(sys.modules, {"pgvector": None, "pgvector.sqlalchemy": None}):
            with pytest.raises(ImportError, match=r"pip install omop-emb\[pgvector\]"):
                importlib.import_module("omop_emb.backends.pgvector.pg_backend")
    finally:
        sys.modules.update(saved)


def test_pgvector_subpackage_missing_pgvector_install_hint():
    """Directly importing omop_emb.backends.pgvector without pgvector raises with the install hint."""
    saved = _evict_pgvector_modules()
    try:
        with patch.dict(sys.modules, {"pgvector": None, "pgvector.sqlalchemy": None}):
            with pytest.raises(ImportError, match=r"pip install omop-emb\[pgvector\]"):
                importlib.import_module("omop_emb.backends.pgvector")
    finally:
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# pgvector — lazy-shielded imports do NOT raise at package level
# ---------------------------------------------------------------------------

def test_omop_emb_importable_without_pgvector():
    """Top-level import of omop_emb succeeds even when pgvector is not installed."""
    with patch.dict(sys.modules, {"pgvector": None, "pgvector.sqlalchemy": None}):
        import omop_emb  # noqa: F401


def test_backends_importable_without_pgvector():
    """omop_emb.backends is importable even when pgvector is not installed."""
    saved = _evict_pgvector_modules()
    try:
        with patch.dict(sys.modules, {"pgvector": None, "pgvector.sqlalchemy": None}):
            importlib.import_module("omop_emb.backends")  # must not raise
    finally:
        sys.modules.update(saved)


def test_backends_pgvector_attr_missing_install_hint():
    """Accessing backends.PGVectorEmbeddingBackend without pgvector raises with the install hint."""
    saved = _evict_pgvector_modules()
    try:
        with patch.dict(sys.modules, {"pgvector": None, "pgvector.sqlalchemy": None}):
            backends = importlib.import_module("omop_emb.backends")
            with pytest.raises(ImportError, match=r"pip install omop-emb\[pgvector\]"):
                _ = backends.PGVectorEmbeddingBackend
    finally:
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# FAISS — direct module import still raises immediately
# ---------------------------------------------------------------------------

def test_faiss_cache_module_missing_install_hint():
    """Directly importing faiss_cache without faiss raises with the install hint."""
    saved = _evict_faiss_modules()
    try:
        with patch.dict(sys.modules, {"faiss": None}):
            with pytest.raises(ImportError, match=r"pip install omop-emb\[faiss"):
                importlib.import_module("omop_emb.storage.faiss.faiss_cache")
    finally:
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# FAISS — lazy-shielded imports do NOT raise at package level
# ---------------------------------------------------------------------------

def test_faiss_storage_importable_without_faiss():
    """omop_emb.storage.faiss is importable even when faiss is not installed."""
    saved = _evict_faiss_modules()
    try:
        with patch.dict(sys.modules, {"faiss": None}):
            importlib.import_module("omop_emb.storage.faiss")  # must not raise
    finally:
        sys.modules.update(saved)


def test_faiss_attr_access_missing_install_hint():
    """Accessing FAISSCache without faiss raises with the install hint."""
    saved = _evict_faiss_modules()
    try:
        with patch.dict(sys.modules, {"faiss": None}):
            mod = importlib.import_module("omop_emb.storage.faiss")
            with pytest.raises(ImportError, match=r"pip install omop-emb\[faiss"):
                _ = mod.FAISSCache
    finally:
        sys.modules.update(saved)
