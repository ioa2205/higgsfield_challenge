"""Shared pytest fixtures.

Tests drive the FastAPI app in-process (Starlette TestClient) against the
dockerized Postgres (`db` service on localhost:5432). They use the deterministic
"fake" embedder, so no model/network is required -- only a reachable DB. Start
it first with:  docker compose up -d db
"""
from __future__ import annotations

import contextlib
import os
import uuid

# Configure the environment BEFORE importing the app (config reads env at import).
os.environ.setdefault("EMBED_BACKEND", "fake")
os.environ.setdefault("PGHOST", "localhost")
# Host-side tests reach the dockerized `db` on 5433 (see docker-compose.yml).
os.environ.setdefault("PGPORT", "5433")
os.environ.setdefault("PGUSER", "memory")
os.environ.setdefault("PGPASSWORD", "memory")
os.environ.setdefault("PGDATABASE", "memory")
os.environ.pop("MEMORY_AUTH_TOKEN", None)
# Tests run fully offline and deterministically on the rule extraction path:
# never call out to a real LLM even if a provider key happens to be in the
# host environment. The LLM branch is exercised separately by monkeypatching
# the extractor seam in test_extraction.py.
os.environ["LLM_PROVIDER"] = "none"

import pytest
from fastapi.testclient import TestClient


@contextlib.contextmanager
def build_client():
    """Fresh app instance (fresh pool) wrapped in a TestClient with lifespan."""
    import importlib

    from src import config, main

    importlib.reload(config)
    importlib.reload(main)
    with TestClient(main.app) as client:
        yield client


# Expose the factory to tests that need their own client (e.g. persistence).
build_client_cm = build_client


@pytest.fixture
def client():
    with build_client() as c:
        yield c


@pytest.fixture
def user_id():
    return "u-" + uuid.uuid4().hex[:10]


@pytest.fixture
def session_id():
    return "s-" + uuid.uuid4().hex[:10]
