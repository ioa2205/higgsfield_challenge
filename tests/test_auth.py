"""Optional bearer auth: enforced iff MEMORY_AUTH_TOKEN is set; /health open."""
from __future__ import annotations

import os
import uuid

import pytest


@pytest.fixture
def auth_token():
    """Set MEMORY_AUTH_TOKEN for the duration of a test, then restore."""
    prev = os.environ.get("MEMORY_AUTH_TOKEN")
    token = "secret-" + uuid.uuid4().hex[:8]
    os.environ["MEMORY_AUTH_TOKEN"] = token
    try:
        yield token
    finally:
        if prev is None:
            os.environ.pop("MEMORY_AUTH_TOKEN", None)
        else:
            os.environ["MEMORY_AUTH_TOKEN"] = prev


def _turn_payload():
    return {
        "user_id": "auth-" + uuid.uuid4().hex[:8],
        "session_id": "s",
        "messages": [{"role": "user", "content": "I live in Berlin."}],
    }


def test_no_token_endpoints_open(client):
    # MEMORY_AUTH_TOKEN unset (conftest pops it) -> no header needed.
    assert "MEMORY_AUTH_TOKEN" not in os.environ
    r = client.post("/turns", json=_turn_payload())
    assert r.status_code == 201
    assert client.get("/health").status_code == 200


def test_token_set_requires_header(auth_token):
    from conftest import build_client_cm

    with build_client_cm() as client:
        # Missing header -> 401.
        r = client.post("/turns", json=_turn_payload())
        assert r.status_code == 401

        # Wrong token -> 403.
        r = client.post(
            "/turns",
            json=_turn_payload(),
            headers={"Authorization": "Bearer wrong"},
        )
        assert r.status_code == 403

        # Correct token -> normal status.
        r = client.post(
            "/turns",
            json=_turn_payload(),
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert r.status_code == 201

        # /health open regardless of auth.
        assert client.get("/health").status_code == 200
