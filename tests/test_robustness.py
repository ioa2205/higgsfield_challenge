"""Phase 5: malformed, oversized, unicode, cold-store, and auth resilience."""
from __future__ import annotations

import json
import os
import uuid
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from src import config


def _turn(content: str = "I live in Berlin.") -> dict:
    return {
        "user_id": "robust-" + uuid.uuid4().hex[:8],
        "session_id": "robust-session",
        "messages": [{"role": "user", "content": content}],
    }


def _assert_healthy(client) -> None:
    assert client.get("/health").status_code == 200


def test_malformed_json_is_4xx_and_health_survives(client):
    response = client.post(
        "/turns",
        content=b'{"session_id":',
        headers={"Content-Type": "application/json"},
    )
    assert 400 <= response.status_code < 500
    _assert_healthy(client)


def test_missing_and_null_required_fields_are_4xx_and_health_survives(client):
    assert 400 <= client.post("/turns", json={"session_id": "s"}).status_code < 500
    payload = _turn()
    payload["session_id"] = None
    assert 400 <= client.post("/turns", json=payload).status_code < 500
    assert 400 <= client.post(
        "/recall", json={"query": None, "session_id": "s", "user_id": "u"}
    ).status_code < 500
    _assert_healthy(client)


def test_oversized_payload_is_413_and_health_survives(client):
    payload = _turn("x" * (config.MAX_REQUEST_BODY_BYTES + 1))
    response = client.post("/turns", content=json.dumps(payload))
    assert response.status_code == 413
    _assert_healthy(client)


def test_unicode_oddities_are_handled_without_crashing(client):
    odd = "emoji: \U0001f680 rtl: \u202eabc\u202c grapheme: " + "e\u0301" * 2000
    response = client.post("/turns", json=_turn(odd))
    assert response.status_code == 201
    _assert_healthy(client)


def test_nul_is_rejected_as_4xx_and_health_survives(client):
    response = client.post("/turns", json=_turn("before\x00after"))
    assert 400 <= response.status_code < 500
    _assert_healthy(client)


def test_cold_store_recall_is_empty_200(client):
    response = client.post(
        "/recall",
        json={
            "query": "What is the user's favorite editor?",
            "session_id": "cold-session",
            "user_id": "cold-" + uuid.uuid4().hex[:8],
        },
    )
    assert response.status_code == 200
    assert response.json() == {"context": "", "citations": []}


def test_missing_and_wrong_auth_are_contained(client):
    previous = os.environ.get("MEMORY_AUTH_TOKEN")
    os.environ["MEMORY_AUTH_TOKEN"] = "robust-secret"
    try:
        assert client.post("/turns", json=_turn()).status_code == 401
        assert client.post(
            "/turns",
            json=_turn(),
            headers={"Authorization": "Bearer wrong"},
        ).status_code == 403
        _assert_healthy(client)
    finally:
        if previous is None:
            os.environ.pop("MEMORY_AUTH_TOKEN", None)
        else:
            os.environ["MEMORY_AUTH_TOKEN"] = previous


def test_unexpected_error_returns_500_and_health_survives(client):
    """The global handler contains unexpected endpoint failures."""
    with patch(
        "src.api.routes.queries.list_memories",
        new=AsyncMock(side_effect=RuntimeError("synthetic failure")),
    ):
        no_raise = TestClient(client.app, raise_server_exceptions=False)
        response = no_raise.get("/users/robust-handler/memories")
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    _assert_healthy(client)
