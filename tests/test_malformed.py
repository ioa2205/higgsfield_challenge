"""Malformed input yields 4xx, never a 5xx/crash; /health stays healthy."""
from __future__ import annotations

import uuid


def test_bad_json_is_4xx(client):
    r = client.post(
        "/turns",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert 400 <= r.status_code < 500


def test_missing_required_fields_is_4xx(client):
    # Missing messages.
    r = client.post("/turns", json={"user_id": "u", "session_id": "s"})
    assert 400 <= r.status_code < 500

    # Empty messages list (violates min_length).
    r = client.post(
        "/turns", json={"user_id": "u", "session_id": "s", "messages": []}
    )
    assert 400 <= r.status_code < 500

    # Wrong type for messages (with unicode payload) -> validation error.
    r = client.post(
        "/turns",
        json={"user_id": "u", "session_id": "s", "messages": "💣🔥"},
    )
    assert 400 <= r.status_code < 500


def test_valid_unicode_is_handled(client):
    """Valid unicode (emoji, non-latin) must be stored without crashing."""
    user = "uni-" + uuid.uuid4().hex[:8]
    r = client.post(
        "/turns",
        json={
            "user_id": user,
            "session_id": "s",
            "messages": [{"role": "user", "content": "Ich wohne in München 🏙️ 北京"}],
        },
    )
    assert r.status_code == 201


def test_health_still_ok_after_malformed(client):
    client.post("/turns", content=b"garbage")
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
