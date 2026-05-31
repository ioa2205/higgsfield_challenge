"""User isolation + intended cross-session sharing for the same user."""
from __future__ import annotations

import uuid


def _new_user():
    return "iso-" + uuid.uuid4().hex[:8]


def test_users_do_not_bleed(client):
    alice, bob = _new_user(), _new_user()

    client.post(
        "/turns",
        json={
            "user_id": alice,
            "session_id": "a1",
            "messages": [{"role": "user", "content": "I live in Berlin."}],
        },
    )
    client.post(
        "/turns",
        json={
            "user_id": bob,
            "session_id": "b1",
            "messages": [{"role": "user", "content": "I live in Tokyo."}],
        },
    )

    # Bob's recall must never surface Alice's Berlin.
    r = client.post(
        "/recall",
        json={"user_id": bob, "session_id": "b2", "query": "Where do I live?"},
    )
    assert r.status_code == 200
    ctx = r.json()["context"].lower()
    assert "tokyo" in ctx
    assert "berlin" not in ctx


def test_same_user_cross_session_sharing(client):
    user = _new_user()

    # Write under session A.
    client.post(
        "/turns",
        json={
            "user_id": user,
            "session_id": "session-A",
            "messages": [{"role": "user", "content": "I live in Berlin."}],
        },
    )

    # Recall under a DIFFERENT session B for the same user — must surface it.
    r = client.post(
        "/recall",
        json={
            "user_id": user,
            "session_id": "session-B",
            "query": "Where do I live?",
        },
    )
    assert r.status_code == 200
    assert "berlin" in r.json()["context"].lower()
