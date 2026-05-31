"""Delete semantics: session-scoped delete, user cascade, idempotency."""
from __future__ import annotations

import uuid


def _write(client, user, session, content):
    r = client.post(
        "/turns",
        json={
            "user_id": user,
            "session_id": session,
            "messages": [{"role": "user", "content": content}],
        },
    )
    assert r.status_code == 201


def test_delete_session_removes_only_that_session(client):
    user = "del-" + uuid.uuid4().hex[:8]
    _write(client, user, "sess-keep", "I live in Berlin.")
    _write(client, user, "sess-drop", "I work at Acme.")

    r = client.delete("/sessions/sess-drop")
    assert r.status_code == 204

    values = [m["value"] for m in client.get(f"/users/{user}/memories").json()["memories"]]
    assert any("Berlin" in v for v in values)
    assert not any("Acme" in v for v in values)


def test_delete_user_cascades(client):
    user = "del-" + uuid.uuid4().hex[:8]
    _write(client, user, "s1", "I live in Berlin.")
    _write(client, user, "s2", "I love cycling.")

    r = client.delete(f"/users/{user}")
    assert r.status_code == 204

    # Memories gone.
    assert client.get(f"/users/{user}/memories").json() == {"memories": []}

    # Recall empty-200.
    r = client.post(
        "/recall",
        json={"user_id": user, "session_id": "x", "query": "Where do I live?"},
    )
    assert r.status_code == 200
    assert r.json() == {"context": "", "citations": []}


def test_delete_is_idempotent(client):
    # Deleting non-existent ids still returns 204.
    assert client.delete("/sessions/does-not-exist").status_code == 204
    assert client.delete("/users/does-not-exist").status_code == 204
