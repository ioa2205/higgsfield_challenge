"""Contract: exact shapes and status codes from CHALLENGE.md §3."""
from __future__ import annotations


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_turn_recall_shapes_and_status(client, user_id):
    # Multi-message turn including a tool message with name set...
    r = client.post(
        "/turns",
        json={
            "user_id": user_id,
            "session_id": "sess-a",
            "messages": [
                {"role": "user", "content": "I live in Berlin and love cycling."},
                {"role": "assistant", "content": "Noted!"},
                {"role": "tool", "content": "{\"ok\": true}", "name": "save_pref"},
            ],
        },
    )
    assert r.status_code == 201
    body = r.json()
    assert set(body.keys()) == {"id"}
    assert isinstance(body["id"], str)

    # ...and a second turn whose tool message has name = null.
    r2 = client.post(
        "/turns",
        json={
            "user_id": user_id,
            "session_id": "sess-a",
            "messages": [
                {"role": "tool", "content": "tool output", "name": None},
            ],
        },
    )
    assert r2.status_code == 201

    # Recall (200) — shape per §3.
    r = client.post(
        "/recall",
        json={
            "user_id": user_id,
            "session_id": "sess-b",
            "query": "I live in Berlin and love cycling.",
            "top_k": 5,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"context", "citations"}
    assert isinstance(body["context"], str)
    assert "berlin" in body["context"].lower()
    assert len(body["citations"]) >= 1
    cit = body["citations"][0]
    assert set(cit.keys()) == {"turn_id", "score", "snippet"}
    assert isinstance(cit["score"], (int, float))


def test_search_shape_and_status(client, user_id):
    client.post(
        "/turns",
        json={
            "user_id": user_id,
            "session_id": "sess-a",
            "messages": [{"role": "user", "content": "My favourite colour is teal."}],
        },
    )
    r = client.post(
        "/search",
        json={"user_id": user_id, "query": "favourite colour is teal", "limit": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"results"}
    assert len(body["results"]) >= 1
    res = body["results"][0]
    assert set(res.keys()) == {"content", "score", "session_id", "timestamp", "metadata"}
    assert isinstance(res["score"], (int, float))


def test_list_memories_shape_and_status(client, user_id):
    client.post(
        "/turns",
        json={
            "user_id": user_id,
            "session_id": "sess-a",
            "messages": [{"role": "user", "content": "I have a dog named Rex."}],
        },
    )
    r = client.get(f"/users/{user_id}/memories")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"memories"}
    assert len(body["memories"]) >= 1
    mem = body["memories"][0]
    required = {"id", "type", "key", "value", "confidence", "active"}
    assert required.issubset(mem.keys())
    # Structured & typed — not raw message text masquerading as a memory.
    assert mem["type"] in {"fact", "preference", "opinion", "event"}
    assert mem["active"] is True


def test_cold_recall_empty_200(client, user_id):
    r = client.post(
        "/recall",
        json={"user_id": user_id, "session_id": "x", "query": "anything", "top_k": 5},
    )
    assert r.status_code == 200
    assert r.json() == {"context": "", "citations": []}


def test_empty_search_empty_200(client, user_id):
    r = client.post("/search", json={"user_id": user_id, "query": "anything"})
    assert r.status_code == 200
    assert r.json() == {"results": []}
