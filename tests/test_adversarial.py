"""Compact hidden-eval-style regression cases beyond the original fixture."""
from __future__ import annotations

import uuid
from pathlib import Path

from fixture_runner import format_report, load_fixtures, run_fixtures

ADVERSARIAL = Path(__file__).resolve().parents[1] / "fixtures" / "adversarial_conversations.json"


def _turn(user, session, content):
    return {
        "user_id": user,
        "session_id": session,
        "messages": [{"role": "user", "content": content}],
    }


def _memories(client, user):
    return client.get(f"/users/{user}/memories").json()["memories"]


def test_adversarial_fixture_metrics(client):
    report = run_fixtures(client, load_fixtures(ADVERSARIAL))
    print(format_report(report))
    assert report.extraction == (9, 9)
    assert report.recall_in_scope == (11, 11)


def test_location_correction_keeps_porto_current(client):
    user = "adv-correction-" + uuid.uuid4().hex[:8]
    client.post("/turns", json=_turn(user, "s1", "I live in Lisbon."))
    client.post("/turns", json=_turn(user, "s2", "Not Lisbon, I meant Porto."))
    rows = [m for m in _memories(client, user) if m["key"] == "location"]
    assert len(rows) == 2
    assert [m["value"] for m in rows if m["active"]] == ["Lives in Porto"]


def test_employment_retirement_supersedes_positive_fact(client):
    user = "adv-retirement-" + uuid.uuid4().hex[:8]
    client.post("/turns", json=_turn(user, "s1", "I work at Stripe."))
    client.post("/turns", json=_turn(user, "s2", "I no longer work at Stripe."))
    rows = [m for m in _memories(client, user) if m["key"] == "employment"]
    assert len(rows) == 2
    assert [m["value"] for m in rows if m["active"]] == ["No longer works at Stripe"]


def test_employment_retirement_phrasings_do_not_reactivate_positive_fact(client):
    for index, phrase in enumerate(
        ("I don't work at Stripe.", "I do not work at Stripe.", "I stopped working at Stripe.")
    ):
        user = f"adv-retirement-phrasing-{index}-" + uuid.uuid4().hex[:8]
        client.post("/turns", json=_turn(user, "s1", "I work at Stripe."))
        client.post("/turns", json=_turn(user, "s2", phrase))
        rows = [m for m in _memories(client, user) if m["key"] == "employment"]
        active = [m["value"] for m in rows if m["active"]]
        assert active == ["No longer works at Stripe"], phrase


def test_multiple_pets_remain_active(client):
    user = "adv-pets-" + uuid.uuid4().hex[:8]
    client.post(
        "/turns",
        json=_turn(user, "s1", "I have a dog named Biscuit and a cat named Mochi."),
    )
    rows = [m for m in _memories(client, user) if m["key"] == "pet.name" and m["active"]]
    values = " ".join(m["value"] for m in rows).lower()
    assert len(rows) == 2
    assert "biscuit" in values and "mochi" in values


def test_tool_events_remain_append_only(client):
    user = "adv-tool-" + uuid.uuid4().hex[:8]
    payload = {
        "user_id": user,
        "session_id": "tool-s1",
        "messages": [{"role": "tool", "name": "calendar", "content": "Calendar result: dentist Friday."}],
    }
    client.post("/turns", json=payload)
    client.post("/turns", json=payload)
    rows = [m for m in _memories(client, user) if m["type"] == "event"]
    assert len(rows) == 2
    assert all(m["active"] for m in rows)


def test_similar_employers_do_not_bleed_between_users(client):
    alice = "adv-alice-" + uuid.uuid4().hex[:8]
    bob = "adv-bob-" + uuid.uuid4().hex[:8]
    client.post("/turns", json=_turn(alice, "a1", "I work at Linear."))
    client.post("/turns", json=_turn(bob, "b1", "I work at Stripe."))
    recall = client.post(
        "/recall",
        json={"user_id": alice, "session_id": "probe", "query": "Who is my employer?"},
    ).json()["context"].lower()
    assert "linear" in recall
    assert "stripe" not in recall
