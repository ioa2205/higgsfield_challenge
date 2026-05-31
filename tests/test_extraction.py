"""Phase 2: hybrid extraction produces typed, structured memories.

Covers §4 extraction quality: typed records (not raw chunks), an IMPLICIT fact
(Biscuit -> pet.name), a CORRECTION, SYNCHRONOUS correctness (queryable with no
delay after /turns), the rule FALLBACK when the LLM errors, and that a configured
LLM result is actually used. The fixture's two metrics are guarded by
test_fixture_metrics so extraction quality can't silently regress.
"""
from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from fixture_runner import format_report, run_fixtures


def _ingest(client, user, content, session="s1"):
    r = client.post(
        "/turns",
        json={
            "user_id": user,
            "session_id": session,
            "messages": [{"role": "user", "content": content}],
        },
    )
    assert r.status_code == 201, r.text
    return r


def _memories(client, user):
    return client.get(f"/users/{user}/memories").json()["memories"]


def test_memories_are_typed_and_structured(client, user_id):
    _ingest(client, user_id, "I work at Stripe as a backend engineer.")
    mems = _memories(client, user_id)
    assert mems, "expected at least one extracted memory"
    for m in mems:
        assert m["type"] in {"fact", "preference", "opinion", "event"}
        assert isinstance(m["value"], str) and m["value"].strip()
        assert isinstance(m["confidence"], (int, float))
        assert 0.0 <= m["confidence"] <= 1.0
        assert m["provenance"]  # which extraction path produced it

    employment = [m for m in mems if m["key"] == "employment"]
    assert employment, "employment fact not extracted"
    assert "stripe" in employment[0]["value"].lower()
    # Structured, not a raw echo of the message text.
    assert employment[0]["value"] != "I work at Stripe as a backend engineer."


def test_implicit_pet_fact(client, user_id):
    # The pet's name is only implied ("walking Biscuit"), never stated as a fact.
    _ingest(client, user_id, "Walking Biscuit this morning before work.")
    pets = [m for m in _memories(client, user_id) if m["key"] == "pet.name"]
    assert pets, "implicit pet.name not extracted"
    assert "biscuit" in pets[0]["value"].lower()
    assert pets[0]["type"] == "fact"


def test_correction_yields_new_value(client, user_id):
    _ingest(client, user_id, "I work at Stripe.", session="c1")
    _ingest(client, user_id, "Sorry, not Stripe — I actually work at Notion now.", session="c2")
    employers = [m for m in _memories(client, user_id) if m["key"] == "employment"]
    values = " ".join(m["value"].lower() for m in employers)
    assert "notion" in values, f"correction not captured: {values!r}"


def test_synchronous_correctness(client, user_id):
    """After /turns returns, the memory is in BOTH /memories and /recall with no
    delay (§5: if you wrote it, you can read it)."""
    _ingest(client, user_id, "I live in Berlin.")

    # Immediately queryable in the inspection endpoint...
    assert any("berlin" in m["value"].lower() for m in _memories(client, user_id))

    # ...and immediately recallable.
    r = client.post(
        "/recall",
        json={"user_id": user_id, "session_id": "other", "query": "Where do I live?"},
    )
    assert r.status_code == 200
    assert "berlin" in r.json()["context"].lower()


def test_rule_fallback_when_llm_errors(client, user_id):
    """If a provider is configured but the call fails/hangs (extractor returns
    None), extraction must fall back to rules and never crash."""

    class _FailingExtractor:
        provider = "gemini"

        def extract(self, messages):
            return None  # simulate timeout/error

    with patch(
        "src.extraction.pipeline.get_llm_extractors", return_value=[_FailingExtractor()]
    ):
        _ingest(client, user_id, "I live in Berlin and work at Acme.")

    mems = _memories(client, user_id)
    keys = {m["key"] for m in mems}
    assert "location" in keys and "employment" in keys
    assert all(m["provenance"].startswith("rule") for m in mems)
    locations = [m["value"] for m in mems if m["key"] == "location"]
    assert locations == ["Lives in Berlin"]


def test_llm_result_is_used_when_available(client, user_id):
    """When a configured LLM returns structured memories, they are persisted
    (normalised + tagged with llm provenance) instead of the rule output."""

    class _StubExtractor:
        provider = "gemini"

        def extract(self, messages):
            return [
                {
                    "type": "fact",
                    "key": "location",
                    "value": "Lives in Reykjavik",
                    "confidence": 0.93,
                }
            ]

    with patch(
        "src.extraction.pipeline.get_llm_extractors", return_value=[_StubExtractor()]
    ):
        _ingest(client, user_id, "some text the rules would parse differently")

    mems = _memories(client, user_id)
    assert len(mems) == 1
    assert mems[0]["value"] == "Lives in Reykjavik"
    assert mems[0]["provenance"] == "llm:gemini"
    assert abs(mems[0]["confidence"] - 0.93) < 1e-6


def test_fixture_metrics_baseline(client):
    """The recall-quality fixture: extraction must be fully captured and every
    in-scope recall probe must pass. Guards against extraction regressions."""
    report = run_fixtures(client)
    print(format_report(report))  # visible with `pytest -s`

    eh, et = report.extraction
    assert et > 0
    assert eh == et, f"extraction regressed: {eh}/{et}"

    rih, rit = report.recall_in_scope
    assert rih == rit, f"in-scope recall regressed: {rih}/{rit}"
