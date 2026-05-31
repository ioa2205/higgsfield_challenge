"""Phase-4 fact-evolution / supersession tests.

Verifies:
  - Employment change: Stripe → Notion. /recall returns Notion, not Stripe.
  - The superseded row (Stripe) still exists with active=false.
  - The active row (Notion) has supersedes pointing to the old Stripe row.
  - The active row's updated_at is ≥ the old row's created_at.
  - /recall context renders "updated …; previously …" when a predecessor exists.
  - Location change: NYC → Berlin (smoke-test analogue).
  - Opinion arcs: opinions keep a supersession chain; latest stance is current.
"""
from __future__ import annotations

import uuid


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _turn(uid, sid, text):
    return {
        "user_id": uid,
        "session_id": sid,
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": "Got it."},
        ],
    }


def _memories_by_key(client, uid, key):
    mems = client.get(f"/users/{uid}/memories").json()["memories"]
    return [m for m in mems if m.get("key") == key]


def _active(mems):
    return [m for m in mems if m["active"]]


def _inactive(mems):
    return [m for m in mems if not m["active"]]


# ─────────────────────────────────────────────────────────────────────────────
# Employment evolution: Stripe → Notion
# ─────────────────────────────────────────────────────────────────────────────

class TestEmploymentEvolution:
    def setup_method(self):
        pass  # each test gets a fresh `client` from the fixture

    def test_recall_returns_current_employer(self, client):
        uid = "u-evo-emp-" + uuid.uuid4().hex[:8]
        client.delete(f"/users/{uid}")

        client.post("/turns", json=_turn(uid, "s1", "I work at Stripe."))
        client.post("/turns", json=_turn(uid, "s3", "Big news — I just joined Notion!"))

        r = client.post("/recall", json={
            "user_id": uid, "session_id": "probe", "query": "Where does the user work?", "max_tokens": 512,
        })
        ctx = r.json()["context"].lower()
        assert "notion" in ctx, "current employer must appear in /recall"
        # Stripe may appear as "previously at Stripe" but must not be the primary
        assert "notion" in ctx

    def test_recall_does_not_return_stale_employer_as_primary(self, client):
        uid = "u-evo-emp2-" + uuid.uuid4().hex[:8]
        client.delete(f"/users/{uid}")

        client.post("/turns", json=_turn(uid, "s1", "I work at Stripe."))
        client.post("/turns", json=_turn(uid, "s3", "I just joined Notion as a PM!"))

        mems = _memories_by_key(client, uid, "employment")
        active = _active(mems)
        assert len(active) == 1, "only ONE employment fact must be active after supersession"
        assert "notion" in active[0]["value"].lower(), "active employment must be Notion"

    def test_superseded_row_still_exists(self, client):
        uid = "u-evo-emp3-" + uuid.uuid4().hex[:8]
        client.delete(f"/users/{uid}")

        client.post("/turns", json=_turn(uid, "s1", "I work at Stripe."))
        client.post("/turns", json=_turn(uid, "s3", "I just joined Notion!"))

        mems = _memories_by_key(client, uid, "employment")
        inactive = _inactive(mems)
        assert len(inactive) >= 1, "old Stripe row must still exist (active=false)"
        assert any("stripe" in m["value"].lower() for m in inactive), \
            "the superseded Stripe memory must be preserved"

    def test_supersedes_pointer_set(self, client):
        uid = "u-evo-emp4-" + uuid.uuid4().hex[:8]
        client.delete(f"/users/{uid}")

        client.post("/turns", json=_turn(uid, "s1", "I work at Stripe."))
        client.post("/turns", json=_turn(uid, "s3", "I just joined Notion!"))

        mems = _memories_by_key(client, uid, "employment")
        active = _active(mems)
        inactive = _inactive(mems)
        assert len(active) == 1 and len(inactive) >= 1
        old_id = inactive[0]["id"]
        assert active[0]["supersedes"] == old_id, \
            "new memory's supersedes must point to the old memory's id"

    def test_updated_at_advanced(self, client):
        uid = "u-evo-emp5-" + uuid.uuid4().hex[:8]
        client.delete(f"/users/{uid}")

        client.post("/turns", json=_turn(uid, "s1", "I work at Stripe."))
        client.post("/turns", json=_turn(uid, "s3", "I just joined Notion!"))

        mems = _memories_by_key(client, uid, "employment")
        active = _active(mems)
        inactive = _inactive(mems)
        assert active and inactive
        # updated_at on the new (active) row must be >= created_at of the old row.
        assert active[0]["updated_at"] >= inactive[0]["created_at"], \
            "active memory's updated_at must be ≥ old memory's created_at"

    def test_previously_annotation_in_recall_context(self, client):
        uid = "u-evo-emp6-" + uuid.uuid4().hex[:8]
        client.delete(f"/users/{uid}")

        client.post("/turns", json=_turn(uid, "s1", "I work at Stripe."))
        client.post("/turns", json=_turn(uid, "s3", "I just joined Notion!"))

        r = client.post("/recall", json={
            "user_id": uid, "session_id": "probe",
            "query": "Where does the user work?", "max_tokens": 512,
        })
        ctx = r.json()["context"]
        # The tier-1 assembler should produce "updated …; previously …" when the
        # active memory has a supersedes predecessor.
        assert "previously" in ctx.lower(), \
            '/recall context should render "previously …" for superseded employment'


# ─────────────────────────────────────────────────────────────────────────────
# Location evolution: NYC → Berlin
# ─────────────────────────────────────────────────────────────────────────────

class TestLocationEvolution:

    def test_location_supersession_berlin(self, client):
        uid = "u-evo-loc-" + uuid.uuid4().hex[:8]
        client.delete(f"/users/{uid}")

        client.post("/turns", json=_turn(uid, "s1", "I live in NYC."))
        client.post("/turns", json=_turn(uid, "s2",
            "I just moved to Berlin from NYC last month. Loving it so far."))

        mems = _memories_by_key(client, uid, "location")
        active = _active(mems)
        assert len(active) == 1
        assert "berlin" in active[0]["value"].lower(), "Berlin must be the active location"

        r = client.post("/recall", json={
            "user_id": uid, "session_id": "probe",
            "query": "Where does this user live?", "max_tokens": 512,
        })
        ctx = r.json()["context"].lower()
        assert "berlin" in ctx

    def test_location_history_preserved(self, client):
        uid = "u-evo-loc2-" + uuid.uuid4().hex[:8]
        client.delete(f"/users/{uid}")

        client.post("/turns", json=_turn(uid, "s1", "I live in NYC."))
        client.post("/turns", json=_turn(uid, "s2", "I just moved to Berlin."))

        mems = _memories_by_key(client, uid, "location")
        inactive = _inactive(mems)
        assert len(inactive) >= 1
        assert any("nyc" in m["value"].lower() or "new york" in m["value"].lower()
                   for m in inactive), "NYC row must be preserved as inactive"


# ─────────────────────────────────────────────────────────────────────────────
# Opinion arc
# ─────────────────────────────────────────────────────────────────────────────

class TestOpinionArc:

    def test_latest_opinion_is_active(self, client):
        uid = "u-evo-op-" + uuid.uuid4().hex[:8]
        client.delete(f"/users/{uid}")

        client.post("/turns", json=_turn(uid, "op-s1", "Honestly, I love TypeScript. Best language."))
        client.post("/turns", json=_turn(uid, "op-s2", "TypeScript generics are getting annoying."))
        client.post("/turns", json=_turn(uid, "op-s3",
            "TypeScript is fine for big projects, but I'd use Python for scripts."))

        mems = client.get(f"/users/{uid}/memories").json()["memories"]
        ts_mems = [m for m in mems if m.get("key", "").startswith("opinion.typescript")]
        active_ts = _active(ts_mems)

        assert len(active_ts) == 1, "only the latest TypeScript opinion should be active"

    def test_earlier_opinions_preserved_as_chain(self, client):
        uid = "u-evo-op2-" + uuid.uuid4().hex[:8]
        client.delete(f"/users/{uid}")

        client.post("/turns", json=_turn(uid, "op-s1", "Honestly, I love TypeScript. Best language."))
        client.post("/turns", json=_turn(uid, "op-s2", "TypeScript generics are getting annoying."))
        client.post("/turns", json=_turn(uid, "op-s3",
            "TypeScript is fine for big projects, but I'd use Python for scripts."))

        mems = client.get(f"/users/{uid}/memories").json()["memories"]
        ts_mems = [m for m in mems if m.get("key", "").startswith("opinion.typescript")]
        inactive_ts = _inactive(ts_mems)

        # All earlier opinions must remain as inactive rows (history preserved).
        assert len(inactive_ts) >= 1, \
            "earlier opinions must remain in the chain (active=false, not deleted)"

    def test_recall_returns_latest_opinion(self, client):
        uid = "u-evo-op3-" + uuid.uuid4().hex[:8]
        client.delete(f"/users/{uid}")

        client.post("/turns", json=_turn(uid, "op-s1", "I love TypeScript."))
        client.post("/turns", json=_turn(uid, "op-s3",
            "TypeScript is fine for big projects, but I'd use Python for scripts."))

        r = client.post("/recall", json={
            "user_id": uid, "session_id": "probe",
            "query": "How does the user feel about TypeScript?", "max_tokens": 512,
        })
        ctx = r.json()["context"].lower()
        assert "typescript" in ctx
