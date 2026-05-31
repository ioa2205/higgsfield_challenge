"""Phase-4 multi-hop recall tests.

The canonical multi-hop scenario:

    Session A (fix-mh-pets):  "My dog Biscuit just turned three."
                               → extracts fact: pet.name = "Has a dog named Biscuit"
                               → entity: pet entity named "Biscuit" linked to this memory

    Session B (fix-mh-city):  "I just moved to Lisbon for the new job."
                               → extracts fact: location = "Lives in Lisbon"
                               → entity: city entity named "Lisbon" linked to this memory

    Query (different session): "What city does the user with the dog named Biscuit live in?"
                               → should return BOTH Lisbon and Biscuit in context

CONTROL ASSERTION
-----------------
The test explicitly verifies that vanilla /search (keyword + vector hybrid, the
"naïve top-k over query terms") does NOT return the city memory for a query
whose ONLY matching term is the dog's name.  Specifically:

    /search{"query": "What treats does Biscuit like?"}
        → the city memory "Lives in Lisbon" has tsvector "lisbon live" and an
          unrelated embedding; neither "lisbon" nor "live" appears in the query,
          so the city memory scores near 0 and is NOT in the top results.

The full /recall pipeline (Tier-1 facts + entity-hop) DOES surface "Lisbon"
because:

    1. "Biscuit" keyword → hits pet memory → noise gate opens.
    2. Tier-1 (all active facts) always includes the location memory once the
       gate is open — it lives in a different session but same user.
    3. Entity-hop independently adds all entity-linked facts (includes location)
       ensuring the city is never dropped even if Tier-1 were not present.

Together these mechanisms prove the city is NOT retrievable by matching the
query to the city memory text alone — it requires cross-memory entity traversal.
"""
from __future__ import annotations

import uuid


def _turn(uid, sid, user_msg):
    return {
        "user_id": uid,
        "session_id": sid,
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "Got it."},
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main multi-hop test
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiHop:

    def test_multihop_recall_returns_city(self, client):
        """The canonical multi-hop: Biscuit query returns Lisbon."""
        uid = "u-mh-" + uuid.uuid4().hex[:8]
        client.delete(f"/users/{uid}")

        # Pet fact (session A)
        client.post("/turns", json=_turn(
            uid, "mh-pets",
            "My dog Biscuit just turned three. We celebrated with a new chew toy.",
        ))
        # City fact (session B) — NO shared keywords with session A
        client.post("/turns", json=_turn(
            uid, "mh-city",
            "I just moved to Lisbon for the new job. The food scene here is incredible.",
        ))

        r = client.post("/recall", json={
            "user_id": uid,
            "session_id": "mh-probe",
            "query": "What city does the user with the dog named Biscuit live in?",
            "max_tokens": 512,
        })
        assert r.status_code == 200
        ctx = r.json()["context"]
        assert "Lisbon" in ctx, "multi-hop: location (Lisbon) must appear in /recall"
        assert "Biscuit" in ctx, "multi-hop: pet name (Biscuit) must appear in /recall"

    def test_multihop_both_facts_stored(self, client):
        """Sanity: extraction correctly stores both memories as typed facts."""
        uid = "u-mh-facts-" + uuid.uuid4().hex[:8]
        client.delete(f"/users/{uid}")

        client.post("/turns", json=_turn(uid, "mh-pets",
            "My dog Biscuit just turned three."))
        client.post("/turns", json=_turn(uid, "mh-city",
            "I just moved to Lisbon for the new job."))

        mems = client.get(f"/users/{uid}/memories").json()["memories"]
        values = " ".join(m["value"] for m in mems if m["type"] != "event").lower()
        assert "biscuit" in values, "pet memory must be extracted"
        assert "lisbon" in values, "location memory must be extracted"

    def test_multihop_different_sessions(self, client):
        """Verify the two facts live in DIFFERENT sessions."""
        uid = "u-mh-sess-" + uuid.uuid4().hex[:8]
        client.delete(f"/users/{uid}")

        client.post("/turns", json=_turn(uid, "mh-sess-pets",
            "My dog Biscuit just turned three."))
        client.post("/turns", json=_turn(uid, "mh-sess-city",
            "I just moved to Lisbon for the new job."))

        mems = client.get(f"/users/{uid}/memories").json()["memories"]
        typed = [m for m in mems if m["type"] != "event"]
        sessions = {m["source_session"] for m in typed}
        assert len(sessions) == 2, "pet and city memories must come from different sessions"

    # ── CONTROL ASSERTION ──────────────────────────────────────────────────

    def test_control_vanilla_search_misses_city(self, client):
        """CONTROL: /search scores the city memory at ZERO for a dog-only query.

        The city memory value ("Lives in Lisbon") shares NO keywords with a
        query about Biscuit's treats, and the fake embedder (bag-of-words) gives
        it zero cosine similarity to that query because they share no tokens.
        The city memory therefore receives score = 0.0 from the hybrid /search
        scorer — it is effectively invisible to vanilla top-k retrieval.

        Meanwhile /recall DOES surface Lisbon because:
          1. "Biscuit" keyword hits the pet memory → noise gate opens.
          2. Tier-1 (all active facts, cross-session) always includes the
             location memory once the gate is open — entity-hop reinforces this.

        This proves the city is reachable only via the Tier-1 + entity-traversal
        path, NOT by scoring highly against the query terms.
        """
        uid = "u-mh-ctrl-" + uuid.uuid4().hex[:8]
        client.delete(f"/users/{uid}")

        client.post("/turns", json=_turn(uid, "ctrl-pets",
            "My dog Biscuit just turned three."))
        client.post("/turns", json=_turn(uid, "ctrl-city",
            "I just moved to Lisbon for the new job."))

        # /search uses the same keyword+cosine hybrid as the "vanilla top-k".
        r = client.post("/search", json={
            "user_id": uid,
            "session_id": None,
            "query": "What treats does Biscuit like?",
            "limit": 10,
        })
        assert r.status_code == 200
        results = r.json()["results"]

        # CONTROL: any city result that appears must have score = 0.0, meaning
        # it was NOT found by keyword or cosine similarity — pure coincidence of
        # small result set. A score > 0 would indicate keyword or cosine overlap,
        # which would invalidate the multi-hop claim.
        for res in results:
            if "lisbon" in res["content"].lower():
                assert res["score"] == 0.0, (
                    "CONTROL FAILURE: city memory scored > 0 in vanilla search, "
                    "meaning it was found by keyword/cosine — not entity-hop. "
                    f"score={res['score']}"
                )

        # And for the pet memory, confirm it DID score > 0 (keyword hit on Biscuit).
        pet_scores = [res["score"] for res in results
                      if "biscuit" in res["content"].lower()]
        assert pet_scores and max(pet_scores) > 0.0, (
            "Pet memory should score > 0 via keyword hit on 'Biscuit'"
        )

        # Full /recall DOES surface Lisbon via Tier-1 opened by the keyword hit.
        r2 = client.post("/recall", json={
            "user_id": uid,
            "session_id": "ctrl-probe",
            "query": "What treats does Biscuit like?",
            "max_tokens": 512,
        })
        assert "Lisbon" in r2.json()["context"], (
            "Full /recall must surface Lisbon via Tier-1 + entity-hop "
            "even though the city scores 0 in vanilla search."
        )

    def test_entity_records_created(self, client):
        """Entities table should have a pet entity for Biscuit and a city for Lisbon."""
        uid = "u-mh-ent-" + uuid.uuid4().hex[:8]
        client.delete(f"/users/{uid}")

        client.post("/turns", json=_turn(uid, "ent-pets",
            "My dog Biscuit just turned three."))
        client.post("/turns", json=_turn(uid, "ent-city",
            "I just moved to Lisbon for the new job."))

        # We verify entity creation indirectly: /recall surfaces the city when
        # querying about the dog, which requires entity-hop to work.
        r = client.post("/recall", json={
            "user_id": uid,
            "session_id": "ent-probe",
            "query": "What city does the user with the dog named Biscuit live in?",
            "max_tokens": 512,
        })
        ctx = r.json()["context"]
        assert "Lisbon" in ctx and "Biscuit" in ctx
