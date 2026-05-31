"""Phase 3: hybrid recall + RRF + tiered, token-budgeted assembly.

Covers the §3/§4 /recall contract that the eval scores hardest:
  * token-budget triage (≤ ~1× max_tokens, never > 2×) including unicode-heavy
    input, with Tier-1 stable facts winning over recent chatter;
  * noise resistance (an undiscussed topic → empty context, not a hallucinated
    digest);
  * the structured /search shape with scoping + limit;
  * a coarse /recall latency ceiling on the fixture.

In-process app + fake embedder (see tests/conftest.py): keyword stemming carries
the deterministic relevance, so these assertions don't depend on a real model.
"""
from __future__ import annotations

import time

from fixture_runner import load_fixtures, run_fixtures
from src.recall.assembly import estimate_tokens


def _ingest(client, user, content, session):
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


def _recall(client, user, query, session="probe", max_tokens=512):
    r = client.post(
        "/recall",
        json={
            "user_id": user,
            "session_id": session,
            "query": query,
            "max_tokens": max_tokens,
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


# --- token budget (incl. unicode) ----------------------------------------- #
def test_budget_respected_with_unicode(client, user_id):
    """A tight budget must not be blown past ~2×, even when the assembled context
    carries multibyte/emoji content, and Tier-1 facts must survive the squeeze."""
    sess = "budget-s1"
    # A unicode-bearing fact (emoji + accents) lands directly in the Tier-1 digest.
    _ingest(client, user_id, "I'm allergic to crustacés 🦐 and live in Berlin.", sess)
    # A long, unicode-heavy recent event in the SAME session (Tier-3 candidate)
    # that the tight budget should squeeze out in favour of Tier-1.
    _ingest(
        client,
        user_id,
        "Standup notes 我们讨论了整个路线图和发布计划 🚀🚀🚀 café crème résumé naïve coördination "
        "façade Größe 这是一段很长的中文文本用来撑满预算 with many extra words to inflate the estimate.",
        sess,
    )

    max_tokens = 64
    body = _recall(client, user_id, "What is the user allergic to?", session=sess, max_tokens=max_tokens)
    ctx = body["context"]

    assert ctx, "expected non-empty context for a relevant query"
    est = estimate_tokens(ctx)
    # Conservative over-count must never exceed 2× the budget (§4 guarantee)...
    assert est <= 2 * max_tokens, f"budget blown: est={est:.0f} > 2×{max_tokens}"
    # ...and in practice stays in the neighbourhood of 1×.
    assert est <= 1.5 * max_tokens, f"budget loose: est={est:.0f} for {max_tokens}"
    # Tier-1 wins the squeeze: the unicode fact is present, the long chatter cut.
    assert ctx.startswith("## Known facts about this user")
    assert "🦐" in ctx and "berlin" in ctx.lower()
    assert "中文文本" not in ctx, "recent chatter should have been trimmed under budget"


def test_tier1_facts_precede_recent_chatter(client, user_id):
    """With room to spare, both sections render — but Tier-1 (stable facts) must
    come BEFORE the 'recent conversations' tier (the §3 priority order)."""
    sess = "order-s1"
    _ingest(client, user_id, "I live in Berlin.", sess)
    _ingest(client, user_id, "The deployment finally finished after lunch.", sess)

    body = _recall(client, user_id, "Where do I live?", session=sess, max_tokens=512)
    ctx = body["context"].lower()

    assert "## known facts about this user" in ctx
    assert "## relevant from recent conversations" in ctx
    # Stable-fact section precedes the recent-chatter section...
    assert ctx.index("## known facts") < ctx.index("## relevant from recent")
    # ...and the Berlin fact precedes the deployment chatter.
    assert ctx.index("berlin") < ctx.index("deployment")


# --- noise resistance ----------------------------------------------------- #
def test_noise_query_returns_empty(client, user_id):
    """A query about a topic the user never discussed returns empty context and
    no citations (§9) — the digest is NOT dumped at an unrelated question."""
    _ingest(client, user_id, "I work at Stripe.", "noise-s1")
    body = _recall(client, user_id, "What is the user's favorite movie genre?")
    assert body == {"context": "", "citations": []}


# --- /search structured shape + scoping + limit --------------------------- #
def test_search_structured_scoping_and_limit(client, user_id):
    # Two sessions for the same user, each with a distinct preference.
    _ingest(client, user_id, "My favourite colour is teal.", "search-a")
    _ingest(client, user_id, "My favourite colour is crimson.", "search-b")

    # User-scoped search returns structured rows honouring the limit.
    r = client.post(
        "/search",
        json={"user_id": user_id, "query": "favourite colour", "limit": 1},
    )
    assert r.status_code == 200
    results = r.json()["results"]
    assert 1 <= len(results) <= 1  # limit respected
    res = results[0]
    assert set(res.keys()) == {"content", "score", "session_id", "timestamp", "metadata"}
    assert isinstance(res["score"], (int, float))
    assert isinstance(res["metadata"], dict)

    # Session-scoped search only surfaces that session's memory.
    r = client.post(
        "/search",
        json={"user_id": user_id, "session_id": "search-b", "query": "favourite colour", "limit": 10},
    )
    rows = r.json()["results"]
    assert rows, "session-scoped search should find the crimson memory"
    assert all(row["session_id"] == "search-b" for row in rows)
    assert any("crimson" in row["content"].lower() for row in rows)
    assert all("teal" not in row["content"].lower() for row in rows)


# --- coarse latency ------------------------------------------------------- #
def test_recall_latency_under_threshold(client):
    """Every fixture probe's /recall must return well under a documented ceiling.
    In-process with the fake embedder this is typically tens of ms; 2.0s is a
    generous CI-safe bound that still catches a pathological regression."""
    data = load_fixtures()
    worst = 0.0
    for scenario in data["scenarios"]:
        user = scenario["user_id"]
        client.delete(f"/users/{user}")
        for convo in scenario["conversations"]:
            client.post(
                "/turns",
                json={
                    "user_id": user,
                    "session_id": convo["session_id"],
                    "messages": convo["messages"],
                },
            )
        for probe in scenario["probes"]:
            t0 = time.perf_counter()
            r = client.post(
                "/recall",
                json={
                    "user_id": user,
                    "session_id": f"{scenario['id']}-lat",
                    "query": probe["query"],
                    "max_tokens": 512,
                },
            )
            dt = time.perf_counter() - t0
            assert r.status_code == 200
            worst = max(worst, dt)
    assert worst < 2.0, f"slowest /recall took {worst:.3f}s (> 2.0s ceiling)"
