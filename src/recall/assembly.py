"""Tiered, token-budgeted context assembly (the §3 /recall body).

Priority tiers (§3/§4 "what gets included when budget is tight"):

  * **Tier 1 — "## Known facts about this user"**: the user's ACTIVE, stable,
    high-confidence facts/preferences/opinions, cross-session. This is the
    durable digest a frozen agent always benefits from; it wins the budget.
    Each line can carry an "updated …; previously …" annotation once Phase 4
    sets ``supersedes`` (the predecessor value is already fetched in Tier 1).
  * **Tier 2/3 — "## Relevant from recent conversations"**: query-relevant
    ``event`` memories (RRF-fused vector+keyword) plus recent session-scoped
    events, dated ``[YYYY-MM-DD]``.

**Noise gate.** Assembly emits NOTHING (``"", []``) unless at least one memory
is genuinely relevant to the query — a keyword hit (ts_rank > 0) or a vector hit
clearing ``RECALL_MIN_SCORE``. A query about a topic the user never discussed
clears neither, so /recall returns empty (§9 noise resistance) rather than
dumping the stable-facts digest at an unrelated question.

**Budget.** A conservative, dependency-free OVER-count — ``max(words×1.3,
chars/4)`` — keeps the estimate ≤ ~1× ``max_tokens`` and never near 2×, even on
unicode-heavy input. Tier 1 is filled first (priority), then Tier 2/3 with the
remaining budget.
"""
from __future__ import annotations

from typing import Any

from .. import config
from .fusion import Fused

_T1_HEADER = "## Known facts about this user"
_T2_HEADER = "## Relevant from recent conversations"


def estimate_tokens(text: str) -> float:
    """Conservative over-count so we never blow the budget (§4). max(words×1.3,
    chars/4): word path catches prose, char path catches unicode/long tokens."""
    words = len(text.split())
    chars = len(text)
    return max(words * 1.3, chars / 4.0)


def _snippet(value: str) -> str:
    value = " ".join(value.split())
    if len(value) > config.SNIPPET_MAX:
        return value[: config.SNIPPET_MAX - 1].rstrip() + "…"
    return value


def _date(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return value.date().isoformat()
    except AttributeError:
        try:
            return value.isoformat()[:10]
        except AttributeError:
            return str(value)[:10]


def _fact_line(row: dict) -> str:
    value = _snippet(row["value"])
    prev = row.get("prev_value")
    if prev:  # Phase-4 supersession annotation (null until then)
        when = _date(row.get("updated_at")) or ""
        when = f"updated {when}; " if when else "updated; "
        value = f"{value} ({when}previously {_snippet(prev)})"
    return f"- {value}"


def _event_line(row: dict) -> str:
    date = _date(row.get("created_at"))
    body = _snippet(row["value"])
    return f"- [{date}] {body}" if date else f"- {body}"


def _citation(row: dict, score: float) -> dict:
    return {
        "turn_id": str(row["source_turn"]) if row.get("source_turn") else "",
        "score": round(float(score), 4),
        "snippet": _snippet(row["value"]),
    }


def assemble(
    tier1: list[dict],
    fused: list[Fused],
    recent: list[dict],
    *,
    max_tokens: int,
    entity_hop: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """Build (context, citations). Returns ("", []) when nothing is relevant.

    ``entity_hop`` (Phase 4) contains fact memories reachable via the entity
    graph that plain hybrid recall may not surface.  They are merged into the
    Tier-1 candidate pool IF the primary recall (keyword OR vector) already
    fired — the noise gate still blocks an entirely off-topic query.
    """
    # --- noise gate: is anything actually relevant to the query? -----------
    relevant = [
        f
        for f in fused
        if f.keyword_hit
        or (f.cosine is not None and f.cosine >= config.RECALL_MIN_SCORE)
    ]

    # --- entity hop: if primary recall fired, widen the Tier-1 pool -------
    # When an entity name from the query matched a known user entity we add
    # ALL active facts to tier1 (they are already there if they exist, but the
    # entity-hop list may contain cross-session facts that keyword+vector missed).
    hop_ids: set = set()
    extra_tier1: list[dict] = []
    if relevant and entity_hop:
        tier1_ids = {r["id"] for r in tier1}
        for m in entity_hop:
            if m["id"] not in tier1_ids and m["id"] not in hop_ids:
                hop_ids.add(m["id"])
                extra_tier1.append(m)
        # Merge at the end of tier1 (lower priority than the existing digest).
        tier1 = tier1 + extra_tier1

    if not relevant and not entity_hop:
        return "", []

    # If ONLY entity-hop fired (no primary recall hit), still return context
    # so multi-hop queries work even when the query terms alone match nothing.
    if not relevant and entity_hop and tier1:
        pass  # fall through to assemble with entity_hop memories

    score_by_id = {f.memory["id"]: (f.cosine if f.cosine is not None else f.rrf_score)
                   for f in fused}

    # --- Tier 2/3 event pool: relevant events first, then recent session ---
    event_rows: list[dict] = []
    seen: set = set()
    for f in relevant:
        m = f.memory
        if m.get("type") == "event" and m["id"] not in seen:
            seen.add(m["id"])
            event_rows.append(m)
    for m in recent:
        if m["id"] not in seen:
            seen.add(m["id"])
            event_rows.append(m)

    # --- greedy budget fill, Tier 1 wins ----------------------------------
    fact_lines: list[str] = []
    fact_cites: list[dict] = []
    for row in tier1:
        line = _fact_line(row)
        trial = "\n".join([_T1_HEADER, *fact_lines, line])
        if estimate_tokens(trial) > max_tokens and fact_lines:
            break  # keep at least the top fact even if it alone is large
        fact_lines.append(line)
        fact_cites.append(_citation(row, score_by_id.get(row["id"], row.get("confidence", 0.0))))

    sections: list[str] = []
    citations: list[dict] = []
    if fact_lines:
        sections.append("\n".join([_T1_HEADER, *fact_lines]))
        citations.extend(fact_cites)

    event_lines: list[str] = []
    for row in event_rows:
        line = _event_line(row)
        trial = "\n\n".join(sections + ["\n".join([_T2_HEADER, *event_lines, line])])
        if estimate_tokens(trial) > max_tokens:
            break  # Tier 1 already placed; stop adding lower-priority events
        event_lines.append(line)
        citations.append(_citation(row, score_by_id.get(row["id"], 0.0)))
    if event_lines:
        sections.append("\n".join([_T2_HEADER, *event_lines]))

    return "\n\n".join(sections), citations
