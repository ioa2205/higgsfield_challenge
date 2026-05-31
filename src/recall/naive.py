"""Recall context assembly.

Renders the recalled memories as the readable, sectioned context block §3 shows
to a frozen LLM, plus the structured ``{turn_id, score, snippet}`` citation list
§3 specifies. Memories are grouped by kind: durable user knowledge (fact /
preference / opinion) under "Known facts about this user", and ``event``
memories under "Relevant from recent conversations" — stable facts first, which
is the priority §3 asks for when context is tight.

Ranking is still cosine top-k (the caller pre-orders rows). Hybrid retrieval,
RRF, and true token-budgeted tiering arrive in Phase 3; this phase only makes
the already-recalled set read well and groups it by priority.
"""
from __future__ import annotations

_SNIPPET_MAX = 240
_FACT_TYPES = ("fact", "preference", "opinion")


def _snippet(value: str) -> str:
    value = value.strip()
    if len(value) > _SNIPPET_MAX:
        return value[: _SNIPPET_MAX - 1].rstrip() + "…"
    return value


def assemble_context(rows: list[dict]) -> tuple[str, list[dict]]:
    """Build (context, citations) from recalled memory rows (ordered by score).

    Rows are dicts with at least value/score/source_turn/type.
    """
    if not rows:
        return "", []

    facts = [r for r in rows if r.get("type") in _FACT_TYPES]
    events = [r for r in rows if r.get("type") == "event"]

    lines: list[str] = []
    if facts:
        lines.append("## Known facts about this user")
        lines.extend(f"- {_snippet(r['value'])}" for r in facts)
    if events:
        if lines:
            lines.append("")
        lines.append("## Relevant from recent conversations")
        lines.extend(f"- {_snippet(r['value'])}" for r in events)

    citations = [
        {
            "turn_id": str(r["source_turn"]) if r["source_turn"] else "",
            "score": round(float(r["score"]), 4),
            "snippet": _snippet(r["value"]),
        }
        for r in rows
    ]
    return "\n".join(lines), citations
