"""Naive search result formatting.

Returns the §3 ``/search`` shape: ``{content, score, session_id, timestamp,
metadata}`` per result (distinct from ``/recall``'s prose + citations).
"""
from __future__ import annotations


def _iso(value) -> str | None:
    if value is None:
        return None
    try:
        return value.isoformat()
    except AttributeError:
        return str(value)


def format_results(rows: list[dict]) -> list[dict]:
    return [
        {
            "content": row["value"],
            "score": round(float(row["score"]), 4),
            "session_id": row.get("source_session"),
            "timestamp": _iso(row.get("created_at")),
            "metadata": {},
        }
        for row in rows
    ]
