"""Structured /search (CHALLENGE.md §3) — distinct from /recall's prose.

``/search`` is the explicit, agent-invoked lookup: it returns ranked, STRUCTURED
rows (``content, score, session_id, timestamp, metadata``), not an assembled
context block. Retrieval is hybrid (the better of full-text ts_rank and vector
cosine), scoped by ``user_id`` and/or ``session_id`` and capped at ``limit``.
Unlike /recall there is no tiering, no prose, and no noise gate — a search tool
returns its best candidates and lets the caller decide.
"""
from __future__ import annotations

import asyncpg

from ..db import queries


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
            "metadata": {"type": row.get("type"), "key": row.get("key")},
        }
        for row in rows
    ]


async def run_search(
    conn: asyncpg.Connection,
    *,
    user_id: str | None,
    session_id: str | None,
    query: str,
    embedding: list[float],
    limit: int,
) -> list[dict]:
    rows = await queries.search_hybrid(
        conn, user_id, session_id, embedding, query, limit
    )
    return format_results([dict(r) for r in rows])
