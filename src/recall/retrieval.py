"""Retrieval: pull the candidate sets recall needs from Postgres.

Three roles, one store (CLAUDE.md): the SAME ``memories`` table answers the
semantic query (pgvector cosine), the keyword query (full-text ts_rank), the
Tier-1 stable-facts digest, and the Tier-3 recent-session events. This module is
the thin async layer that issues those reads (via ``db.queries``) and hands the
rows to fusion + assembly. Tier-1/semantic/keyword are CROSS-SESSION for the
user; the recent tier is SESSION-scoped (the Phase-1 scoping rule).
"""
from __future__ import annotations

from dataclasses import dataclass

import asyncpg

from .. import config
from ..db import queries
from .decompose import entity_hop_candidates


@dataclass
class Candidates:
    tier1: list[dict]      # active stable facts (digest), confidence-ordered
    semantic: list[dict]   # vector top-N (cosine in `score`)
    keyword: list[dict]    # full-text top-N (ts_rank in `score`)
    recent: list[dict]     # session-scoped recent events
    entity_hop: list[dict] # Phase-4 multi-hop: facts reachable via entity links


async def gather_candidates(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    session_id: str | None,
    query: str,
    embedding: list[float],
) -> Candidates:
    """Fetch every candidate set for one /recall. Sequential on the single pooled
    connection (asyncpg forbids concurrent queries on one connection)."""
    tier1 = await queries.tier1_facts(conn, user_id)
    semantic = await queries.semantic_recall(conn, user_id, embedding, config.SEM_TOP_N)
    keyword = await queries.keyword_recall(conn, user_id, query, config.KW_TOP_N)
    recent = await queries.recent_session_events(conn, session_id, config.TIER3_RECENT_N)
    # Phase 4: entity-hop for multi-hop queries (no-op when no entity matches).
    hop = await entity_hop_candidates(conn, user_id, query)
    return Candidates(
        tier1=[dict(r) for r in tier1],
        semantic=[dict(r) for r in semantic],
        keyword=[dict(r) for r in keyword],
        recent=[dict(r) for r in recent],
        entity_hop=hop,
    )
