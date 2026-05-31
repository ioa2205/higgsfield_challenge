"""Recall orchestration: retrieve → fuse → assemble.

One entry point, ``run_recall``, wires the Phase-3 pipeline so ``api/routes`` stays
thin: gather candidate sets (Tier-1 facts, semantic top-N, keyword top-N, recent
session events), fuse the two retrievers with RRF, then assemble the tiered,
token-budgeted §3 context with the noise gate.
"""
from __future__ import annotations

import logging

import asyncpg

from .. import config
from ..logging_config import log_event
from .assembly import assemble
from .fusion import rrf_fuse
from .retrieval import gather_candidates

logger = logging.getLogger("memory.recall")


async def run_recall(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    session_id: str | None,
    query: str,
    embedding: list[float],
    max_tokens: int,
) -> tuple[str, list[dict]]:
    cands = await gather_candidates(
        conn,
        user_id=user_id,
        session_id=session_id,
        query=query,
        embedding=embedding,
    )
    fused = rrf_fuse(cands.semantic, cands.keyword, config.RRF_K)
    context, citations = assemble(
        cands.tier1,
        fused,
        cands.recent,
        max_tokens=max_tokens,
        entity_hop=cands.entity_hop,
    )
    log_event(
        logger,
        "recall.tiers",
        user_id=user_id,
        tier1=len(cands.tier1),
        semantic=len(cands.semantic),
        keyword=len(cands.keyword),
        recent=len(cands.recent),
        entity_hop=len(cands.entity_hop),
        emitted_citations=len(citations),
        empty=not context,
    )
    return context, citations
