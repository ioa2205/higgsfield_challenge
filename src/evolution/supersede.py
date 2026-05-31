"""Slot-based fact supersession logic (Phase 4).

Called synchronously inside the /turns transaction so the effect is visible to
/recall and /users/{id}/memories the moment /turns returns 201.
"""
from __future__ import annotations

import logging
import re
import unicodedata
import uuid

import asyncpg

from .. import config
from ..db import queries
from ..extraction.draft import MemoryDraft
from ..logging_config import log_event

logger = logging.getLogger("memory.evolution")


def normalized_memory_value(key: str | None, value: str) -> str:
    """Normalize harmless presentation differences without erasing meaning."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    if key == "pet.name":
        match = re.search(r"\bnamed\s+([a-z][a-z'-]*)\b", normalized)
        if match:
            return f"named {match.group(1)}"
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


async def is_unchanged_active_memory(
    conn: asyncpg.Connection,
    draft: MemoryDraft,
    user_id: str,
) -> bool:
    """Return True when an equivalent active non-event slot already exists."""
    if draft.type == "event" or not draft.key:
        return False
    expected = normalized_memory_value(draft.key, draft.value)
    rows = await queries.find_active_by_key(conn, user_id, draft.key)
    return any(
        normalized_memory_value(draft.key, row["value"]) == expected
        for row in rows
    )


async def apply_supersession(
    conn: asyncpg.Connection,
    draft: MemoryDraft,
    new_memory_id: uuid.UUID,
    user_id: str,
    embedding: list[float],
) -> None:
    """Find and supersede any conflicting active memory for this user.

    Runs INSIDE the caller's transaction so the reads and the UPDATE are atomic.

    Strategy
    --------
    1. If the draft has a canonical key, look for any active memory with the
       SAME key.  If found, supersede the most-recently-updated one.
    2. If no exact-key conflict is found (or the draft has no key), apply the
       fuzzy path: cosine similarity ≥ SUPERSESSION_SIM_THRESHOLD on same type.
       This catches LLM-produced variant keys and generic slots.
    3. ``event`` memories are NEVER superseded — they are session-scoped
       ephemeral records, not durable slot-typed facts.
    """
    if draft.type == "event":
        return  # events are ephemeral; no supersession

    # A user can have multiple pets. Keep pet names append-only as a small
    # set-valued exception; the pre-insert duplicate check still suppresses
    # repeated mentions of the same pet.
    if draft.key == "pet.name":
        return

    old_id: uuid.UUID | None = None

    # --- path 1: exact slot-key match ---
    if draft.key:
        # Exclude the new memory so it doesn't self-supersede (both have
        # active=true + same key at the moment of the query).
        rows = await queries.find_active_by_key(
            conn, user_id, draft.key, exclude_id=new_memory_id
        )
        if rows:
            # Supersede the most-recently-updated conflicting active memory.
            old_id = rows[0]["id"]

    # --- path 2: fuzzy embedding match (safety net) ---
    if old_id is None:
        candidates = await queries.find_active_by_type_and_embedding(
            conn,
            user_id,
            draft.type,
            embedding,
            config.SUPERSESSION_SIM_THRESHOLD,
            exclude_id=new_memory_id,
        )
        if candidates:
            # The top hit exceeded the threshold; treat it as the same slot.
            top = candidates[0]
            # Extra guard: don't fuse two opinions with DIFFERENT subjects if
            # both happened to score above threshold (rare but possible).
            if draft.key and top["key"] and draft.key != top["key"]:
                logger.debug(
                    "fuzzy supersession skipped: key mismatch %r vs %r (sim %.3f)",
                    draft.key,
                    top["key"],
                    float(top["sim"]),
                )
            else:
                old_id = top["id"]
                logger.debug(
                    "fuzzy supersession: sim=%.3f, old key=%r new key=%r",
                    float(top["sim"]),
                    top["key"],
                    draft.key,
                )

    if old_id is None:
        return  # nothing to supersede

    if old_id == new_memory_id:
        return  # safety: never self-supersede

    await queries.supersede_memory(conn, old_id, new_memory_id)
    log_event(
        logger,
        "memory.superseded",
        old_id=str(old_id),
        new_id=str(new_memory_id),
        key=draft.key,
        user_id=user_id,
    )
