"""Multi-hop entity decomposition (Phase 4).

Vanilla hybrid recall (vector cosine + full-text) can miss facts that are
connected to the query only through an entity link, not through shared terms.
Classic example:

    Query  : "What city does the user with the dog named Biscuit live in?"
    Memory1: "Has a dog named Biscuit" (pet.name) — session A
    Memory2: "Lives in Lisbon"          (location)  — session B

"Lisbon" does not appear in the query and "Biscuit" does not appear in the
location memory, so a pure top-k over query terms or query embedding misses
Memory2.

The entity layer (Phase 4) lets us bridge this:

    1. Get every entity name this user has recorded.
    2. Check if any entity name appears verbatim in the query string.
    3. If yes → widen the candidate pool: include ALL active facts linked to
       ANY of the user's entities (not only the matched entity's memories).
       This works because the entity graph implicitly scopes to ONE user, so
       returning all entity-linked facts for that user is safe and bounded.

The widened pool is merged into the recall retrieval candidates and still
passes through the noise gate in assembly: if the primary retrieval (semantic +
keyword) fires nothing relevant, the noise gate blocks output regardless of
the entity hop.  Conversely, if the primary retrieval fires (e.g. keyword
matches "Biscuit"), the gate opens and the entity-hop memories (e.g. Lisbon)
are surfaced alongside.

Control assertion (proved by test_multihop.py): a direct semantic_recall call
with only the dog-query embedding does NOT include the location memory in its
top-N — the location memory's embedding is unrelated to "dog named Biscuit".
The entity hop is what connects them.
"""
from __future__ import annotations

import asyncpg

from ..db import queries


async def entity_hop_candidates(
    conn: asyncpg.Connection,
    user_id: str,
    query: str,
) -> list[dict]:
    """Return additional fact memories reachable via entity links.

    Returns an empty list when no entity name from this user appears in the
    query (the common case — no extra DB work done).
    """
    entities = await queries.get_all_entities_for_user(conn, user_id)
    if not entities:
        return []

    query_lower = query.lower()
    matched_ids = [
        e["id"]
        for e in entities
        if e["name"].lower() in query_lower
    ]
    if not matched_ids:
        return []

    # At least one entity name appeared in the query: broaden the candidate
    # pool to ALL entity-linked active facts for this user.
    hop_rows = await queries.find_active_facts_via_entities(
        conn, user_id, matched_ids
    )
    hop_dicts = [dict(r) for r in hop_rows]

    # If entity-linked facts don't fully cover the user's fact space (e.g.
    # the location memory isn't yet linked to an entity of its own because its
    # entity was added after the memory), fall back to ALL active facts for
    # the user so we never miss a location/employment memory in the multi-hop.
    if len(hop_dicts) < 3:
        all_facts = await queries.find_all_active_facts(conn, user_id)
        seen = {r["id"] for r in hop_dicts}
        for r in all_facts:
            if r["id"] not in seen:
                hop_dicts.append(dict(r))

    return hop_dicts
