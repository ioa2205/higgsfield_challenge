"""All SQL the service issues, kept in one place.

Recall/search filter by ``user_id`` only (NOT session): a user's durable facts
are shared across all of that user's sessions, while different users never
bleed. Session scoping applies only to "recent conversation" context, which is
not part of the naive recall this phase.
"""
from __future__ import annotations

import uuid

import asyncpg


async def insert_turn(
    conn: asyncpg.Connection, turn_id: uuid.UUID, user_id: str, session_id: str
) -> None:
    await conn.execute(
        "INSERT INTO turns (id, user_id, session_id) VALUES ($1, $2, $3)",
        turn_id,
        user_id,
        session_id,
    )


async def insert_message(
    conn: asyncpg.Connection,
    turn_id: uuid.UUID,
    user_id: str,
    session_id: str,
    position: int,
    role: str,
    name: str | None,
    content: str,
) -> None:
    await conn.execute(
        """
        INSERT INTO messages
            (id, turn_id, user_id, session_id, position, role, name, content)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        uuid.uuid4(),
        turn_id,
        user_id,
        session_id,
        position,
        role,
        name,
        content,
    )


async def insert_memory(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    mtype: str,
    key: str | None,
    value: str,
    confidence: float,
    source_session: str,
    source_turn: uuid.UUID,
    embedding: list[float],
    provenance: str | None = None,
) -> uuid.UUID:
    """Persist one typed memory with its vector AND its full-text `search`
    tsvector (key + value) so keyword recall (a later phase) needs no backfill.
    """
    mem_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO memories
            (id, user_id, type, key, value, confidence,
             source_session, source_turn, embedding, provenance, search)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                to_tsvector('english', coalesce($4, '') || ' ' || $5))
        """,
        mem_id,
        user_id,
        mtype,
        key,
        value,
        confidence,
        source_session,
        source_turn,
        embedding,
        provenance,
    )
    return mem_id


async def recall_by_vector(
    conn: asyncpg.Connection, user_id: str, embedding: list[float], top_k: int
) -> list[asyncpg.Record]:
    """Top-k active memories for a user by cosine similarity (shared across the
    user's sessions)."""
    return await conn.fetch(
        """
        SELECT id, type, key, value, confidence, active,
               source_session, source_turn, created_at,
               1 - (embedding <=> $2) AS score
        FROM memories
        WHERE user_id = $1 AND active AND embedding IS NOT NULL
        ORDER BY embedding <=> $2
        LIMIT $3
        """,
        user_id,
        embedding,
        top_k,
    )


# --- Phase 3: hybrid retrieval -------------------------------------------- #
# An OR-tsquery built from the raw query: websearch_to_tsquery AND-joins terms
# (so a memory value almost never matches every term), which is wrong for
# recall. We OR-join the lexemes instead so ANY salient term can match, and rank
# by ts_rank. The replace turns '&'/'<->' into '|'; an all-stopword query yields
# '' which `@@` treats as "no match" (so a noise query keyword-retrieves nothing).
_ORQUERY = "regexp_replace(websearch_to_tsquery('english', $2)::text, '<->|&', '|', 'g')::tsquery"


async def semantic_recall(
    conn: asyncpg.Connection, user_id: str, embedding: list[float], top_n: int
) -> list[asyncpg.Record]:
    """Top-N active memories for the user by vector cosine (cross-session)."""
    return await conn.fetch(
        """
        SELECT id, type, key, value, confidence, active,
               source_session, source_turn, created_at,
               1 - (embedding <=> $2) AS score
        FROM memories
        WHERE user_id = $1 AND active AND embedding IS NOT NULL
        ORDER BY embedding <=> $2
        LIMIT $3
        """,
        user_id,
        embedding,
        top_n,
    )


async def keyword_recall(
    conn: asyncpg.Connection, user_id: str, query: str, top_n: int
) -> list[asyncpg.Record]:
    """Top-N active memories for the user by full-text ts_rank (cross-session).

    Uses the OR-tsquery so any salient term matches; an all-stopword/empty query
    returns nothing (the keyword half of the noise gate)."""
    return await conn.fetch(
        f"""
        SELECT id, type, key, value, confidence, active,
               source_session, source_turn, created_at,
               ts_rank(search, {_ORQUERY}) AS score
        FROM memories
        WHERE user_id = $1 AND active AND search IS NOT NULL
              AND search @@ {_ORQUERY}
        ORDER BY score DESC
        LIMIT $3
        """,
        user_id,
        query,
        top_n,
    )


async def tier1_facts(conn: asyncpg.Connection, user_id: str) -> list[asyncpg.Record]:
    """All ACTIVE stable facts/preferences/opinions for the user (cross-session),
    highest confidence first — the "Known facts about this user" digest.

    LEFT JOINs the immediately-superseded predecessor (``supersedes``) so the
    assembler can render an "updated …; previously …" annotation. Nothing is
    superseded until Phase 4, so ``prev_value`` is null today, but the path
    exists now (CLAUDE.md: avoid a Phase-3↔4 "read only active" vs "show it
    evolved" conflict)."""
    return await conn.fetch(
        """
        SELECT m.id, m.type, m.key, m.value, m.confidence, m.active,
               m.source_session, m.source_turn, m.created_at, m.updated_at,
               p.value AS prev_value, p.created_at AS prev_created_at
        FROM memories m
        LEFT JOIN memories p ON p.id = m.supersedes
        WHERE m.user_id = $1 AND m.active
              AND m.type IN ('fact', 'preference', 'opinion')
        ORDER BY m.confidence DESC, m.updated_at DESC
        """,
        user_id,
    )


async def recent_session_events(
    conn: asyncpg.Connection, session_id: str | None, top_n: int
) -> list[asyncpg.Record]:
    """Most-recent ACTIVE ``event`` memories for THIS session (session-scoped —
    the Tier-3 'recent conversation' tier). Empty when session_id is null."""
    if not session_id:
        return []
    return await conn.fetch(
        """
        SELECT id, type, key, value, confidence, active,
               source_session, source_turn, created_at,
               0.0::float8 AS score
        FROM memories
        WHERE source_session = $1 AND active AND type = 'event'
        ORDER BY created_at DESC
        LIMIT $2
        """,
        session_id,
        top_n,
    )


async def search_hybrid(
    conn: asyncpg.Connection,
    user_id: str | None,
    session_id: str | None,
    embedding: list[float],
    query: str,
    limit: int,
) -> list[asyncpg.Record]:
    """Structured /search: active memories scored by the better of full-text
    ts_rank and vector cosine, scoped by user_id and/or session_id. The hybrid
    score keeps purely-keyword and purely-semantic matches both retrievable.

    Fixed params: $1=embedding, $2=query, $3=limit. Optional scope params start
    at $4 (user_id) then $5 (session_id)."""
    orq = "regexp_replace(websearch_to_tsquery('english', $2)::text, '<->|&', '|', 'g')::tsquery"
    where = ["active"]
    params: list = [embedding, query, limit]
    if user_id:
        params.append(user_id)
        where.append(f"user_id = ${len(params)}")
    if session_id:
        params.append(session_id)
        where.append(f"source_session = ${len(params)}")
    clause = " AND ".join(where)
    return await conn.fetch(
        f"""
        SELECT id, type, key, value, confidence, active,
               source_session, source_turn, created_at,
               GREATEST(
                   COALESCE(ts_rank(search, {orq}), 0),
                   CASE WHEN embedding IS NULL THEN 0
                        ELSE (1 - (embedding <=> $1)) END
               ) AS score
        FROM memories
        WHERE {clause}
        ORDER BY score DESC
        LIMIT $3
        """,
        *params,
    )


async def list_memories(
    conn: asyncpg.Connection, user_id: str
) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT id, type, key, value, confidence, active,
               source_session, source_turn, created_at, updated_at,
               supersedes, provenance
        FROM memories
        WHERE user_id = $1
        ORDER BY created_at
        """,
        user_id,
    )


async def delete_session(conn: asyncpg.Connection, session_id: str) -> None:
    """Remove a single session's data only. Memories derived from this session
    go too (privacy); other sessions' memories remain recallable."""
    async with conn.transaction():
        await conn.execute(
            "DELETE FROM memories WHERE source_session = $1", session_id
        )
        # messages cascade via turns.
        await conn.execute("DELETE FROM turns WHERE session_id = $1", session_id)
        await conn.execute("DELETE FROM messages WHERE session_id = $1", session_id)


async def delete_user(conn: asyncpg.Connection, user_id: str) -> None:
    """Cascade-delete everything for a user (memories + turns + sessions +
    entities). memory_entities cascades via memories/entities."""
    async with conn.transaction():
        await conn.execute("DELETE FROM memories WHERE user_id = $1", user_id)
        await conn.execute("DELETE FROM entities WHERE user_id = $1", user_id)
        await conn.execute("DELETE FROM turns WHERE user_id = $1", user_id)
        await conn.execute("DELETE FROM messages WHERE user_id = $1", user_id)


# --- Phase 4: supersession -------------------------------------------------- #


async def find_active_by_key(
    conn: asyncpg.Connection,
    user_id: str,
    key: str,
    exclude_id: uuid.UUID | None = None,
) -> list[asyncpg.Record]:
    """Return all ACTIVE memories with exactly this slot key for the user,
    optionally excluding a specific memory id (e.g. the just-inserted one)."""
    if exclude_id is not None:
        return await conn.fetch(
            """
            SELECT id, type, key, value, confidence, embedding, created_at, updated_at
            FROM memories
            WHERE user_id = $1 AND key = $2 AND active AND id != $3
            ORDER BY updated_at DESC
            """,
            user_id,
            key,
            exclude_id,
        )
    return await conn.fetch(
        """
        SELECT id, type, key, value, confidence, embedding, created_at, updated_at
        FROM memories
        WHERE user_id = $1 AND key = $2 AND active
        ORDER BY updated_at DESC
        """,
        user_id,
        key,
    )


async def find_active_by_type_and_embedding(
    conn: asyncpg.Connection,
    user_id: str,
    mtype: str,
    embedding: list[float],
    threshold: float,
    exclude_id: uuid.UUID | None = None,
    top_n: int = 3,
) -> list[asyncpg.Record]:
    """Return ACTIVE memories of the same type whose embedding cosine similarity
    to the new memory exceeds ``threshold``. Used as a fuzzy supersession fallback
    when the slot keys don't match exactly. ``exclude_id`` is the just-inserted
    memory so it cannot self-supersede."""
    if exclude_id is not None:
        return await conn.fetch(
            """
            SELECT id, type, key, value, confidence, embedding, created_at, updated_at,
                   1 - (embedding <=> $3) AS sim
            FROM memories
            WHERE user_id = $1 AND type = $2 AND active AND embedding IS NOT NULL
                  AND id != $6
                  AND 1 - (embedding <=> $3) >= $4
            ORDER BY sim DESC
            LIMIT $5
            """,
            user_id,
            mtype,
            embedding,
            threshold,
            top_n,
            exclude_id,
        )
    return await conn.fetch(
        """
        SELECT id, type, key, value, confidence, embedding, created_at, updated_at,
               1 - (embedding <=> $3) AS sim
        FROM memories
        WHERE user_id = $1 AND type = $2 AND active AND embedding IS NOT NULL
              AND 1 - (embedding <=> $3) >= $4
        ORDER BY sim DESC
        LIMIT $5
        """,
        user_id,
        mtype,
        embedding,
        threshold,
        top_n,
    )


async def supersede_memory(
    conn: asyncpg.Connection, old_id: uuid.UUID, new_id: uuid.UUID
) -> None:
    """Mark ``old_id`` as superseded by ``new_id``.

    Sets old.active=false; sets new.supersedes=old_id and advances
    new.updated_at to now(). Both writes happen inside the caller's transaction.
    """
    await conn.execute(
        "UPDATE memories SET active = false, updated_at = now() WHERE id = $1",
        old_id,
    )
    await conn.execute(
        "UPDATE memories SET supersedes = $1, updated_at = now() WHERE id = $2",
        old_id,
        new_id,
    )


# --- Phase 4: entity layer -------------------------------------------------- #


async def upsert_entity(
    conn: asyncpg.Connection, user_id: str, entity_type: str, name: str
) -> uuid.UUID:
    """Return the id of the entity with (user_id, entity_type, name), creating
    it if it does not exist. Using a unique constraint so concurrent inserts are
    safe; the SELECT after the DO NOTHING path returns the existing row."""
    eid = uuid.uuid4()
    row = await conn.fetchrow(
        """
        INSERT INTO entities (id, user_id, entity_type, name)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (user_id, entity_type, name) DO NOTHING
        RETURNING id
        """,
        eid,
        user_id,
        entity_type,
        name,
    )
    if row:
        return row["id"]
    # Conflict: fetch the existing id.
    existing = await conn.fetchrow(
        "SELECT id FROM entities WHERE user_id = $1 AND entity_type = $2 AND name = $3",
        user_id,
        entity_type,
        name,
    )
    return existing["id"]


async def link_memory_entity(
    conn: asyncpg.Connection, memory_id: uuid.UUID, entity_id: uuid.UUID, relation: str
) -> None:
    """Create a memory→entity link (idempotent via ON CONFLICT DO NOTHING)."""
    await conn.execute(
        """
        INSERT INTO memory_entities (memory_id, entity_id, relation)
        VALUES ($1, $2, $3)
        ON CONFLICT DO NOTHING
        """,
        memory_id,
        entity_id,
        relation,
    )


async def get_all_entities_for_user(
    conn: asyncpg.Connection, user_id: str
) -> list[asyncpg.Record]:
    """Return every entity recorded for the user (all types)."""
    return await conn.fetch(
        "SELECT id, entity_type, name FROM entities WHERE user_id = $1",
        user_id,
    )


async def find_active_facts_via_entities(
    conn: asyncpg.Connection, user_id: str, entity_ids: list[uuid.UUID]
) -> list[asyncpg.Record]:
    """Return ACTIVE fact/preference/opinion memories for the user that are
    linked to ANY of the given entity ids.  Used by multi-hop decomposition to
    surface connected knowledge that query-term recall would miss."""
    if not entity_ids:
        return []
    placeholders = ", ".join(f"${i + 2}" for i in range(len(entity_ids)))
    return await conn.fetch(
        f"""
        SELECT DISTINCT m.id, m.type, m.key, m.value, m.confidence, m.active,
               m.source_session, m.source_turn, m.created_at, m.updated_at,
               0.0::float8 AS score
        FROM memories m
        JOIN memory_entities me ON me.memory_id = m.id
        WHERE m.user_id = $1 AND m.active AND m.type IN ('fact', 'preference', 'opinion')
              AND me.entity_id IN ({placeholders})
        """,
        user_id,
        *entity_ids,
    )


async def find_all_active_facts(
    conn: asyncpg.Connection, user_id: str
) -> list[asyncpg.Record]:
    """Return ALL active fact/preference/opinion memories for the user. Used as
    the expanded candidate pool when multi-hop entity matching fires."""
    return await conn.fetch(
        """
        SELECT id, type, key, value, confidence, active,
               source_session, source_turn, created_at, updated_at,
               0.0::float8 AS score
        FROM memories
        WHERE user_id = $1 AND active AND type IN ('fact', 'preference', 'opinion')
        ORDER BY confidence DESC, updated_at DESC
        """,
        user_id,
    )
