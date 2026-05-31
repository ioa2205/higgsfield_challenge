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
