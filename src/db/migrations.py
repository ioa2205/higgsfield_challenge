"""Idempotent schema migrations, run on every startup.

This is the schema the *whole product* needs -- created now so later phases
(hybrid retrieval, supersession, entity/multi-hop) only add data, never run a
destructive migration. Everything is ``IF NOT EXISTS`` and safe to re-run.
"""
from __future__ import annotations

import asyncpg

SCHEMA = """
-- pgvector must exist before any vector column or the register_vector codec.
CREATE EXTENSION IF NOT EXISTS vector;

-- Raw conversation turns (provenance + re-extraction source).
CREATE TABLE IF NOT EXISTS turns (
    id          uuid PRIMARY KEY,
    user_id     text NOT NULL,
    session_id  text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_turns_user    ON turns(user_id);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);

-- Individual messages within a turn. `name` is nullable (tool messages).
CREATE TABLE IF NOT EXISTS messages (
    id          uuid PRIMARY KEY,
    turn_id     uuid NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    user_id     text NOT NULL,
    session_id  text NOT NULL,
    position    int  NOT NULL,
    role        text NOT NULL,
    name        text,
    content     text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_messages_turn    ON messages(turn_id);
CREATE INDEX IF NOT EXISTS idx_messages_user    ON messages(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

-- Durable, structured memories. Columns are FIXED now; later phases only
-- populate the currently-unused ones (supersedes, search, entity links).
CREATE TABLE IF NOT EXISTS memories (
    id             uuid PRIMARY KEY,
    user_id        text NOT NULL,
    type           text NOT NULL
                       CHECK (type IN ('fact', 'preference', 'opinion', 'event')),
    key            text,
    value          text NOT NULL,
    confidence     real NOT NULL DEFAULT 0.5,
    source_session text,
    source_turn    uuid REFERENCES turns(id) ON DELETE SET NULL,
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now(),
    supersedes     uuid REFERENCES memories(id) ON DELETE SET NULL,
    active         boolean NOT NULL DEFAULT true,
    embedding      vector(384),
    search         tsvector
);
-- Phase 2 (additive, idempotent): record WHICH extraction path produced a
-- memory ("llm:gemini", "rule", "rule:event"). The per-path half of §4
-- provenance; source_session/source_turn are the other half.
ALTER TABLE memories ADD COLUMN IF NOT EXISTS provenance text;
CREATE INDEX IF NOT EXISTS idx_memories_user    ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(source_session);
CREATE INDEX IF NOT EXISTS idx_memories_active  ON memories(user_id, active);
-- Full-text retrieval (phase 2) and vector retrieval (now).
CREATE INDEX IF NOT EXISTS idx_memories_search    ON memories USING GIN (search);
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories
    USING hnsw (embedding vector_cosine_ops);

-- Entities + links: columns fixed now, populated in a later phase.
CREATE TABLE IF NOT EXISTS entities (
    id          uuid PRIMARY KEY,
    user_id     text NOT NULL,
    entity_type text NOT NULL
                    CHECK (entity_type IN ('user', 'pet', 'employer', 'city')),
    name        text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_entities_user ON entities(user_id);

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id uuid NOT NULL REFERENCES memories(id)  ON DELETE CASCADE,
    entity_id uuid NOT NULL REFERENCES entities(id)  ON DELETE CASCADE,
    relation  text NOT NULL DEFAULT 'related',
    PRIMARY KEY (memory_id, entity_id, relation)
);

-- Phase 4: upsert_entity needs a unique constraint on (user_id, entity_type, name).
-- Use a DO block because ADD CONSTRAINT IF NOT EXISTS is Postgres 16.4+ only.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'uq_entity_user_type_name'
    ) THEN
        ALTER TABLE entities
            ADD CONSTRAINT uq_entity_user_type_name
            UNIQUE (user_id, entity_type, name);
    END IF;
END
$$;
"""


async def run_migrations(dsn: str) -> None:
    """Apply the schema on a fresh connection (before the pooled codec needs
    the `vector` type to exist)."""
    conn = await asyncpg.connect(dsn=dsn)
    try:
        await conn.execute(SCHEMA)
    finally:
        await conn.close()
