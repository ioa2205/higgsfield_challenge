"""HTTP endpoints (CHALLENGE.md §3). Exact shapes and status codes."""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Request, Response

from ..db import queries
from ..entities import populate_entities
from ..evolution import apply_supersession, is_unchanged_active_memory
from ..extraction import extract_memories
from ..logging_config import log_event
from ..recall import run_recall
from ..search import run_search
from .auth import require_auth
from .models import (
    HealthResponse,
    MemoriesResponse,
    RecallRequest,
    RecallResponse,
    SearchRequest,
    SearchResponse,
    TurnRequest,
    TurnResponse,
)

logger = logging.getLogger("memory.api")

router = APIRouter()
# Protected routes share this dependency; /health is deliberately excluded.
protected = APIRouter(dependencies=[Depends(require_auth)])


def _effective_user(user_id: Optional[str], session_id: Optional[str]) -> str:
    """§3 allows a null user_id. We keep the NOT NULL column honest by scoping
    anonymous turns to their session so they still round-trip without bleeding
    into real users."""
    if user_id:
        return user_id
    return f"anon:{session_id or 'unknown'}"


async def _embed(request: Request, text: str) -> list[float]:
    embedder = request.app.state.embedder
    return await asyncio.to_thread(embedder.embed_one, text)


def _iso(value) -> Optional[str]:
    if value is None:
        return None
    try:
        return value.isoformat()
    except AttributeError:
        return str(value)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@protected.post("/turns", response_model=TurnResponse, status_code=201)
async def post_turn(body: TurnRequest, request: Request) -> TurnResponse:
    pool = request.app.state.pool
    user_id = _effective_user(body.user_id, body.session_id)
    turn_id = uuid.uuid4()
    messages = [m.model_dump() for m in body.messages]

    drafts = extract_memories(messages)
    paths = sorted({d.provenance for d in drafts}) if drafts else []
    log_event(
        logger,
        "turn.ingest",
        turn_id=str(turn_id),
        user_id=user_id,
        session_id=body.session_id,
        n_messages=len(messages),
        n_memories=len(drafts),
        extraction=",".join(paths) or "none",
    )

    async with pool.acquire() as conn:
        async with conn.transaction():
            await queries.insert_turn(conn, turn_id, user_id, body.session_id)
            for pos, msg in enumerate(body.messages):
                await queries.insert_message(
                    conn,
                    turn_id,
                    user_id,
                    body.session_id,
                    pos,
                    msg.role,
                    msg.name,
                    msg.content,
                )
            stored = 0
            skipped = 0
            for draft in drafts:
                if draft.type != "event" and draft.key:
                    await queries.lock_memory_slot(conn, user_id, draft.key)
                if await is_unchanged_active_memory(conn, draft, user_id):
                    skipped += 1
                    log_event(
                        logger,
                        "memory.duplicate_suppressed",
                        turn_id=str(turn_id),
                        user_id=user_id,
                        key=draft.key,
                    )
                    continue

                # Avoid embedding unchanged facts. The slot lock and duplicate
                # check stay inside the transaction so concurrent repeats are
                # serialized before either row can be inserted.
                embedding = await _embed(request, draft.value)
                mem_id = await queries.insert_memory(
                    conn,
                    user_id=user_id,
                    mtype=draft.type,
                    key=draft.key,
                    value=draft.value,
                    confidence=draft.confidence,
                    source_session=body.session_id,
                    source_turn=turn_id,
                    embedding=embedding,
                    provenance=draft.provenance,
                )
                # Phase 4: supersede any conflicting active memory, then
                # register entity links for the new memory.
                await apply_supersession(conn, draft, mem_id, user_id, embedding)
                await populate_entities(conn, mem_id, draft, user_id)
                stored += 1

    log_event(
        logger,
        "turn.memories_persisted",
        turn_id=str(turn_id),
        user_id=user_id,
        stored=stored,
        duplicate_suppressed=skipped,
    )

    return TurnResponse(id=str(turn_id))


@protected.post("/recall", response_model=RecallResponse)
async def post_recall(body: RecallRequest, request: Request) -> RecallResponse:
    pool = request.app.state.pool
    user_id = _effective_user(body.user_id, body.session_id)
    embedding = await _embed(request, body.query)

    async with pool.acquire() as conn:
        context, citations = await run_recall(
            conn,
            user_id=user_id,
            session_id=body.session_id,
            query=body.query,
            embedding=embedding,
            max_tokens=body.max_tokens,
        )

    log_event(
        logger,
        "recall",
        user_id=user_id,
        session_id=body.session_id,
        max_tokens=body.max_tokens,
        n_hits=len(citations),
        empty=not context,
    )
    return RecallResponse(context=context, citations=citations)


@protected.post("/search", response_model=SearchResponse)
async def post_search(body: SearchRequest, request: Request) -> SearchResponse:
    pool = request.app.state.pool
    # /search honours explicit §3 scoping: filter by user_id and/or session_id
    # exactly as supplied (both nullable). No anon-key rewrite — a search with
    # neither scope is a global search, which the contract permits.
    embedding = await _embed(request, body.query)

    async with pool.acquire() as conn:
        results = await run_search(
            conn,
            user_id=body.user_id,
            session_id=body.session_id,
            query=body.query,
            embedding=embedding,
            limit=body.limit,
        )

    log_event(
        logger,
        "search",
        user_id=body.user_id,
        session_id=body.session_id,
        limit=body.limit,
        n_results=len(results),
    )
    return SearchResponse(results=results)


@protected.get("/users/{user_id}/memories", response_model=MemoriesResponse)
async def get_memories(user_id: str, request: Request) -> MemoriesResponse:
    pool = request.app.state.pool
    async with pool.acquire() as conn:
        rows = await queries.list_memories(conn, user_id)
    memories = [
        {
            "id": str(r["id"]),
            "type": r["type"],
            "key": r["key"],
            "value": r["value"],
            "confidence": round(float(r["confidence"]), 4),
            "source_session": r["source_session"],
            "source_turn": str(r["source_turn"]) if r["source_turn"] else None,
            "created_at": _iso(r["created_at"]),
            "updated_at": _iso(r["updated_at"]),
            "supersedes": str(r["supersedes"]) if r["supersedes"] else None,
            "active": r["active"],
            "provenance": r["provenance"],
        }
        for r in rows
    ]
    return MemoriesResponse(memories=memories)


@protected.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, request: Request) -> Response:
    pool = request.app.state.pool
    async with pool.acquire() as conn:
        await queries.delete_session(conn, session_id)
    log_event(logger, "delete.session", session_id=session_id)
    return Response(status_code=204)


@protected.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: str, request: Request) -> Response:
    pool = request.app.state.pool
    async with pool.acquire() as conn:
        await queries.delete_user(conn, user_id)
    log_event(logger, "delete.user", user_id=user_id)
    return Response(status_code=204)
