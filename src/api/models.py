"""Pydantic request/response models mirroring CHALLENGE.md §3 exactly."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .. import config


def _reject_nul(value: str | None) -> str | None:
    """Postgres rejects NUL bytes in text; reject them as client input."""
    if value is not None and "\x00" in value:
        raise ValueError("NUL bytes are not allowed")
    return value


# --- Requests ---------------------------------------------------------------
class Message(BaseModel):
    role: str = Field(min_length=1, max_length=32)
    content: str = Field(max_length=config.MAX_MESSAGE_CONTENT_CHARS)
    # Nullable; present on tool-role messages, absent/None otherwise.
    name: Optional[str] = Field(default=None, max_length=256)

    _no_nul = field_validator("role", "content", "name")(_reject_nul)


class TurnRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=512)
    user_id: Optional[str] = Field(default=None, max_length=512)
    messages: List[Message] = Field(
        min_length=1, max_length=config.MAX_MESSAGES_PER_TURN
    )
    timestamp: Optional[str] = Field(default=None, max_length=128)
    metadata: Optional[Dict[str, Any]] = None

    _no_nul = field_validator("session_id", "user_id", "timestamp")(_reject_nul)


class RecallRequest(BaseModel):
    query: str = Field(min_length=1, max_length=config.MAX_QUERY_CHARS)
    session_id: str = Field(min_length=1, max_length=512)
    user_id: Optional[str] = Field(default=None, max_length=512)
    max_tokens: int = Field(default=1024, ge=1, le=32768)

    _no_nul = field_validator("query", "session_id", "user_id")(_reject_nul)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=config.MAX_QUERY_CHARS)
    session_id: Optional[str] = Field(default=None, max_length=512)
    user_id: Optional[str] = Field(default=None, max_length=512)
    limit: int = Field(default=10, ge=1, le=100)

    _no_nul = field_validator("query", "session_id", "user_id")(_reject_nul)


# --- Responses --------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str


class TurnResponse(BaseModel):
    id: str


class Citation(BaseModel):
    turn_id: str
    score: float
    snippet: str


class RecallResponse(BaseModel):
    context: str
    citations: List[Citation]


class SearchResultItem(BaseModel):
    content: str
    score: float
    session_id: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: List[SearchResultItem]


class MemoryRecord(BaseModel):
    """Inspection shape per §3 reference (schema is "up to you"; we mirror it)."""

    id: str
    type: str
    key: Optional[str] = None
    value: str
    confidence: float
    source_session: Optional[str] = None
    source_turn: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    supersedes: Optional[str] = None
    active: bool
    # Which extraction path produced this memory ("llm:gemini", "rule", …).
    provenance: Optional[str] = None


class MemoriesResponse(BaseModel):
    memories: List[MemoryRecord]
