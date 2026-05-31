"""Pydantic request/response models mirroring CHALLENGE.md §3 exactly."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --- Requests ---------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str
    # Nullable; present on tool-role messages, absent/None otherwise.
    name: Optional[str] = None


class TurnRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    messages: List[Message] = Field(min_length=1)
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RecallRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    max_tokens: int = 1024


class SearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    limit: int = 10


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
