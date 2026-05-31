"""Extract structured entities from a MemoryDraft and persist them (Phase 4).

The entity vocabulary is small and derives directly from canonical slot keys:

    employment  → entity_type="employer"  name=company name from "Works at X"
    location    → entity_type="city"      name=city from "Lives in X"
    origin      → entity_type="city"      name=place from "Originally from X"
    pet.name    → entity_type="pet"       name=pet name from "Has a * named X"

``memory_entities`` links the memory to its entity with a relation label:
"employer_of", "city_of", "pet_of". This makes the entity traversal for
multi-hop decomposition simple: given an entity id, find all memories that
link to it.
"""
from __future__ import annotations

import re
import uuid

import asyncpg

from ..db import queries
from ..extraction.draft import MemoryDraft

# Patterns to extract the salient proper noun from a memory value string.
# Each returns the first group as the entity name.
_EMPLOYMENT_RE = re.compile(
    r"works?(?:\s+as\b.*?)?\s+at\s+([A-Za-z0-9][A-Za-z0-9 &.'-]*?)(?:\s+as\b|$)",
    re.IGNORECASE,
)
_LOCATION_RE = re.compile(
    r"lives?\s+in\s+([A-Za-z][A-Za-z .'-]*?)(?:\s*\(|$)",
    re.IGNORECASE,
)
_ORIGIN_RE = re.compile(
    r"originally\s+from\s+([A-Za-z][A-Za-z .'-]*?)(?:\s*[,.(]|$)",
    re.IGNORECASE,
)
_PET_RE = re.compile(
    r"(?:has\s+a\s+)?(?:pet|dog|cat|puppy|kitten|hamster|rabbit|parrot|bird)"
    r"(?:\s+is)?\s+named\s+([A-Za-z][A-Za-z'-]*)\b",
    re.IGNORECASE,
)


def _extract_name(value: str, pattern: re.Pattern) -> str | None:
    m = pattern.search(value)
    if not m:
        return None
    name = m.group(1).strip().strip(".,;:!?'\"")
    return name if name else None


def _entities_from_draft(draft: MemoryDraft) -> list[tuple[str, str, str]]:
    """Return (entity_type, name, relation) triples for the draft, or []."""
    key = draft.key or ""
    value = draft.value

    if key == "employment":
        name = _extract_name(value, _EMPLOYMENT_RE)
        if name:
            return [("employer", name, "employer_of")]

    if key == "location":
        name = _extract_name(value, _LOCATION_RE)
        if name:
            return [("city", name, "city_of")]

    if key == "origin":
        name = _extract_name(value, _ORIGIN_RE)
        if name:
            return [("city", name, "origin_city_of")]

    if key == "pet.name":
        name = _extract_name(value, _PET_RE)
        if name:
            return [("pet", name, "pet_of")]

    return []


async def populate_entities(
    conn: asyncpg.Connection,
    memory_id: uuid.UUID,
    draft: MemoryDraft,
    user_id: str,
) -> None:
    """Create entity records and memory→entity links for a newly-inserted memory.

    Runs inside the caller's open transaction so it's atomic with the insert.
    Idempotent: ``upsert_entity`` returns the existing id on conflict, and
    ``link_memory_entity`` is ON CONFLICT DO NOTHING.
    """
    for entity_type, name, relation in _entities_from_draft(draft):
        entity_id = await queries.upsert_entity(conn, user_id, entity_type, name)
        await queries.link_memory_entity(conn, memory_id, entity_id, relation)
