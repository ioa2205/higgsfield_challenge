"""Hybrid extraction orchestration (synchronous).

One entry point, ``extract(messages)``, used by ``POST /turns`` before it
returns 201 — so every memory is persisted, embedded and indexed synchronously
(§5 "if you wrote it, you can read it"). Policy:

  1. If an LLM provider+key is configured, try it (bounded by ``LLM_TIMEOUT``,
     well inside the 60s /turns budget). On a usable, non-empty result, USE it —
     normal LLM latency does not trigger the fallback.
  2. Otherwise — no key, an error/timeout, or an empty/garbage result — use the
     deterministic rule path.
  3. If neither yields a typed memory, emit ONE ``event`` memory summarising the
     user's utterance, so an arbitrary turn is still recallable. This is the
     only place a near-raw value is stored, and it is typed ``event`` (never a
     fake "fact") — not a raw chunk masquerading as structured knowledge.

Both paths normalise into ``MemoryDraft`` (same schema the DB stores), tagged
with provenance so a reviewer can see which path produced each memory.
"""
from __future__ import annotations

import logging

from .draft import MEMORY_TYPES, MemoryDraft, clamp_confidence
from .llm import get_llm_extractor
from .rules import _user_text, rule_extract

logger = logging.getLogger("memory.extraction")

_VALUE_MAX = 500


def _normalize_llm(items: list[dict], provenance: str) -> list[MemoryDraft]:
    """Coerce raw LLM dicts into valid MemoryDrafts, dropping unusable ones."""
    out: list[MemoryDraft] = []
    for item in items:
        mtype = str(item.get("type", "")).strip().lower()
        if mtype not in MEMORY_TYPES:
            continue
        value = str(item.get("value", "")).strip()[:_VALUE_MAX]
        if not value:
            continue
        key = item.get("key")
        key = str(key).strip() or None if key is not None else None
        out.append(
            MemoryDraft(
                type=mtype,
                key=key,
                value=value,
                confidence=clamp_confidence(item.get("confidence")),
                provenance=provenance,
            )
        )
    return out


def _event_fallback(messages: list[dict]) -> list[MemoryDraft]:
    text = _user_text(messages)
    if not text:
        # No user content at all (e.g. tool-only turn): summarise everything so
        # the turn is still queryable.
        text = "  ".join(
            (m.get("content") or "").strip()
            for m in messages
            if (m.get("content") or "").strip()
        )
    if not text:
        return []
    return [MemoryDraft("event", None, text[:_VALUE_MAX], 0.4, "rule:event")]


def extract(messages: list[dict]) -> list[MemoryDraft]:
    """Return typed memory drafts for one turn (synchronous, never raises)."""
    drafts: list[MemoryDraft] = []
    path = "rule"

    extractor = get_llm_extractor()
    if extractor is not None:
        items = extractor.extract(messages)  # None on error/timeout
        if items:
            drafts = _normalize_llm(items, provenance=f"llm:{extractor.provider}")
            path = f"llm:{extractor.provider}"

    if not drafts:  # no LLM, LLM failed, or LLM returned nothing usable
        drafts = rule_extract(messages)
        path = "rule"

    if not drafts:  # nothing typed matched — keep the turn recallable
        drafts = _event_fallback(messages)
        path = "rule:event"

    logger.debug("extracted %d memories via %s", len(drafts), path)
    return drafts
