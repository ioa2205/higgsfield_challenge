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
import re

from .. import config
from ..logging_config import log_event
from .draft import MEMORY_TYPES, MemoryDraft, clamp_confidence
from .llm import get_llm_extractors
from .rules import _user_text, rule_extract

logger = logging.getLogger("memory.extraction")

_VALUE_MAX = 500
_KEY_MAX = 128
_FACT_KEYS = {"employment", "location", "origin", "pet.name", "pet.species", "diet", "allergy"}
_KEY_ALIASES = {
    "company": "employment",
    "employer": "employment",
    "job": "employment",
    "occupation": "employment",
    "city": "location",
    "current_city": "location",
    "current_location": "location",
    "home_city": "location",
    "residence": "location",
    "former_city": "origin",
    "previous_city": "origin",
    "previous_location": "origin",
}


def _canonical_key(key: str | None) -> str | None:
    """Normalize common LLM slot aliases into the product vocabulary."""
    if not key:
        return None
    key = key.replace("\x00", "").strip().lower()[:_KEY_MAX]
    key = re.sub(r"\s+", "_", key)
    key = _KEY_ALIASES.get(key, key)
    if re.fullmatch(r"pets?(?:\.[^.]+)?\.name", key):
        return "pet.name"
    if re.fullmatch(r"pets?(?:\.[^.]+)?\.(?:species|type)", key):
        return "pet.species"
    return key or None


def _canonical_value(key: str | None, value: str) -> str:
    """Keep LLM values readable and compatible with entity population."""
    if key == "location" and not re.search(r"\b(?:lives?|moved|located|based)\b", value, re.I):
        return f"Lives in {value}"
    if key == "origin" and not re.search(r"\b(?:from|previously|formerly|origin)\b", value, re.I):
        return f"Originally from {value}"
    if key == "allergy" and not re.search(r"\ballerg", value, re.I):
        return f"Allergic to {value}"
    if key == "pet.name" and not re.search(r"\bnamed\b", value, re.I):
        return f"Has a pet named {value}"
    return value


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
        raw_key = item.get("key")
        key = _canonical_key(str(raw_key) if raw_key is not None else None)
        value = _canonical_value(key, value)[:_VALUE_MAX]
        if key in _FACT_KEYS or (key and key.startswith("family.")):
            mtype = "fact"
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


def _sanitize(drafts: list[MemoryDraft]) -> list[MemoryDraft]:
    """Bound model/rule output before embedding or writing it to Postgres."""
    out: list[MemoryDraft] = []
    for draft in drafts:
        if draft.type not in MEMORY_TYPES:
            continue
        value = draft.value.replace("\x00", "").strip()[:_VALUE_MAX]
        key = _canonical_key(draft.key)
        value = _canonical_value(key, value)[:_VALUE_MAX]
        if value:
            out.append(
                MemoryDraft(
                    draft.type,
                    key or None,
                    value,
                    clamp_confidence(draft.confidence),
                    draft.provenance,
                )
            )
    return out


def extract(messages: list[dict]) -> list[MemoryDraft]:
    """Return typed memory drafts for one turn (synchronous, never raises)."""
    drafts: list[MemoryDraft] = []
    path = "rule"

    extractors = get_llm_extractors()
    for extractor in extractors:
        log_event(
            logger,
            "extraction.provider_attempt",
            provider=extractor.provider,
            model=getattr(extractor, "model", None),
            timeout=getattr(extractor, "timeout", None),
        )
        items = extractor.extract(messages)  # None on error/timeout
        if items:
            drafts = _normalize_llm(items, provenance=f"llm:{extractor.provider}")
            if drafts:
                path = f"llm:{extractor.provider}"
                log_event(
                    logger,
                    "extraction.provider_success",
                    provider=extractor.provider,
                    n_memories=len(drafts),
                )
                break
        log_event(
            logger,
            "extraction.provider_failed",
            provider=extractor.provider,
            reason="failed_empty_or_unusable_output",
        )

    if not extractors:
        reason = (
            "llm_disabled"
            if config.LLM_PROVIDER in ("none", "off", "disabled", "fake")
            else "api_key_missing"
        )
        log_event(logger, "extraction.degraded", reason=reason)
    elif not drafts:
        log_event(
            logger,
            "extraction.degraded",
            reason="all_configured_providers_failed",
            providers=[extractor.provider for extractor in extractors],
        )

    if not drafts:  # no LLM, LLM failed, or LLM returned nothing usable
        drafts = rule_extract(messages)
        path = "rule"

    if not drafts:  # nothing typed matched — keep the turn recallable
        drafts = _event_fallback(messages)
        path = "rule:event"

    drafts = _sanitize(drafts)
    log_event(logger, "extraction.path", path=path, n_memories=len(drafts))
    return drafts
