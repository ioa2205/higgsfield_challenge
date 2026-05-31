"""Provider-agnostic LLM extraction.

When an API key is configured (see ``config.resolve_llm``) the pipeline asks an
LLM to read a turn and return STRUCTURED JSON — a list of typed memories with a
canonical slot key, value, and confidence. We force structured output rather
than parsing free text:

  * **Gemini** (default/primary): ``responseMimeType=application/json`` +
    ``responseSchema`` — the model can only emit JSON matching the schema.
  * **Anthropic**: a single ``extract_memories`` *tool* with ``tool_choice``
    forcing that tool; the result is read from the ``tool_use`` block.
  * **OpenAI**: the same idea via a forced function ``tool_choice``.

All three are called over plain HTTPS with ``httpx`` (no heavyweight SDKs in the
offline image; nothing imported unless a key is actually set), share ONE schema,
and return the SAME ``list[dict]`` shape, which ``pipeline.py`` normalises into
``MemoryDraft``s. Any error, timeout, or unusable response raises/returns None
so the caller falls back to the rule path — the LLM is an *enhancer*, never a
hard dependency.
"""
from __future__ import annotations

import json
import logging
import re

from .. import config
from .draft import MEMORY_TYPES

logger = logging.getLogger("memory.extraction.llm")

_SECRET_QUERY_PARAM = re.compile(
    r"([?&](?:key|api_key|apikey|access_token)=)[^&\s]+",
    re.IGNORECASE,
)

# Shared JSON schema describing the structured output we want. Kept in the
# OpenAPI subset all three providers accept.
_MEMORY_ITEM = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": list(MEMORY_TYPES)},
        "key": {"type": "string"},
        "value": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["type", "key", "value", "confidence"],
}
_RESULT_SCHEMA = {
    "type": "object",
    "properties": {"memories": {"type": "array", "items": _MEMORY_ITEM}},
    "required": ["memories"],
}

_SYSTEM = (
    "You extract durable, structured memories from one conversation turn for a "
    "long-term memory service. Capture personal facts (employment, location, "
    "family, pets), preferences, opinions, corrections, and IMPLICIT facts "
    "(e.g. 'walking Biscuit this morning' implies a pet named Biscuit). "
    "Do NOT store raw message text or transient chit-chat. For each memory pick "
    "type ∈ {fact,preference,opinion,event}; a canonical snake/dotted slot key "
    "(employment, location, origin, pet.name, diet, allergy, "
    "preference.answer_style, opinion.<subject>, …) so the same topic always "
    "reuses the same key. Use EXACTLY employment (not employer/job), location "
    "(not current_city), origin (not previous_city), and pet.name (not "
    "pet.dog.name). Use the main subject for opinion keys so an evolving stance "
    "reuses one slot (opinion.typescript, not opinion.typescript_generics). "
    "Write a short third-person value preserving the salient "
    "proper noun (e.g. 'Works at Notion'); and confidence in [0,1]. If a turn "
    "contains a correction ('actually, not X — Y'), emit the corrected (new) "
    "value. Return only memories actually supported by the turn; [] is valid."
)


def _render(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "user")
        name = m.get("name")
        who = f"{role}:{name}" if name else role
        lines.append(f"{who}: {m.get('content', '')}")
    return "\n".join(lines)


def _prompt(messages: list[dict]) -> str:
    return f"{_SYSTEM}\n\n--- TURN ---\n{_render(messages)}\n--- END ---"


def _coerce(payload: object) -> list[dict] | None:
    """Pull the memories list out of a parsed provider payload."""
    if isinstance(payload, dict):
        items = payload.get("memories")
    elif isinstance(payload, list):
        items = payload
    else:
        return None
    if not isinstance(items, list):
        return None
    return [i for i in items if isinstance(i, dict)]


def _safe_error(exc: Exception, api_key: str) -> str:
    """Keep provider failures useful without ever logging credentials."""
    detail = str(exc)
    if api_key:
        detail = detail.replace(api_key, "<redacted>")
    return _SECRET_QUERY_PARAM.sub(r"\1<redacted>", detail)


class LLMExtractor:
    """One configured provider. ``extract`` returns dicts or None (→ fallback)."""

    def __init__(self, provider: str, api_key: str, model: str) -> None:
        self.provider = provider
        self.api_key = api_key
        self.model = model

    def extract(self, messages: list[dict]) -> list[dict] | None:
        try:
            import httpx
        except ImportError:  # pragma: no cover - httpx ships in requirements
            logger.warning("httpx unavailable; using rule fallback")
            return None

        try:
            with httpx.Client(timeout=config.LLM_TIMEOUT) as client:
                if self.provider == "gemini":
                    return self._gemini(client, messages)
                if self.provider == "anthropic":
                    return self._anthropic(client, messages)
                if self.provider == "openai":
                    return self._openai(client, messages)
        except Exception as exc:  # network/timeout/parse — fall back to rules
            logger.warning(
                "llm extract failed (%s, %s): %s",
                self.provider,
                type(exc).__name__,
                _safe_error(exc, self.api_key),
            )
            return None
        return None

    # --- providers ---------------------------------------------------------
    def _gemini(self, client, messages):
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        body = {
            "contents": [{"role": "user", "parts": [{"text": _prompt(messages)}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": _RESULT_SCHEMA,
            },
        }
        r = client.post(
            url,
            headers={"x-goog-api-key": self.api_key},
            json=body,
        )
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return _coerce(json.loads(text))

    def _anthropic(self, client, messages):
        tool = {
            "name": "extract_memories",
            "description": "Return the structured memories found in the turn.",
            "input_schema": _RESULT_SCHEMA,
        }
        body = {
            "model": self.model,
            "max_tokens": 1024,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": "extract_memories"},
            "messages": [{"role": "user", "content": _prompt(messages)}],
        }
        r = client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
        )
        r.raise_for_status()
        for block in r.json().get("content", []):
            if block.get("type") == "tool_use":
                return _coerce(block.get("input"))
        return None

    def _openai(self, client, messages):
        fn = {
            "type": "function",
            "function": {
                "name": "extract_memories",
                "description": "Return the structured memories found in the turn.",
                "parameters": _RESULT_SCHEMA,
            },
        }
        body = {
            "model": self.model,
            "tools": [fn],
            "tool_choice": {"type": "function", "function": {"name": "extract_memories"}},
            "messages": [{"role": "user", "content": _prompt(messages)}],
        }
        r = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body,
        )
        r.raise_for_status()
        call = r.json()["choices"][0]["message"]["tool_calls"][0]
        return _coerce(json.loads(call["function"]["arguments"]))


def get_llm_extractor() -> LLMExtractor | None:
    """Build an extractor from the live config, or None when no key is set."""
    resolved = config.resolve_llm()
    if not resolved:
        return None
    return LLMExtractor(*resolved)
