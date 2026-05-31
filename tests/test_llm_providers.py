"""Provider request contracts: current defaults, structured output, secret safety."""
from __future__ import annotations

import json

from src import config
from src.entities.populate import _entities_from_draft
from src.extraction.draft import MemoryDraft
from src.extraction.llm import LLMExtractor, _safe_error
from src.extraction.pipeline import _normalize_llm

MESSAGES = [{"role": "user", "content": "I live in Berlin."}]
SECRET = "secret-provider-key"


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Client:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return _Response(self.response)


def test_current_provider_defaults():
    assert config._DEFAULT_MODELS == {
        "gemini": "gemini-flash-latest",
        "anthropic": "claude-haiku-4-5",
        "openai": "gpt-5-mini",
    }


def test_gemini_uses_header_auth_and_structured_output():
    client = _Client(
        {"candidates": [{"content": {"parts": [{"text": '{"memories": []}'}]}}]}
    )
    result = LLMExtractor("gemini", SECRET, "gemini-flash-latest")._gemini(
        client, MESSAGES
    )
    call = client.calls[0]
    assert result == []
    assert SECRET not in call["url"]
    assert "params" not in call
    assert call["headers"] == {"x-goog-api-key": SECRET}
    assert call["json"]["generationConfig"]["responseMimeType"] == "application/json"
    assert "responseSchema" in call["json"]["generationConfig"]


def test_anthropic_uses_header_auth_and_forced_tool():
    client = _Client(
        {"content": [{"type": "tool_use", "input": {"memories": []}}]}
    )
    result = LLMExtractor("anthropic", SECRET, "claude-haiku-4-5")._anthropic(
        client, MESSAGES
    )
    call = client.calls[0]
    assert result == []
    assert SECRET not in call["url"]
    assert call["headers"]["x-api-key"] == SECRET
    assert call["json"]["tool_choice"] == {
        "type": "tool",
        "name": "extract_memories",
    }


def test_openai_uses_header_auth_and_forced_tool():
    client = _Client(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"arguments": json.dumps({"memories": []})}}
                        ]
                    }
                }
            ]
        }
    )
    result = LLMExtractor("openai", SECRET, "gpt-5-mini")._openai(client, MESSAGES)
    call = client.calls[0]
    assert result == []
    assert SECRET not in call["url"]
    assert call["headers"] == {"Authorization": f"Bearer {SECRET}"}
    assert "temperature" not in call["json"]
    assert call["json"]["tool_choice"] == {
        "type": "function",
        "function": {"name": "extract_memories"},
    }


def test_provider_error_redaction_removes_key_and_query_secret():
    detail = _safe_error(
        RuntimeError(
            f"request failed: https://example.test/run?key={SECRET}&other=1 "
            f"token={SECRET}"
        ),
        SECRET,
    )
    assert SECRET not in detail
    assert "key=<redacted>" in detail


def test_llm_aliases_normalize_into_canonical_fact_slots():
    drafts = _normalize_llm(
        [
            {"type": "preference", "key": "current_city", "value": "Berlin", "confidence": 0.9},
            {"type": "fact", "key": "pet.dog.name", "value": "Biscuit", "confidence": 0.8},
            {"type": "fact", "key": "pet.type", "value": "Dog", "confidence": 0.8},
            {"type": "fact", "key": "allergy", "value": "shellfish", "confidence": 0.9},
        ],
        provenance="llm:gemini",
    )
    assert [(d.type, d.key, d.value) for d in drafts] == [
        ("fact", "location", "Lives in Berlin"),
        ("fact", "pet.name", "Has a pet named Biscuit"),
        ("fact", "pet.species", "Dog"),
        ("fact", "allergy", "Allergic to shellfish"),
    ]


def test_entity_population_accepts_llm_employment_wording():
    assert _entities_from_draft(
        MemoryDraft("fact", "employment", "Works as a backend engineer at Stripe", 0.9)
    ) == [("employer", "Stripe", "employer_of")]


def test_entity_population_accepts_llm_pet_wording():
    assert _entities_from_draft(
        MemoryDraft("fact", "pet.name", "Dog is named Biscuit", 0.9)
    ) == [("pet", "Biscuit", "pet_of")]
