"""Provider request contracts: current defaults, structured output, secret safety."""
from __future__ import annotations

import json

import httpx

from src import config
from src.entities.populate import _entities_from_draft
from src.extraction.draft import MemoryDraft
from src.extraction.llm import LLMExtractor, _safe_error, get_llm_extractors
from src.extraction.pipeline import _normalize_llm, extract

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


class _StubExtractor:
    def __init__(self, provider, result, calls):
        self.provider = provider
        self.model = f"{provider}-model"
        self.timeout = 1.0
        self._result = result
        self._calls = calls

    def extract(self, messages):
        self._calls.append(self.provider)
        return self._result


def _location(value):
    return [{"type": "fact", "key": "location", "value": value, "confidence": 0.9}]


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


def test_auto_gemini_success_stops_failover(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "src.extraction.pipeline.get_llm_extractors",
        lambda: [
            _StubExtractor("gemini", _location("Lives in Lisbon"), calls),
            _StubExtractor("anthropic", _location("Lives in Porto"), calls),
            _StubExtractor("openai", _location("Lives in Berlin"), calls),
        ],
    )
    drafts = extract(MESSAGES)
    assert calls == ["gemini"]
    assert drafts[0].provenance == "llm:gemini"


def test_auto_gemini_failure_anthropic_success(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "src.extraction.pipeline.get_llm_extractors",
        lambda: [
            _StubExtractor("gemini", None, calls),
            _StubExtractor("anthropic", _location("Lives in Porto"), calls),
            _StubExtractor("openai", _location("Lives in Berlin"), calls),
        ],
    )
    drafts = extract(MESSAGES)
    assert calls == ["gemini", "anthropic"]
    assert drafts[0].provenance == "llm:anthropic"


def test_auto_gemini_and_anthropic_failure_openai_success(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "src.extraction.pipeline.get_llm_extractors",
        lambda: [
            _StubExtractor("gemini", None, calls),
            _StubExtractor("anthropic", None, calls),
            _StubExtractor("openai", _location("Lives in Berlin"), calls),
        ],
    )
    drafts = extract(MESSAGES)
    assert calls == ["gemini", "anthropic", "openai"]
    assert drafts[0].provenance == "llm:openai"


def test_all_configured_providers_fail_rules_run(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "src.extraction.pipeline.get_llm_extractors",
        lambda: [
            _StubExtractor("gemini", None, calls),
            _StubExtractor("anthropic", None, calls),
            _StubExtractor("openai", None, calls),
        ],
    )
    drafts = extract([{"role": "user", "content": "I live in Lisbon."}])
    assert calls == ["gemini", "anthropic", "openai"]
    assert drafts[0].value == "Lives in Lisbon"
    assert drafts[0].provenance == "rule"


def test_forced_gemini_resolves_only_gemini(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    assert [item[0] for item in config.resolve_llms()] == ["gemini"]


def test_no_configured_keys_uses_rules_without_network(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("network client must not be constructed")

    monkeypatch.setattr(httpx, "Client", fail_if_called)
    drafts = extract([{"role": "user", "content": "I live in Lisbon."}])
    assert called is False
    assert drafts[0].provenance == "rule"


def test_auto_model_override_applies_only_to_first_provider(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(config, "LLM_MODEL", "gemini-custom")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    assert [(p, model) for p, _, model in config.resolve_llms()] == [
        ("gemini", "gemini-custom"),
        ("anthropic", "claude-haiku-4-5"),
        ("openai", "gpt-5-mini"),
    ]


def test_auto_timeout_budget_is_divided_across_attempts(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(config, "LLM_TIMEOUT", 25.0)
    monkeypatch.setattr(config, "LLM_AUTO_TOTAL_TIMEOUT", 45.0)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    assert [extractor.timeout for extractor in get_llm_extractors()] == [15.0, 15.0, 15.0]


def test_malformed_llm_output_runs_rules(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "src.extraction.pipeline.get_llm_extractors",
        lambda: [_StubExtractor("gemini", [{"type": "invalid"}], calls)],
    )
    drafts = extract([{"role": "user", "content": "I live in Lisbon."}])
    assert calls == ["gemini"]
    assert drafts[0].value == "Lives in Lisbon"
    assert drafts[0].provenance == "rule"


def test_provider_timeout_runs_rules_and_logs_no_secret(monkeypatch, caplog):
    secret = "timeout-secret-key"

    class _TimeoutClient:
        def __init__(self, **kwargs):
            raise httpx.TimeoutException(f"https://example.test/run?key={secret}")

    monkeypatch.setattr(httpx, "Client", _TimeoutClient)
    extractor = LLMExtractor("gemini", secret, "gemini-flash-latest", timeout=0.1)
    assert extractor.extract(MESSAGES) is None
    assert secret not in caplog.text
    assert "key=<redacted>" in caplog.text
