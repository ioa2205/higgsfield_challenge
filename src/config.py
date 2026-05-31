"""Single tunable surface for the memory service.

Every knob later phases want to tweak (recall, extraction, search, embedding)
lives here so there is exactly one place to look. Secrets and deployment
specifics come from the environment; this module only reads them.
"""
from __future__ import annotations

import os

# --- Embeddings -------------------------------------------------------------
# bge-small-en-v1.5 produces 384-dim vectors. EMBED_DIM is shared by the
# embedder and the `vector(384)` DB column so they can never drift apart.
EMBED_DIM = 384
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# "fastembed" = the real, offline, baked-into-the-image model (used in the
# container). "fake" = a deterministic hashing stand-in used by the test suite
# so pytest needs neither the model nor any network.
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "fastembed")
# Fixed cache dir the model is baked into at build time and loaded from offline.
EMBED_CACHE_DIR = os.environ.get("EMBED_CACHE_DIR", "/models")

# --- LLM extraction (provider-agnostic) ------------------------------------
# Phase 2 extracts typed memories with an LLM when an API key is present, and
# falls back to deterministic rules otherwise (or on any LLM error/timeout).
# The LLM layer is provider-agnostic: Gemini (default/primary), Anthropic, or
# OpenAI. Selection is `LLM_PROVIDER` (auto = pick by whichever key is set).
#
# The core service stays fully offline: with no key set, `resolve_llm()`
# returns None and extraction uses the rule path. The eval harness can enable
# the LLM by providing a key (see .env.example) without any code change.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "auto").strip().lower()
# Optional explicit model override; otherwise a current per-provider default is
# used (see _DEFAULT_MODELS). Kept configurable so a retired id is one env away.
LLM_MODEL = os.environ.get("LLM_MODEL") or None
# Generous budget: a safety net for hangs/failures, well inside the §3 60s
# /turns budget. Normal LLM latency must NOT trip the rule fallback.
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "25"))

# Current fast, production-sensible defaults per provider (override with
# LLM_MODEL). Gemini uses Google's evergreen Flash alias so a retired concrete
# model cannot break extraction; Anthropic and OpenAI use their current
# fast/high-quality extraction models.
_DEFAULT_MODELS = {
    "gemini": "gemini-flash-latest",
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-5-mini",
}
# Env var names that carry each provider's key. GOOGLE_API_KEY is accepted as an
# alias for Gemini.
_PROVIDER_KEYS = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
}


def _key_for(provider: str) -> str | None:
    for var in _PROVIDER_KEYS.get(provider, ()):
        val = os.environ.get(var)
        if val:
            return val
    return None


def resolve_llm() -> tuple[str, str, str] | None:
    """Resolve (provider, api_key, model) for LLM extraction, or None.

    Read live (not cached at import) so tests/the eval can toggle keys between
    app instances. Returns None when no usable provider/key is configured, which
    is the signal for the pipeline to use the deterministic rule path.
    """
    if LLM_PROVIDER in ("none", "off", "disabled", "fake"):
        return None

    order = (
        [LLM_PROVIDER]
        if LLM_PROVIDER in _PROVIDER_KEYS
        else ["gemini", "anthropic", "openai"]  # auto: first key wins
    )
    for provider in order:
        key = _key_for(provider)
        if key:
            model = LLM_MODEL or _DEFAULT_MODELS[provider]
            return provider, key, model
    return None


# --- Recall: hybrid retrieval + RRF + tiered assembly (Phase 3) ------------
# THE recall tuning surface. The measure-tune loop changes exactly ONE of these
# per round and re-runs tests/fixture_runner.py (see CHANGELOG v0.3).
#
# Legacy naive top-k (kept for reference / any direct vector call).
TOP_K = int(os.environ.get("RECALL_TOP_K", "5"))
# Per-source retrieval depth fed into RRF (semantic = pgvector cosine, keyword =
# Postgres full-text ts_rank). Deeper N = more recall, more fusion work.
SEM_TOP_N = int(os.environ.get("SEM_TOP_N", "20"))
KW_TOP_N = int(os.environ.get("KW_TOP_N", "20"))
# Reciprocal Rank Fusion constant: rrf = Σ 1/(RRF_K + rank). ~60 is the standard
# value; larger flattens the rank weighting, smaller sharpens the head.
RRF_K = int(os.environ.get("RRF_K", "60"))
# THE NOISE GATE. A memory counts as relevant (→ /recall emits context) iff it
# is a keyword hit (ts_rank > 0) OR its vector cosine clears this floor. A query
# about an undiscussed topic clears neither, so /recall returns empty (§9). With
# the real bge embedder this is the main knob to keep noise empty without
# dropping true hits; the keyword half carries the deterministic test path.
RECALL_MIN_SCORE = float(os.environ.get("RECALL_MIN_SCORE", "0.55"))
# Tier-3 "recent conversation" tier: how many recent session-scoped events to
# consider for the second section.
TIER3_RECENT_N = int(os.environ.get("TIER3_RECENT_N", "5"))
# Max characters per rendered snippet / citation (keeps lines budget-friendly).
SNIPPET_MAX = int(os.environ.get("RECALL_SNIPPET_MAX", "240"))

# --- Phase 4: supersession ------------------------------------------------
# Fuzzy supersession similarity floor. When two memories share the same slot
# key, exact-key match fires (deterministic). When they DON'T share a key,
# the fuzzy path checks cosine similarity: if >= this threshold AND same type,
# the newer supersedes the older. Too low → false merges (unrelated facts
# wrongly collapsed); too high → missed merges (stale facts remain active).
# Tuned via the measure-tune loop recorded in CHANGELOG v0.4.
SUPERSESSION_SIM_THRESHOLD = float(
    os.environ.get("SUPERSESSION_SIM_THRESHOLD", "0.92")
)

# --- Database --------------------------------------------------------------
POOL_MIN = int(os.environ.get("POOL_MIN", "1"))
POOL_MAX = int(os.environ.get("POOL_MAX", "10"))


def database_url() -> str:
    """Build the asyncpg DSN from DATABASE_URL or discrete PG* env vars."""
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    user = os.environ.get("PGUSER", "memory")
    password = os.environ.get("PGPASSWORD", "memory")
    name = os.environ.get("PGDATABASE", "memory")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


# --- Auth ------------------------------------------------------------------
def auth_token() -> str | None:
    """Optional bearer token. Read live so it can be toggled at runtime/tests."""
    return os.environ.get("MEMORY_AUTH_TOKEN") or None


LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# --- Request hardening ------------------------------------------------------
# Enforced at the ASGI boundary before JSON parsing. Field-level limits below
# keep valid-but-pathological payloads bounded after the body-size check.
MAX_REQUEST_BODY_BYTES = int(os.environ.get("MAX_REQUEST_BODY_BYTES", str(1024 * 1024)))
MAX_MESSAGES_PER_TURN = int(os.environ.get("MAX_MESSAGES_PER_TURN", "100"))
MAX_MESSAGE_CONTENT_CHARS = int(os.environ.get("MAX_MESSAGE_CONTENT_CHARS", "50000"))
MAX_QUERY_CHARS = int(os.environ.get("MAX_QUERY_CHARS", "4096"))
