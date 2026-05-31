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

# Current, non-retired defaults per provider (override with LLM_MODEL).
_DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-3-5-haiku-latest",
    "openai": "gpt-4o-mini",
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


# --- Recall (naive cosine top-k this phase) --------------------------------
TOP_K = int(os.environ.get("RECALL_TOP_K", "5"))
# Cosine-similarity floor for a memory to count as a match. Default is
# permissive (-1.0 keeps everything) so cold/irrelevant only returns empty when
# the user genuinely has no memories. Later phases tune this.
RECALL_MIN_SCORE = float(os.environ.get("RECALL_MIN_SCORE", "-1.0"))

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
