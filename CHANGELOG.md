# Changelog

## v0.2 — Hybrid extraction + recall-quality fixture

**What changed:** `extraction/naive.py` (one `event` memory per turn) is replaced by a
hybrid, synchronous pipeline — `extraction/{draft,rules,llm,pipeline}.py`:

- **Provider-agnostic LLM path** (`llm.py`). Originally specced for Claude only; generalised
  to any provider. `config.resolve_llm()` selects **Gemini (primary)**, **Anthropic**, or
  **OpenAI** by `LLM_PROVIDER` (`auto` = whichever API key is set). Each provider is called
  over plain HTTPS with `httpx` (no heavyweight SDKs in the offline image, imported only when
  a key exists) and forced to emit STRUCTURED JSON: Gemini via `responseSchema`, Anthropic and
  OpenAI via a single `extract_memories` tool with a forced tool choice. All three share one
  JSON schema and return the same `list[dict]`. Any error/timeout (`LLM_TIMEOUT`, default 25s,
  inside the 60s `/turns` budget) returns `None` → rule fallback. Model ids are configurable
  (`LLM_MODEL`) with current per-provider defaults so a retired id is one env var away.
- **Deterministic rule path** (`rules.py`). Regex extractor for employment, location/origin,
  pets (explicit *and* implicit — "walking Biscuit" → `pet.name`), family, diet, allergy,
  preferences, and opinions, with correction cues raising confidence. It is the offline default
  and the safety net, so it is built to be genuinely useful, not a stub.
- **One schema, one slot vocabulary** (`draft.py`). Both paths emit `MemoryDraft(type, key,
  value, confidence, provenance)` with canonical slot keys (`employment`, `location`,
  `pet.name`, `diet`, `opinion.<subject>`, …) so a later phase can supersede same-slot facts.
  A new additive `provenance` column records which path produced each memory; the `search`
  tsvector is now populated on insert for phase-3 keyword recall. `/turns` stays synchronous —
  extract + embed + index all happen before 201.
- **Pipeline policy** (`pipeline.py`): LLM-primary → rule fallback → (only if nothing typed
  matched) a single typed `event` memory so an arbitrary turn is still recallable. The event
  fallback is the *only* place a near-raw value is stored, and it is typed `event`, never a
  fake "fact" — no raw chunk masquerading as structured knowledge.

**Self-eval fixture** (`fixtures/conversations.json` + `tests/fixture_runner.py`): 6 scripted
multi-session scenarios (basic recall, implicit pet, fact evolution, multi-hop, noise, opinion
arc) with probes. The runner reports two metrics — EXTRACTION (expected facts present as typed,
**non-`event`** memories) and RECALL-CONTEXT (present in `/recall` context) — with a per-probe
breakdown, and is idempotent (DELETEs each user before ingest).

**Baseline (rule path, fake embedder):**
- EXTRACTION **8/8 (100%)**. RECALL-CONTEXT 8/9 (the noise probe is the one miss). RECALL
  in-scope 5/5 — fact-evolution, multi-hop, and noise recall probes are intentionally out of
  scope this phase (they need supersession / an entity graph / a recall threshold, all phase 3).

**Tuning rounds (real before/after on the fixture):**
- *Round 0 → honest metric.* The first metric scored 7/7, but inspection showed the implicit-
  pet and multi-hop "pet" facts were only present in the raw `event`-fallback text, not as
  typed memories — the exact raw-chunk anti-pattern the challenge warns about. Changing the
  EXTRACTION metric to count **typed, non-`event`** memories only dropped the honest baseline to
  **6/8 (75%)** and exposed two real misses.
- *Round 1: pet regex casing → 6/8 → 8/8.* The pet patterns lacked case-insensitivity, so
  sentence-initial "Took Biscuit" / "My dog Biscuit" never matched and fell to the event
  fallback. Scoped inline flags (`(?i:…)`) on the verb/keyword — keeping the NAME group
  case-sensitive so only a capitalised proper noun is captured — fixed both. Extraction reached
  **8/8 (100%)**; further fixture changes can't increase it, so tuning stopped (per the loop's
  two-no-gain stop rule).
- *Caught by tests, not the fixture:* `test_correction` and the rule-fallback test failed
  because employment required "I" immediately before "work" — so "…and work at Acme" and
  "I actually work at Notion" missed. Made the leading "I (am)" + adverb optional so the match
  anchors on "work(s) at/for X". (The self-fixture is co-authored with the rules, so it can't
  catch every gap; the unit tests and the eventual LLM path generalise to unseen phrasings.)

**Honest limitations:** the rule path is brittle to phrasings outside its patterns — that is
precisely what the LLM path is for, and what the held-out private eval will probe. A 100%
self-fixture mainly proves the rules cover the cases I designed; it is an iteration instrument,
not a generalisation claim.

**Next (phase 3):** hybrid recall (vector + tsvector, RRF), token-budgeted tiered assembly, a
recall threshold for noise, then supersession (fact evolution) and the entity layer (multi-hop).

## v0.1 — Skeleton, contract & Docker

**What changed:** Stood up the service end-to-end so the eval harness can talk
to it. All seven §3 endpoints exist with exact request/response shapes and
status codes (`/health` 200, `/turns` 201, `/recall` 200, `/search` 200,
`/users/{id}/memories` 200, the two `DELETE`s 204). `/turns` accepts single- and
multi-message turns including `tool`-role messages with a nullable `name`.
Optional bearer auth (enforced only when `MEMORY_AUTH_TOKEN` is set; `/health`
always open). `docker compose up` boots a `pgvector/pgvector:pg16` database and
the app on port 8080 with healthchecks, `depends_on: service_healthy`, and one
named volume for persistence.

**Why these choices:**
- **Single Postgres + pgvector store.** One backing store for vector, future
  full-text, and relational fact history removes a whole class of cross-store
  consistency bugs and makes the "if you wrote it, you can read it" guarantee a
  single-transaction property. The *full* product schema (`turns`, `messages`,
  `memories` with supersession/embedding/tsvector columns, `entities`,
  `memory_entities`) is created now via idempotent migrations so no later phase
  needs a destructive migration.
- **Offline local embeddings.** `BAAI/bge-small-en-v1.5` (384-dim) is baked into
  the image at build time and loaded offline at runtime, so boot needs no
  network and the hot path never depends on an API key the evaluator might not
  set. `EMBED_DIM=384` is shared between the embedder and the `vector(384)`
  column so they cannot drift.
- **Naive recall as an honest baseline.** Each turn yields at least one typed
  `event` memory (never a raw message chunk); `/recall` returns the top-k by
  cosine (`<=>`). This is deliberately a baseline — the challenge says vanilla
  cosine-top-k won't score well, and hybrid retrieval + RRF + tiered assembly
  arrive in later phases. There is **no recall-quality metric yet**; that
  instrument (the fixture + runner) is built in Phase 2.
- **Cross-session scoping decision.** A user's facts are shared across all of
  that user's sessions (recall filters by `user_id`, not `session_id`) so the
  smoke test's `smoke-1`→`smoke-2` Berlin recall works; different users never
  bleed. Only the future "recent conversation" tier is session-scoped.

**Tests:** contract roundtrip (incl. tool message, `name` null & set), restart
persistence, user isolation + same-user cross-session sharing, malformed input
(4xx, no crash), both auth branches, and both delete cascades (idempotent).

**Next:** Phase 2 — hybrid extraction (Claude + deterministic rule fallback)
producing typed memories, plus the recall-quality fixture and a two-metric
runner to drive every later tuning decision.
