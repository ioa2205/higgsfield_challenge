# PLAN.md — Phased Build Plan for the Memory Service

> This is a build plan, not code. It splits one shippable product into five construction
> stages. Each stage leaves the system working and must not regress an earlier stage.
> Each phase is executed by a **fresh Claude Code session** that has no memory of any prior
> chat **and does not read this prose** — it reads only its own copy-paste prompt plus
> `CLAUDE.md`, `CHALLENGE.md`, and the actual code on disk. Therefore every load-bearing
> detail is restated inside the per-phase prompts, and Phase 1 records the cross-cutting
> conventions into `CLAUDE.md` so later sessions inherit them.

---

## 1. Architecture summary & why the phases split this way

**Architecture (one paragraph).** A single async **FastAPI (Python 3.11)** monolith backed by
**one Postgres 16 + pgvector container** with **one named Docker volume**. Postgres is the single
backing store playing three roles at once: **pgvector** for semantic search, native **full-text
search** (`tsvector` + GIN, `ts_rank`) for exact keyword search, and ordinary **relational tables**
for typed memories, slot-based supersession history, and a small entity graph. Embeddings are
produced by a **local bge-small model baked into the image** (fastembed/ONNX or sentence-transformers,
384-dimensional) — no API key on the hot path, no rate limits, deterministic, and it survives a
missing key. Extraction is **hybrid**: the Anthropic Claude API does the intelligent reading, with a
**deterministic rule-based fallback** that runs when the key is absent or the call hangs/errors, so
extraction degrades gracefully. Recall is **hybrid retrieval** — pgvector (semantic) + full-text
(keyword) fused with **Reciprocal Rank Fusion**, then a **priority + token-budget assembly** pass
(stable user facts → query-relevant → recent). Fact evolution uses a **`memories` table with slot
keys, a `supersedes` pointer, and an `active` flag** — old facts are marked superseded, never deleted.
Multi-hop is handled by a **small entity layer** (user / pet / employer / city as linked entities)
plus **query decomposition**. Default port **8080**; `docker compose up` boots everything with no
manual setup; data survives `docker compose down && up`.

**Cross-session scoping (an intentional design decision, documented in the README).** A user's
**stable facts/preferences are shared across that user's sessions** — this is required by the §7 smoke
test (it writes a turn under `session_id=smoke-1` and recalls under `session_id=smoke-2` for the same
`user-1`, still expecting "Berlin"). **Session scoping applies only to the "recent conversation" tier
and to `/search` when a `session_id` is supplied.** Concurrent sessions belonging to **different users
never bleed**. This is the single most important behavioral decision the plan pins, because it is
where a naive "scope everything by session" implementation would silently fail the smoke test and the
§9 cross-session category.

**Why these choices (justifying the locked architecture).**
- *One Postgres for all three roles* removes a class of consistency/orchestration bugs: no cross-store
  sync, no "vector DB says X but relational says Y," and the §5 *synchronous-correctness* requirement
  ("if you wrote it, you can read it") falls out of a single transaction. It is also the most
  defensible answer to the §6 README "backing store choice" prompt.
- *Local embeddings* satisfy the §5/§9 robustness and "missing API keys" categories: the hot retrieval
  path never depends on a network call or a key the evaluator might not set.
- *Hybrid extraction* hedges the same risk for writes: Claude gives high-quality typed extraction when
  the key is present; the rule fallback guarantees the service still extracts structured memories
  (not raw chunks) when it is not — exactly the §9 graceful-degradation behavior.
- *RRF + priority assembly* is the deliberate, non-vanilla recall the challenge demands
  ("vanilla cosine-top-k will not score"). RRF needs no score calibration between the two retrievers,
  and tiered assembly is the explicit budget triage §4/§10 ask us to defend.
- *Slot + supersedes + active* is the minimal honest model for "supersede, never delete, keep the
  chain inspectable" (§4 fact evolution, §9).

**Why this phase ordering.** The phases are **floors of one house**, not a bad house rebuilt:
1. **Contract + Docker first** so the eval harness can *always* talk to the service. After phase 1,
   everything is a *quality* improvement, never a *compatibility* risk — the riskiest external
   requirements ("`docker compose up` with no setup," "passes the smoke test," "exact shapes/status
   codes") are locked down on day one.
2. **Extraction + the measurement fixture together**, because every later quality decision is made by
   *measuring*, not by writing code. The ruler must exist before any tuning begins — so the fixture
   is built here, in the extraction phase, and run after every subsequent change.
3. **Hybrid recall** is where the **primary eval signal** (recall quality) gets its first real number
   and its first measure-tune loop, on top of a store that already holds structured memories.
4. **Fact evolution + multi-hop** are the two hardest grading categories; they sit on a *measurable*
   base so their tuning (the supersession threshold) is driven by real fixture cases.
5. **Hardening + final tune + docs** turns a working system into a *shippable* one and writes the
   README/CHANGELOG that carry ~50% of the grade.

Each phase's fixture score is an **honest measurement of a working product**, not a planted flaw: in
phase 2 the advanced probes (evolution, multi-hop) legitimately score low because those features do
not exist yet — that is a true baseline, not a deliberate bug to "fix later." The product is fully
functional for what it claims at every phase.

**Intended repository layout** (phase 1 establishes this in `CLAUDE.md`; later phases extend it and
must *discover the actual names on disk* rather than assume them):

```
memory-service/
├── README.md            # stub in phase 1, full in phase 5
├── CHANGELOG.md         # one honest entry appended per phase + per real tuning round
├── CLAUDE.md            # architecture, conventions, file map, Status (done/next) — the cross-session anchor
├── docker-compose.yml   # app + db (pgvector/pgvector:pg16), named volume, port 8080, healthchecks
├── Dockerfile           # service image; bakes the bge-small model into a fixed cache dir at build time
├── .env.example         # ANTHROPIC_API_KEY (optional), MEMORY_AUTH_TOKEN (optional), DB vars
├── src/
│   ├── main.py          # FastAPI app + lifespan (load embedder, open pool w/ pgvector adapter, run idempotent migrations)
│   ├── config.py        # env settings + ALL tunables (top-k, RRF k, top-Ns, tier token splits, thresholds, EMBED_DIM)
│   ├── api/             # routes.py (7 endpoints), schemas.py (exact contract shapes), auth.py (optional bearer)
│   ├── db/              # pool.py (asyncpg pool + register_vector init), migrations.sql (idempotent), repository.py
│   ├── embeddings/      # embedder.py (local bge-small load from baked offline cache + encode)
│   ├── extraction/      # llm.py (Claude), rules.py (fallback), pipeline.py (orchestration → normalized memories)
│   ├── recall/          # retrieval.py (pgvector + FTS), fusion.py (RRF), assembly.py (tiered token-budget builder)
│   ├── search/          # search.py (structured /search)
│   ├── evolution/       # supersession.py (slot detection + supersede chain)
│   └── entities/        # entities.py (entity layer + linking), decompose.py (multi-hop decomposition)
├── tests/               # test_contract, test_persistence, test_isolation, test_malformed, test_auth, test_delete,
│   │                    #   test_extraction, test_recall, test_evolution, test_multihop, test_robustness, fixture_runner
└── fixtures/            # scripted multi-session conversations + probe queries with expected facts
```

**Pinned implementation specifics** (so fresh sessions don't guess; Phase 1 writes these into
`CLAUDE.md`'s conventions, and each phase prompt restates the ones it needs):
- **DB image:** `pgvector/pgvector:pg16` (stock `postgres:16` lacks the extension). `CREATE EXTENSION IF NOT EXISTS vector`.
- **Embedding:** `BAAI/bge-small-en-v1.5` → **384 dims**. Column is `embedding vector(384)`; `EMBED_DIM=384`
  is a config constant the embedder and column share so they cannot drift. The model is **downloaded into
  a fixed cache dir at Docker build time and loaded offline at runtime** (set the cache path + offline env
  flags) so boot needs no network.
- **pgvector access:** async driver (**asyncpg** by default) with the **pgvector type adapter registered on
  every pooled connection** (`pgvector.asyncpg.register_vector` via the pool `init` callback). Cosine via
  the `<=>` operator; if an index is created, `USING hnsw (embedding vector_cosine_ops)` (HNSW builds on an
  empty table at first-boot migration; IVFFlat does not). At this data scale an exact scan is correct and
  fast — the index is for correctness/future scale, not a speed requirement.
- **Token budget:** approximate with a conservative, dependency-free heuristic that **over-counts**
  (`tokens ≈ max(words × 1.3, chars / 4)`), or `tiktoken`/the SDK counter if present; trim to stay under
  budget so the "never > 2×" guarantee survives unicode-heavy input.
- **Anthropic extraction:** read the installed `anthropic` SDK to pick a current model id (don't hardcode a
  possibly-retired string); force structured JSON via a defined extraction **tool + `tool_choice`**, and
  read the result from the `tool_use` content block. Fallback fires on **missing key or real failure/hang**
  (a generous ~20–30s timeout that fits inside the §3 60s `/turns` budget) — **not** on normal LLM latency.

**The single tunable surface.** All ranking weights and thresholds live in `src/config.py` (overridable by
env), **seeded in Phase 1** with the naive recall's `top_k`. The measure-tune loop in phases 3–5 changes
**one named value there per round** — that is what makes "change ONE thing, re-run, compare" a clean,
honest operation.

---

## 2. Phases

### Phase 1 — Skeleton, Contract & Docker

**Goal.** Stand up the service so the eval harness can talk to it: all seven endpoints exist with the
exact shapes and status codes from `CHALLENGE.md` §3, a `pgvector/pgvector:pg16` container with a named
volume and healthchecks, offline local embeddings, and a naive-but-working store/recall path so the
system runs end-to-end and passes the challenge smoke test (including its **cross-session, same-user**
recall). This phase also creates `CLAUDE.md`, the cross-session anchor, and seeds the conventions later
phases depend on.

**In scope.**
- FastAPI app + lifespan startup (open asyncpg pool with the pgvector adapter registered, run idempotent
  migrations, load the local embedder from its baked offline cache).
- All 7 endpoints with exact request/response shapes and status codes per §3 (`GET /health`→200,
  `POST /turns`→201, `POST /recall`→200, `POST /search`→200, `GET /users/{id}/memories`→200,
  `DELETE /sessions/{id}`→204, `DELETE /users/{id}`→204). `/turns` must accept single- and
  **multi-message turns including `tool`-role messages with a nullable `name`**.
- Optional `Authorization: Bearer` auth: enforced iff `MEMORY_AUTH_TOKEN` is set, ignored otherwise;
  `/health` stays open. Both branches tested.
- Full DB schema in idempotent migrations (the schema the *whole product* needs, created now so later
  phases never do destructive migrations): `turns`, `messages`, `memories`
  (id, user_id, type, key, value, confidence, source_session, source_turn, created_at, updated_at,
  supersedes, active, `embedding vector(384)`, `search tsvector`), **`entities`
  (id, user_id, entity_type ∈ {user,pet,employer,city}, name, created_at)** and **`memory_entities`
  (memory_id, entity_id, relation)** — populated later, but with their columns fixed now — plus the GIN
  index on the tsvector and an HNSW `vector_cosine_ops` index on the embedding.
- **Create the package directories `src/extraction/`, `src/recall/`, `src/search/`** now (even if they
  hold only the thin naive helpers this phase) so phases 2–4 extend them in place.
- Local bge-small embedder (384-dim) baked into the image at build time into a fixed cache dir, loaded
  **offline** at runtime; used to embed stored text.
- A **naive but honest** store/recall: persist turns + a minimal structured memory per turn (at least a
  typed `event` record, never a raw chunk), embed it, and recall by vector cosine (`<=>`) formatted as
  the §3 example context with citations. **Stable user facts recall across the same user's sessions**
  (so the smoke test's `smoke-1`→`smoke-2` Berlin recall works); session scoping applies only to recent
  context. Return `{"context":"","citations":[]}` with 200 on cold/irrelevant.
- `config.py` seeded with `top_k` and `EMBED_DIM=384` as the first named tunables.
- Structured logging of lifecycle events (startup, embedder load, extraction path chosen, auth
  enforced/skipped) — a §10 "well-logged" down-payment.
- `docker-compose.yml` (`app` on 8080 + `db` on `pgvector/pgvector:pg16`, one named volume, healthchecks
  on both, `depends_on: condition: service_healthy`), `Dockerfile`, `.env.example`.
- A short `README.md` stub and `CHANGELOG.md` (`v0.1` entry).
- **`CLAUDE.md`** with: architecture overview, **the pinned implementation specifics above** as
  conventions, the file map, and a `## Status — done / next` section.

**Explicitly out of scope.** LLM extraction, the rule fallback, RRF, full-text retrieval, priority-tier
assembly, supersession, the entity/multi-hop population, the recall-quality fixture, any tuning. Recall
stays naive cosine top-k this phase — but the schema must already support every future feature so no
later phase needs a destructive migration.

**Definition of done (testable).**
- From a clean checkout with **no network egress**, `docker compose up -d` then
  `until curl -sf localhost:8080/health; do sleep 1; done` succeeds; `/health` returns 200 (proves the
  embedder loads offline).
- The §7 smoke sequence passes: `/turns`→`201 {"id": ...}`; **`/recall` under `session_id=smoke-2` for
  `user-1` returns context mentioning Berlin** (cross-session, same user); `/users/user-1/memories`
  returns **structured, typed JSON records**, not raw message text.
- `pytest` green for these named files:
  - `test_contract.py` — write a turn (including a multi-message turn with a `tool` message, `name` both
    null and non-null) → recall it → response **shapes and explicit status codes** match §3 (assert
    `/turns`=201, `/recall`=200, `/search`=200, `/users/{id}/memories`=200); cold `/recall` and empty
    `/search` return 200 with empty payloads.
  - `test_persistence.py` — write, `docker compose down && up`, data still recallable.
  - `test_isolation.py` — two **different users** don't bleed; **and** a same-user write in session A is
    recallable in session B (cross-session sharing works as designed).
  - `test_malformed.py` — bad JSON / missing fields / unicode → 4xx, never a crash; `/health` still 200 after.
  - `test_auth.py` — with `MEMORY_AUTH_TOKEN` unset, all endpoints work with no header; with it set, a
    missing/wrong token → 401/403 and the correct token → normal status; `/health` open regardless.
  - `test_delete.py` — `DELETE /sessions/{id}`→204 removes that session only; `DELETE /users/{id}`→204
    cascades to memories+turns+sessions (then `/users/{id}/memories` empty, `/recall` empty-200); deleting
    a non-existent id still returns 204 (idempotent).
- `CLAUDE.md` exists with all four sections; `CHANGELOG.md` has the `v0.1` entry.

**Artifacts created/updated.** `CLAUDE.md` (new), `docker-compose.yml`, `Dockerfile`, `.env.example`,
`src/` (main, config, api/, db/, embeddings/, and thin extraction/, recall/, search/ packages),
`tests/` (test_contract, test_persistence, test_isolation, test_malformed, test_auth, test_delete),
`README.md` (stub), `CHANGELOG.md` (v0.1).

**Copy-paste prompt:**

````
You are a fresh Claude Code session. Build PHASE 1 of a Dockerized "memory service" for an AI agent.

FIRST, orient yourself from disk (do not assume any prior state):
1. Read CHALLENGE.md in full — it is the spec. Focus on §3 (HTTP contract: exact request/response shapes
   and status codes; note /turns accepts multi-message turns INCLUDING role "tool" with a nullable
   "name"), §5 (hard constraints), §7 (smoke test + required tests — note it writes session_id "smoke-1"
   and RECALLS under session_id "smoke-2" for the SAME user-1 and still expects Berlin), §8 (the exact
   `docker compose up` setup graders use), §9/§10 (grading).
2. List the repository contents and read whatever already exists. CLAUDE.md does NOT exist yet — you will
   create it. There may be only CHALLENGE.md and a PLAN.md.

LOCKED ARCHITECTURE (do not re-decide; implement it):
- Python 3.11 + FastAPI (async), single monolith.
- ONE Postgres 16 + pgvector container as the SINGLE backing store, ONE named Docker volume. Use the
  image `pgvector/pgvector:pg16` (stock postgres:16 has no pgvector). It will later serve three roles
  (vector + full-text + relational fact history) — create the FULL schema now via idempotent migrations
  so no later phase needs a destructive migration. Run `CREATE EXTENSION IF NOT EXISTS vector`.
- Embeddings: a LOCAL bge-small model — BAAI/bge-small-en-v1.5, which produces 384-dim vectors. DOWNLOAD
  it into a fixed cache dir at Docker BUILD time and load it OFFLINE at runtime (set the library's cache
  path and offline flags, e.g. HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE or fastembed's cache path, and verify
  the actual offline knobs for whichever library you pick — fastembed/ONNX or sentence-transformers).
  Boot must need NO network. Put EMBED_DIM=384 in config so the embedder and the DB column share it.
- DB access: an async driver (asyncpg by default) with the pgvector TYPE ADAPTER registered on EVERY
  pooled connection (pgvector.asyncpg register_vector via the pool init callback) — without this, vectors
  round-trip as strings. Use the cosine operator `<=>`. If you create a vector index use
  `USING hnsw (embedding vector_cosine_ops)` (HNSW builds on an empty table at first-boot migration;
  IVFFlat does not). At this scale an exact scan is fine; the index is for correctness/scale, not speed.
- Default port 8080. `docker compose up` must boot everything with no manual setup. Data must survive
  `docker compose down && docker compose up` via the named volume.

BUILD (Phase 1 = skeleton, contract, Docker — a working naive end-to-end path):
- FastAPI app with a lifespan startup that loads the local embedder (offline), opens the asyncpg pool
  with register_vector on init, and runs idempotent migrations (CREATE EXTENSION IF NOT EXISTS vector;
  CREATE TABLE IF NOT EXISTS ...; CREATE INDEX IF NOT EXISTS ...).
- All SEVEN endpoints from §3 with EXACT shapes and status codes, using Pydantic models that mirror §3.
  /turns must accept single AND multi-message turns including a "tool"-role message with name null or
  set. Optional bearer auth: enforce Authorization only if env MEMORY_AUTH_TOKEN is set; ignore it
  otherwise; keep /health open.
- DB schema (idempotent migrations) with FIXED columns now (only data is added later):
  * turns, messages
  * memories: id, user_id, type (fact|preference|opinion|event), key (slot), value, confidence,
    source_session, source_turn, created_at, updated_at, supersedes (nullable FK), active (bool),
    embedding vector(384), search tsvector. Add a GIN index on search and an HNSW vector_cosine_ops
    index on embedding.
  * entities: id, user_id, entity_type (user|pet|employer|city), name, created_at.
  * memory_entities: memory_id, entity_id, relation. (entities/memory_entities are populated in a later
    phase but their columns are fixed NOW.)
- Create the package directories src/extraction/, src/recall/, src/search/ now (even if they hold only
  the thin naive helpers this phase) so later phases extend them in place rather than relocating code.
- Seed src/config.py with named tunables: EMBED_DIM=384 and the naive recall top_k. This file is the
  single tunable surface later phases extend.
- A NAIVE but HONEST store/recall so the system runs end to end: on POST /turns, persist the turn and its
  messages, create at least one minimally-structured TYPED memory record per turn (e.g. a typed `event`;
  never store a raw message chunk as a "memory"), embed its text, and return 201 {"id": ...}. On POST
  /recall, return the top-k memories by vector cosine (<=>), formatted as readable context like the §3
  example, with citations. CRITICAL SCOPING: a user's stored facts must be recallable across that user's
  sessions (so a write under session "smoke-1" is recalled under session "smoke-2" for the same user-1 —
  the smoke test depends on this); only "recent conversation" context is session-scoped. Different users
  must never bleed. Return {"context":"","citations":[]} with 200 when nothing matches. Implement
  /search, /users/{id}/memories, and the two DELETE endpoints. DELETE /sessions/{id} removes that
  session's data (204); DELETE /users/{id} cascades to memories+turns+sessions (204); both are idempotent
  (a non-existent id still returns 204).
- Structured logging of lifecycle events (startup, embedder load, auth enforced/skipped, extraction path).
- docker-compose.yml: an `app` service (port 8080) and a `db` service using `pgvector/pgvector:pg16`, ONE
  named volume for the DB, healthchecks on both, `depends_on` with `condition: service_healthy`.
  Dockerfile builds the app image and BAKES the embedding model into a fixed cache dir at build time
  (run a tiny encode during build to materialize it). .env.example listing optional ANTHROPIC_API_KEY and
  MEMORY_AUTH_TOKEN plus DB settings.

TESTS — create these named files and run `pytest`, fix until green:
- test_contract.py: POST a turn (include a multi-message turn with a "tool" message, name both null and
  non-null) then POST /recall; assert response JSON shapes AND explicit status codes match §3
  (/turns=201, /recall=200, /search=200, /users/{id}/memories=200); assert cold /recall and empty /search
  return 200 with empty payloads.
- test_persistence.py: write turns, restart the stack against the same volume, recall — data survives.
- test_isolation.py: two DIFFERENT users do not bleed; AND a same-user write in session A is recallable in
  session B (cross-session sharing works as designed).
- test_malformed.py: bad JSON, missing required fields, unicode oddities → 4xx (not 5xx, not a crash);
  /health still 200 afterward.
- test_auth.py: with MEMORY_AUTH_TOKEN unset, endpoints work with no header; with it set, missing/wrong
  token → 401/403 and correct token → normal status; /health open regardless.
- test_delete.py: write a user's data across two sessions; DELETE /sessions/{id} → 204 removes only that
  session; DELETE /users/{id} → 204 then /users/{id}/memories is empty and /recall returns empty-200;
  deleting a non-existent id still returns 204.
Also run the CHALLENGE.md §7 smoke test manually (the Berlin example with smoke-1/smoke-2) and confirm
/turns→201, /recall (session smoke-2, user-1) mentions Berlin, /users/user-1/memories returns structured
typed records.

ACCEPTANCE CRITERIA:
- Clean `docker compose up -d` + the §8 health-wait loop succeed on a machine with NO network egress
  (proves offline embedding load); /health is 200.
- The §7 smoke sequence passes as described (cross-session Berlin recall).
- All six named test files above are green.
- Default port 8080; data survives down+up; both auth branches and both DELETE cascades verified.

DO NOT introduce a deliberate flaw to fix later. Keep the contract and Docker boot sacred for all later
phases.

FINISH by:
1. Creating CLAUDE.md with FOUR sections: (a) Architecture — the locked stack; (b) Conventions — RECORD
   the pinned specifics so future sessions inherit them: image pgvector/pgvector:pg16; embedding model
   BAAI/bge-small-en-v1.5 at 384 dims baked offline into a fixed cache dir with EMBED_DIM in config;
   asyncpg + register_vector on pool init; cosine `<=>` and HNSW vector_cosine_ops; the single tunable
   surface src/config.py; the cross-session scoping rule (user facts shared across that user's sessions,
   recent context session-scoped, different users never bleed); how to run the service and tests.
   (c) File map — the actual files you created and what each does. (d) "## Status — done / next" — Phase 1
   done; Phase 2 (hybrid extraction + recall-quality fixture) next.
2. Writing a short README.md stub and appending an honest CHANGELOG.md entry "v0.1 — Skeleton, contract &
   Docker" describing what was built and why (single Postgres store, offline local embeddings, naive
   recall as the working baseline, cross-session scoping decision). Do not fabricate metrics — there is no
   recall metric yet.
````

---

### Phase 2 — Extraction & the Recall-Quality Fixture

**Goal.** Replace naive storage with **hybrid extraction** (Claude API + deterministic rule fallback)
that turns turns into **typed memories** (`fact`/`preference`/`opinion`/`event`) with confidence,
provenance, implicit-fact inference, and correction handling — so `/users/{id}/memories` returns clean
structured records, not message chunks. **Build the recall-quality fixture and a two-metric runner
here** — the measurement instrument the rest of the build depends on.

**In scope.**
- `extraction/pipeline.py` orchestrating two interchangeable paths into one normalized memory schema:
  - `llm.py`: an Anthropic Claude call that returns **structured JSON via a defined extraction tool +
    `tool_choice`** (read the result from the `tool_use` block; pick a current model id from the
    installed SDK, don't hardcode a possibly-retired string), extracting personal facts, preferences,
    opinions, corrections, and **implicit facts** ("walking Biscuit this morning" → pet named Biscuit).
  - `rules.py`: a deterministic regex/NLP fallback used when `ANTHROPIC_API_KEY` is missing **or the LLM
    call hangs/errors** (a generous ~20–30s timeout that fits inside the §3 60s `/turns` budget — not a
    trigger on normal latency); must still capture obvious facts (moves, named pets, job statements).
- Canonical **slot keys** (e.g. `employment`, `location`, `pet.name`, `diet`, `allergy`,
  `preference.answer_style`) so phase 4 supersession has a stable join key.
- Provenance on every memory (which path produced it; `source_session`, `source_turn`).
- Memories persisted with embedding + tsvector; `/users/{id}/memories` returns them typed. **`/turns`
  fully persists + extracts + embeds + indexes before returning 201 — no background tasks** (the §9
  synchronous-correctness guarantee).
- **`fixtures/`**: 3–5 scripted multi-session conversations + probe queries with expected facts,
  covering: basic recall, implicit facts (Biscuit), fact evolution (Stripe→Notion), multi-hop (dog→city,
  **with the two facts in different sessions/turns and no shared keywords** so it genuinely needs the hop),
  noise/empty (topic never discussed), and an opinion arc (TypeScript love→annoyance→nuance). Stored as
  data files, not hardcoded in tests.
- **`tests/fixture_runner.py`** reporting **two numbers**: an **extraction metric** ("X of Y expected
  facts present as typed memories in `/users/{id}/memories`") and a **recall-context metric** ("X of Y
  expected facts present in `/recall` context"), with a per-probe breakdown. Idempotent (cleans up via
  the DELETE endpoints between runs).

**Explicitly out of scope.** RRF / full-text fusion and tiered assembly (phase 3); supersession and the
entity/multi-hop population (phase 4). The fixture's evolution/multi-hop **recall** probes will score low
now — an honest baseline of features that do not exist yet, recorded in the CHANGELOG, **not** a planted
flaw.

**Definition of done (testable).**
- `test_extraction.py`: after ingesting fixture turns, `/users/{id}/memories` returns **typed** records
  (type ∈ the four) with non-empty slot key + value + confidence — asserted structured, not raw text;
  includes an **implicit-fact** case (Biscuit → `pet.name`) and a **correction** case.
- A **synchronous-correctness** assertion: POST `/turns`, then with **no delay/retry** the extracted
  memory appears in **both** `/users/{id}/memories` and `/recall`.
- A **fallback** assertion (in `test_extraction.py`): force the rule path by **running an app instance
  with `ANTHROPIC_API_KEY` unset** (a second instance / restarted service / a patched config seam — not
  by unsetting the env var inside the pytest process, which can't affect the running container), ingest a
  turn, assert rule-extracted memories appear and the service never crashes.
- `fixture_runner.py` runs end-to-end and prints **both** baseline metrics. Record the **actual**
  extraction-metric number in the CHANGELOG.
- **Re-run the §7 smoke** (Berlin) and confirm `/users/user-1/memories` now returns typed records and
  recall still mentions Berlin.
- **No regression**: all six phase 1 test files still pass.
- **Light measure-tune loop on extraction**: tune the extraction prompt / rules until the **extraction
  metric** (facts present as typed memories) stops improving. **Stop condition**: two consecutive
  prompt/rule edits yield no increase in the extraction metric. Each real round appends a CHANGELOG note
  with before/after.

**Artifacts created/updated.** `src/extraction/` (llm, rules, pipeline), updated `db/repository.py` and
`api/routes.py`, `fixtures/`, `tests/test_extraction.py`, `tests/fixture_runner.py`, `CLAUDE.md`
(status + file map), `CHANGELOG.md`.

**Copy-paste prompt:**

````
You are a fresh Claude Code session. Build PHASE 2 of a Dockerized "memory service".

FIRST, orient yourself from disk (do not assume prior state from any memory):
1. Read CLAUDE.md fully — architecture, conventions (image, embedding model + EMBED_DIM, asyncpg +
   register_vector, cosine/HNSW, the single tunable surface, the cross-session scoping rule), file map,
   and "Status — done / next".
2. Read CHALLENGE.md fully — focus on §4 ("Extraction, not just storage": personal facts, preferences,
   opinions, corrections, implicit facts), the §3 shape of GET /users/{id}/memories and the /turns 60s
   budget, §5/§9 synchronous correctness ("if you wrote it, you can read it"), §7 (required recall-quality
   fixture), §9/§10 (extraction quality is graded; raw chunks are a red flag).
3. Read the ACTUAL code on disk: the thin src/extraction/ and src/recall/ packages, the memories table
   schema in the migrations, db/repository.py, api/routes.py, and src/config.py. Build on what exists;
   discover real module and column names rather than assuming them.

GOAL: turn raw turns into TYPED, structured memories via HYBRID extraction (synchronously), and build the
recall-quality fixture + a two-metric runner the rest of the project iterates against.

BUILD (locked design — implement, do not replace):
1. Hybrid extraction pipeline:
   - Anthropic Claude path: read a turn's messages and return STRUCTURED JSON by defining an extraction
     TOOL and setting tool_choice to force it; read the result from the tool_use content block. Pick a
     CURRENT model id from the installed anthropic SDK (do not hardcode a possibly-retired string). Each
     memory has: type (fact|preference|opinion|event), a canonical slot key, a value, confidence in [0,1],
     and provenance (source_session, source_turn, which path). Extract personal facts (employment,
     location, family, pets), preferences/opinions, corrections ("actually I meant…", "sorry, not X — Y"),
     and IMPLICIT facts ("walking Biscuit this morning" → pet.name=Biscuit).
   - Deterministic RULE fallback (regex / lightweight NLP) that runs when ANTHROPIC_API_KEY is missing OR
     the Claude call hangs/errors. Use a GENEROUS timeout (~20–30s, inside the 60s /turns budget) — the
     fallback is a safety net for failures/hangs, NOT a trigger on normal LLM latency; when the key is
     present and the call succeeds in budget, USE the LLM result.
   - Both paths normalize into the SAME memory schema already in the DB. Persist memories with their
     embedding (offline local model) and tsvector. Never store a raw message chunk as a "memory".
   - Use canonical slot keys (employment, location, pet.name, diet, allergy, preference.answer_style, …)
     so a later phase can supersede same-slot facts. Keep a small documented slot-key list.
   - SYNCHRONOUS: POST /turns must persist + extract + embed + index BEFORE returning 201. No background
     tasks, no eventual consistency.
2. fixtures/: 3–5 scripted MULTI-SESSION conversations plus probe queries, each probe listing its expected
   fact(s). Cover: (a) basic recall; (b) an implicit fact (a named pet); (c) fact evolution ("I work at
   Stripe" early, "I just joined Notion" later — current answer Notion); (d) MULTI-HOP ("what city does the
   user with the dog named Biscuit live in?") — put the pet fact and the city fact in DIFFERENT
   sessions/turns with NO shared keywords so answering genuinely requires connecting two separate memories;
   (e) noise resistance (a query about a topic never discussed — expected context empty); (f) an opinion arc
   (TypeScript love → annoyance → nuanced). Store fixtures as data files, not hardcoded in tests.
3. tests/fixture_runner.py: ingest every fixture conversation via POST /turns, then report TWO metrics with
   a per-probe breakdown: an EXTRACTION metric ("X of Y expected facts present as typed memories in
   /users/{id}/memories") and a RECALL-CONTEXT metric ("X of Y expected facts present in /recall context").
   Make it repeatable and idempotent (clean up via the DELETE endpoints between runs).

TESTS (run with `pytest`, fix until green):
- tests/test_extraction.py: after ingesting fixture turns, GET /users/{id}/memories returns typed records
  (type ∈ {fact,preference,opinion,event}) with non-empty key+value+confidence — assert structured, not
  raw text. Include an implicit-fact case (Biscuit → pet.name) and a correction case. Add a
  SYNCHRONOUS-CORRECTNESS assertion: POST /turns then with NO delay assert the memory is in BOTH
  /users/{id}/memories AND /recall. Add a FALLBACK assertion: force the rule path by running an app
  instance with ANTHROPIC_API_KEY UNSET (a second instance / restarted service / a patched config seam —
  NOT by unsetting the env var inside the pytest process, which cannot affect the running container);
  assert rule-extracted memories still appear and nothing crashes.
- Re-run the CHALLENGE.md §7 smoke sequence (Berlin, smoke-1→smoke-2) and confirm /turns→201, /recall
  mentions Berlin, /users/user-1/memories now returns typed structured records.
- Re-run the phase 1 tests (test_contract, test_persistence, test_isolation, test_malformed, test_auth,
  test_delete) and confirm they STILL PASS.

MEASURE-TUNE LOOP (you run it yourself this session):
- Boot the service, run tests/fixture_runner.py, and read which expected facts were NOT captured as typed
  memories (use the EXTRACTION metric and the per-probe breakdown). Improve the extraction prompt OR the
  rules — change ONE thing — and re-run. STOP when two consecutive changes yield no increase in the
  extraction metric. NOTE: the evolution and multi-hop RECALL probes will still fail because those features
  are built later — that is expected; judge extraction by the EXTRACTION metric, not recall ranking yet.

ACCEPTANCE CRITERIA:
- /users/{id}/memories returns typed, structured memories with confidence + provenance (test_extraction.py
  green), including implicit-fact and correction cases.
- Synchronous-correctness assertion passes (memory queryable with no delay after /turns).
- Fallback assertion passes with the key unset; service never crashes.
- fixture_runner.py prints both baseline metrics; the §7 smoke still passes.
- All six phase 1 test files still pass.

DO NOT regress phase 1: the seven endpoints, shapes/status codes, Docker boot, persistence, isolation, auth,
deletes, and cross-session scoping must keep working. Do not introduce a deliberate flaw to fix later.

FINISH by:
1. Updating CLAUDE.md: extend the file map with the new modules + fixture/runner + slot-key list; update
   "Status — done / next" (Phase 2 done; Phase 3 = hybrid recall + RRF + tiered context assembly next).
2. Appending an honest CHANGELOG entry "v0.2 — Hybrid extraction + recall-quality fixture" describing the
   extraction design and recording the REAL baseline EXTRACTION metric (X of Y). If you ran tuning rounds,
   append one note per round with the actual before/after counts. Never fabricate a number or invent a fake
   mistake.
````

---

### Phase 3 — Hybrid Recall & Context Assembly

**Goal.** Make recall real and good — the **primary eval signal**. Retrieve with **pgvector (semantic) +
Postgres full-text (keyword)** fused by **Reciprocal Rank Fusion**, then assemble `/recall` context in
**priority tiers** (stable user facts → query-relevant → recent) inside `max_tokens`. Make `/search`
structured. Then **run the fixture, record the first real recall metric, and tune** until it plateaus.

**In scope.**
- `recall/retrieval.py`: semantic top-N via pgvector cosine (`<=>`) + keyword top-N via
  `tsvector`/`ts_rank` (`websearch_to_tsquery`). Tier-1 user facts are fetched **cross-session for the
  user**; the recent tier is session-scoped (per the Phase-1 scoping rule).
- `recall/fusion.py`: RRF (`score = Σ 1/(k + rank)`, `k≈60` tunable) over the two ranked lists — **not**
  an average of raw cosine/ts_rank scores.
- `recall/assembly.py`: tiered, token-budgeted builder — **Tier 1** stable high-confidence **active**
  user facts/preferences; **Tier 2** query-relevant fused results; **Tier 3** recent session context —
  formatted like the §3 example (reproduce its two section headers), with citations
  (`turn_id`, `score`, `snippet`). **Design Tier 1 from the start to also fetch the immediately-superseded
  value for a slot when present**, so it can render the §3 "updated …; previously …" annotation once
  Phase 4 sets `active`/`supersedes` — this avoids a Phase-3↔Phase-4 conflict between "read only active"
  and "show that it evolved." Token budget via the conservative over-counting heuristic
  (`max(words×1.3, chars/4)` or a real token counter if present); trim to stay ≤ ~1×, never > 2×; Tier 1
  wins when budget is tight.
- **Noise resistance**: a relevance/score threshold so irrelevant queries return
  `{"context":"","citations":[]}` with 200 — no hallucinated memories.
- `search/search.py`: structured `/search` (content, score, session_id, timestamp, metadata), honoring
  `limit` and `session_id`/`user_id` scoping.
- Add the new tunables (RRF `k`, per-source top-N, tier token fractions, score thresholds, snippet length)
  to the **existing** `src/config.py` alongside the Phase-1 `top_k`.

**Explicitly out of scope.** Supersession and entity/multi-hop population (phase 4). Final global tuning
and docs (phase 5).

**Definition of done (testable).**
- `fixture_runner.py` reports the **first real recall-context metric**. Record the **actual** number in
  the CHANGELOG.
- **Measure-tune loop run by this session**: boot, run the fixture, read which probes returned the wrong
  facts and why (semantic miss? keyword miss? wrong tier? budget cut it?), **change ONE tunable**, re-run.
  **Stop condition**: two consecutive single-variable changes fail to improve the metric **without
  regressing the noise/empty probes**. Every real round → one CHANGELOG note with before/after.
- `test_recall.py`: a tight-`max_tokens` test (estimated output ≤ ~1× budget, never > 2×, including a
  unicode-heavy input; Tier-1 facts appear before recent chatter); a **noise** test (irrelevant query →
  empty context); a `/search` shape test (structured, correct scoping, respects `limit`); a **coarse
  recall-latency** assertion (`/recall` on the fixture returns under a documented threshold on the dev box).
- **No regression**: all phase 1 + phase 2 tests still pass; the §7 smoke still passes; extraction unchanged.

**Artifacts created/updated.** `src/recall/` (retrieval, fusion, assembly), `src/search/search.py` (filled
in / migrated from the Phase-1 inline path), tunables in `src/config.py`, `tests/test_recall.py`,
`CLAUDE.md` (status + file map), `CHANGELOG.md` (real recall metric + tuning rounds).

**Copy-paste prompt:**

````
You are a fresh Claude Code session. Build PHASE 3 of a Dockerized "memory service".

FIRST, orient yourself from disk (do not assume prior state from memory):
1. Read CLAUDE.md fully (architecture, conventions, file map, Status — note the cross-session scoping rule,
   EMBED_DIM/384, cosine `<=>`, the single tunable surface src/config.py).
2. Read CHALLENGE.md fully — focus on §3 POST /recall behavior (priority order: stable user facts →
   query-relevant → recent; respect max_tokens, never blow past ~2×; empty 200 on cold/irrelevant; the §3
   example context format with "## Known facts about this user" and "## Relevant from recent
   conversations"), §3 POST /search (structured), §4 "Context assembly under budget", §5 recall-budget/
   latency, §9/§10 (recall is the PRIMARY signal; vanilla cosine-top-k will not score).
3. Read the ACTUAL code on disk: the current recall code in src/recall/ (and any /search code, which Phase
   1 may have implemented inline in api/routes.py — MOVE it into src/search/search.py and re-wire routes,
   do not create a parallel duplicate), the memories schema (embedding vector(384) + tsvector), the
   repository layer, the extraction output, and especially fixtures/ + tests/fixture_runner.py — you will
   iterate against that runner. Find the existing config and its tunable names; ADD new tunables there.

GOAL: replace naive recall with HYBRID retrieval + RRF + tiered token-budget assembly, make /search
structured, then MEASURE and TUNE recall quality on the fixture until it plateaus.

BUILD (locked design — implement, do not replace):
1. Retrieval: semantic top-N via pgvector cosine (`<=>`) AND keyword top-N via Postgres full-text
   (tsvector / ts_rank, e.g. websearch_to_tsquery). Tier-1 user facts are fetched CROSS-SESSION for the
   user; the recent tier is session-scoped (per the CLAUDE.md scoping rule).
2. Fusion: Reciprocal Rank Fusion — score = sum of 1/(k + rank) with k a tunable (~60). Do NOT average raw
   cosine and ts_rank scores; use rank-based RRF.
3. Context assembly for /recall, within max_tokens, in PRIORITY TIERS: Tier 1 = stable, high-confidence,
   ACTIVE user facts/preferences; Tier 2 = query-relevant fused results; Tier 3 = recent session context.
   Reproduce the §3 example's two section headers verbatim. Return citations (turn_id, score, snippet).
   Design Tier 1 NOW to also fetch the immediately-superseded value for a slot when present, so it can
   render an "updated …; previously …" annotation once a later phase sets active/supersedes (this avoids a
   conflict between "read only active" and "show that it evolved"). Enforce the budget with a CONSERVATIVE
   over-counting token estimate (tokens ≈ max(words×1.3, chars/4), or a real token counter if available);
   trim to keep the estimate ≤ ~1× and NEVER > 2×, even on unicode-heavy input. Tier 1 wins when budget is
   tight.
4. Noise resistance: a relevance/score threshold so a query about a topic never discussed returns
   {"context":"","citations":[]} with status 200 — never hallucinate.
5. /search: structured results (content, score, session_id, timestamp, metadata), honoring `limit` and
   session_id/user_id scoping. Different shape from /recall (structured vs prose).
6. Add EVERY new tunable (RRF k, per-source top-N, tier token fractions, score thresholds, snippet length)
   to the EXISTING src/config.py alongside the Phase-1 top_k, so tuning means changing ONE named value.

TESTS (run with `pytest`, fix until green):
- tests/test_recall.py: a tight-max_tokens test (estimated output ≤ ~1× budget, never > 2×, include a
  unicode-heavy input; Tier-1 facts appear before recent chatter); a noise test (irrelevant query → empty
  context); a /search shape test (structured, correct scoping, respects limit); a coarse recall-latency
  assertion (/recall on the fixture returns under a documented threshold).
- Re-run ALL earlier tests (test_contract, test_persistence, test_isolation, test_malformed, test_auth,
  test_delete, test_extraction incl. the fallback case) and the §7 smoke; confirm they STILL PASS.

MEASURE-TUNE LOOP (you MUST run this yourself — the core of the phase):
- Boot the service, run tests/fixture_runner.py to get the FIRST real recall-context metric, and read the
  per-probe breakdown to see WHICH probes returned the wrong facts and WHY (semantic miss? keyword miss?
  wrong tier? budget cut it?). Change exactly ONE tunable in config, re-run, compare. Keep the change only
  if the metric improved WITHOUT regressing the noise/empty probes; revert otherwise. STOP when two
  consecutive single-variable changes fail to improve the metric without regressing the noise/empty probes.
  Note: evolution and multi-hop probes may still fail — those arrive in phase 4; tune for the recall, noise,
  and basic/implicit-fact probes now.

ACCEPTANCE CRITERIA:
- fixture_runner.py prints a real recall-context metric; test_recall.py (budget incl. unicode, noise,
  /search shape, latency) is green; noise/empty probe returns empty; the measure-tune loop reached its stop
  condition with rounds recorded.
- All phase 1–2 tests still pass and the §7 smoke still passes.

DO NOT regress phases 1–2. Do not introduce a deliberate flaw to fix later.

FINISH by:
1. Updating CLAUDE.md: extend the file map (recall/, search/, new config tunables) and "Status — done /
   next" (Phase 3 done; Phase 4 = fact evolution/supersession + entity layer + multi-hop next).
2. Appending an honest CHANGELOG entry "v0.3 — Hybrid recall (pgvector + full-text + RRF) and tiered
   assembly" recording the REAL first recall metric, plus one note per actual tuning round with the true
   before/after numbers and the single change you made. Never fabricate a number or a fake mistake.
````

---

### Phase 4 — Fact Evolution & Multi-Hop

**Goal.** Add the two hardest grading categories on a measurable base. Implement **slot-based
supersession** (detect same-topic facts, mark the old one superseded — never deleted — return the
current one, preserve the chain) and a **small entity layer + query decomposition** so multi-hop
questions connect facts that live apart. **Run the fixture and tune the supersession threshold against
real cases.**

**In scope.**
- `evolution/supersession.py`: on a new memory whose slot key matches an existing **active** memory for
  the same user (and, for fuzzy cases, whose embedding similarity exceeds a threshold), set the old memory
  `active=false`, set the new memory's `supersedes` pointer to it, **advance `updated_at` on the chain
  head**, and keep the old row. Handles §4 contradictions (Stripe→Notion, NYC→Berlin) and corrections.
- **Same-topic detection**: primarily exact slot-key match; for cases the slot key misses, an
  embedding-similarity threshold decides "same slot." **This threshold is the tuning target.**
- **Opinion arcs**: opinions are not hard-overwritten — keep a per-topic stance history; recall surfaces
  the latest stance and notes that it evolved, reusing the Tier-1 "updated …; previously …" rendering that
  Phase 3 already designed for. Partial is acceptable — **document** it.
- `entities/entities.py`: populate the existing `entities` (user/pet/employer/city) and `memory_entities`
  tables from extracted memories. `entities/decompose.py`: query decomposition for multi-hop ("city of the
  user whose dog is Biscuit" → resolve the pet entity → its owner → the owner's city) — connecting facts
  that are NOT co-retrievable by the query terms alone.
- Wire results back into recall: Tier 1 reflects supersession (only `active`, plus the prior value for the
  annotation); multi-hop answers are assembled via entity links.

**Explicitly out of scope.** Final global tuning and the README (phase 5). Destructive migrations — the
Phase-1 schema already has the `memories`, `entities`, and `memory_entities` columns; add only additive,
idempotent columns if truly needed.

**Definition of done (testable).**
- `test_evolution.py`: ingest Stripe-then-Notion; `/recall` returns **Notion** (current), not Stripe;
  `/users/{id}/memories` shows the **chain** (old `active=false`, new `active=true` with a `supersedes`
  pointer, **`updated_at` advanced**); the superseded row **still exists** (history preserved). Assert the
  `/recall` context renders the "updated …; previously …" presentation from the §3 example. Add the
  NYC→Berlin case.
- `test_multihop.py`: ingest the dog-and-city scenario **with the two facts in different sessions/turns and
  no shared keywords**; assert `/recall` returns the correct city; include a **control assertion** that a
  vanilla top-k over the query terms alone would miss the city memory (so the test proves the decomposition
  path, not incidental co-retrieval). State the expected city explicitly.
- `fixture_runner.py` overall metric is **re-run and the actual number recorded** (it is expected to rise
  as evolution/multi-hop probes now pass — but record the real value whatever it is; never force a number).
- **Supersession threshold tuned by this session**: sweep the threshold one step at a time, inspecting
  false-merges (unrelated facts wrongly superseded) vs missed-merges (a contradiction left as two active
  facts). **Stop condition**: two consecutive single-step adjustments fail to improve the net (evolution +
  multi-hop probes correct AND no unrelated facts wrongly merged); lock the prior best. One CHANGELOG note
  per round with before/after.
- **No regression**: all earlier tests pass; the noise/empty probe is still empty; the §7 smoke still passes.

**Artifacts created/updated.** `src/evolution/supersession.py`, `src/entities/` (entities, decompose),
additive migrations only if needed, recall wiring updates, `tests/test_evolution.py`,
`tests/test_multihop.py`, `CLAUDE.md` (status + file map), `CHANGELOG.md`.

**Copy-paste prompt:**

````
You are a fresh Claude Code session. Build PHASE 4 of a Dockerized "memory service".

FIRST, orient yourself from disk (do not assume prior state from memory):
1. Read CLAUDE.md fully (architecture, conventions, file map, Status).
2. Read CHALLENGE.md fully — focus on §4 "Fact evolution and contradiction handling" (detect same topic;
   store new as active, mark old superseded NOT deleted; return current from /recall; preserve history in
   /users/{id}/memories; the §3 example shows the "updated 2025-03-15; previously at Stripe" presentation;
   opinion arcs are harder and may be partial-but-documented) and §9 (Fact evolution + Multi-hop are
   explicit graded categories; "what city does the user with the dog named Biscuit live in" is the
   canonical multi-hop example).
3. Read the ACTUAL code on disk: the memories table (slot key, supersedes, active, updated_at columns), the
   entities + memory_entities tables (their columns are already fixed), the extraction output and its slot
   keys, the recall assembly (especially how Tier 1 selects facts and whether it already fetches the prior
   superseded value for the annotation), and fixtures/ + tests/fixture_runner.py. Discover real column and
   module names; build on what exists. Prefer additive, idempotent schema changes — do NOT break the
   existing schema or data.

GOAL: implement slot-based fact supersession (history preserved) and a small entity layer + query
decomposition for multi-hop recall, then TUNE the supersession threshold against the fixture's real cases.

BUILD (locked design — implement, do not replace):
1. Supersession (evolution/): when a newly extracted memory has the SAME slot key as an existing ACTIVE
   memory for the same user — or, for fuzzy cases the slot key misses, when its embedding similarity to an
   active same-category memory exceeds a TUNABLE threshold — set the OLD memory active=false, set the NEW
   memory's supersedes pointer to it, advance updated_at on the chain head, and keep the old row (never
   delete). Handle §4 contradictions (Stripe→Notion, NYC→Berlin) and explicit corrections. Recall's Tier 1
   reads only active facts for the current value, but ALSO surfaces the immediately-superseded value as an
   "updated …; previously …" annotation (reuse the Tier-1 path Phase 3 designed for this).
2. Opinion arcs: do NOT hard-overwrite opinions. Keep a per-topic stance history; let recall surface the
   latest stance and note it evolved. Partial is acceptable — but DOCUMENT precisely how you handle it.
3. Entity layer (entities/): populate the existing entities (user/pet/employer/city) and memory_entities
   tables from extracted memories. Add query decomposition (decompose.py): for a multi-hop question,
   decompose into sub-lookups over the entity links (resolve the pet named Biscuit → its owner → the
   owner's city) and wire the connected facts into recall — facts that are NOT co-retrievable by the query
   terms alone.

TESTS (run with `pytest`, fix until green):
- tests/test_evolution.py: ingest "I work at Stripe" then later "I just joined Notion"; assert /recall
  returns Notion not Stripe; assert /users/{id}/memories shows old active=false and new active=true with a
  supersedes pointer and an advanced updated_at; assert the superseded row STILL EXISTS; assert the /recall
  context renders the "updated …; previously …" presentation. Add NYC→Berlin.
- tests/test_multihop.py: ingest the dog-and-city scenario with the pet fact and the city fact in DIFFERENT
  sessions/turns and NO shared keywords; assert /recall returns the correct city (state which city). Add a
  CONTROL assertion that a vanilla top-k over the query terms alone would miss the city memory — so the
  test proves the decomposition path, not incidental co-retrieval.
- Re-run ALL earlier tests (contract, persistence, isolation, malformed, auth, delete, extraction incl.
  fallback, recall incl. noise/search/latency) and the §7 smoke. Confirm they STILL PASS and the noise
  probe is still empty.

MEASURE-TUNE LOOP (you MUST run this yourself):
- Boot the service, run tests/fixture_runner.py, and look at the evolution + multi-hop probes plus any
  unrelated facts wrongly superseded. Sweep the supersession similarity threshold ONE step at a time: too
  low → unrelated facts wrongly merged (false-merge); too high → real contradictions stay as two active
  facts (missed-merge). STOP when two consecutive single-step adjustments fail to improve the net
  (evolution + multi-hop probes correct AND no unrelated facts wrongly merged); lock the prior best.
  Re-run the runner and record the real overall metric whatever it is — never force a number.

ACCEPTANCE CRITERIA:
- test_evolution.py and test_multihop.py green (current fact returned, chain + updated_at + history
  inspectable, "previously" rendered, multi-hop proven via the control assertion).
- Supersession threshold tuning reached its stop condition with rounds recorded; overall fixture metric
  re-run and truthfully recorded.
- All earlier tests + the §7 smoke still pass; noise probe still empty.

DO NOT regress phases 1–3. Prefer additive, idempotent migrations. Do not introduce a deliberate flaw.

FINISH by:
1. Updating CLAUDE.md: extend the file map (evolution/, entities/, decompose) and "Status — done / next"
   (Phase 4 done; Phase 5 = hardening + final global tuning + README/CHANGELOG finalization next).
2. Appending an honest CHANGELOG entry "v0.4 — Fact evolution (slot supersession) + multi-hop via entity
   layer" recording the real fixture metric and one note per actual threshold-tuning round with true
   before/after numbers and the single change you made. Never fabricate a number or a fake mistake.
````

---

### Phase 5 — Hardening, Final Tuning & Docs

**Goal.** Turn a working system into a *shippable* one: a **robustness pass** so the service never crashes
on bad input or a missing key (errors are 4xx/5xx, service stays up), **one final global tuning pass** to
lock the best ranking config, and the **README.md + finalized CHANGELOG.md** that carry ~50% of the grade.
Confirm the repo exactly matches the required §6 structure.

**In scope.**
- **Robustness**: malformed JSON, missing fields, **oversized payloads** (body-size limit → 413/422),
  unicode oddities (emoji, RTL, NUL, long graphemes), **missing `ANTHROPIC_API_KEY`** (fallback),
  **missing-auth-when-token-set**, **cold/empty store** (empty recall), and **restart mid-write**
  (transactions + idempotent startup so a partial write can't corrupt the store). A global exception
  handler ensures every failure is a 4xx/5xx, never an uncaught crash; `/health` stays 200.
- **Observability**: confirm/extend structured logging of key lifecycle events — extraction path chosen
  (LLM vs rule fallback), key-missing degradation, supersession events, recall tier decisions — so the
  reviewer can see the system reason (§10 "well-logged").
- **Final global tuning**: one pass over all tunables (RRF `k`, per-source top-N, tier token splits, score
  thresholds, supersession threshold) on the full fixture; lock the best config; record the final metric.
- **`README.md`** with all 8 §6 sections: architecture diagram + paragraphs; backing-store choice & why;
  extraction pipeline (what/how/what it misses); recall strategy (ranking, token budget, priority logic);
  fact evolution (contradictions/corrections/opinion arcs); tradeoffs; failure modes (no data / slow disk /
  missing keys) **incl. the measured recall latency**; how to run the tests. Document the **cross-session
  same-user sharing** decision (§5/§9) and a short **originality note** (§11).
- **CHANGELOG finalization**: ensure ≥4 honest substantive entries with real before/after metrics reading
  as a coherent narrative.
- **Structure check**: repo matches §6 exactly.

**Explicitly out of scope.** New features. Anything that regresses earlier phases. Production
multi-tenancy / horizontal-scale work (§12 out of scope).

**Definition of done (testable).**
- `test_robustness.py` passes: each bad-input class (malformed JSON, missing fields, oversized body,
  unicode) → 4xx/5xx and `/health` still 200 afterward; **missing-auth-when-token-set** → 401/403.
- **Missing-key test**: an app instance with `ANTHROPIC_API_KEY` unset (a separately launched/restarted
  instance, not an in-process unset) runs the full fixture via the fallback with no crash.
- **Restart-mid-write resilience** (concrete): start ingesting N turns, kill/restart the app container
  mid-batch against the same volume, then assert `/health` 200, that previously-committed turns are
  recallable, and that **no half-written memory row exists** (every memory row has its required columns
  non-null) — i.e. each `/turns` is one all-or-nothing transaction.
- **Final fixture metric** re-run and recorded; if it is not ≥ phase 4's, explain why in the CHANGELOG
  rather than tuning to hit a target. **Stop condition** for the final tune: two consecutive single-variable
  changes fail to improve the metric without regressing the noise/empty probe (cap at one full sweep of the
  tunable list).
- `README.md` has all 8 sections + the cross-session decision + originality note; `CHANGELOG.md` has ≥4
  substantive entries with real metrics.
- Repo structure matches §6 exactly; on a clean checkout (no network) `docker compose up -d` + the §8
  health-wait loop succeed and the full `pytest` suite + the §7 smoke test are green.

**Artifacts created/updated.** Hardening across `src/` (global exception handler, body-size limit,
transactional writes, logging), `tests/test_robustness.py`, locked tunables in `src/config.py`,
**`README.md` (final)**, **`CHANGELOG.md` (final)**, `CLAUDE.md` (final status).

**Copy-paste prompt:**

````
You are a fresh Claude Code session. Build PHASE 5 (final) of a Dockerized "memory service".

FIRST, orient yourself from disk (do not assume prior state from memory):
1. Read CLAUDE.md fully (architecture, conventions, file map, Status).
2. Read CHALLENGE.md fully — focus on §5 (hard constraints: persistence, concurrent sessions, synchronous
   correctness, resilience — "must not crash on malformed input, oversized payloads, or unicode oddities"),
   §6 (required repo structure + the 8 mandatory README sections; CHANGELOG is the most important
   deliverable), §7/§8 (smoke test + the exact clean-machine setup), §9 (Robustness + Persistence +
   Cross-session scoping graded categories), §10 ("what excellent looks like"), §11 (originality).
3. Read the ACTUAL code on disk: every endpoint, the extraction pipeline (LLM + fallback), the recall
   pipeline (retrieval, RRF, assembly), the evolution/entity layers, the central tunable config
   (src/config.py), the full tests/ suite, and fixtures/ + tests/fixture_runner.py. Read the existing
   CHANGELOG.md so the new entries continue its narrative honestly.

GOAL: harden the service so it never crashes, do ONE final global tuning pass to lock the best ranking
config, and write the README.md + finalize CHANGELOG.md. Add NO new features.

BUILD / HARDEN:
1. Robustness — the service must STAY UP and return 4xx/5xx (never an uncaught 500 crash) for: malformed
   JSON; missing/null required fields; OVERSIZED payloads (enforce a request body-size limit → 413 or 422);
   unicode oddities (emoji, RTL, NUL, very long graphemes); MISSING ANTHROPIC_API_KEY (rule fallback);
   MISSING/WRONG auth header when MEMORY_AUTH_TOKEN is set (401/403); a COLD/empty store (recall returns
   {"context":"","citations":[]} with 200); and RESTART MID-WRITE (wrap each /turns ingest in ONE
   transaction and keep startup idempotent so a partial write cannot corrupt the store). Add a global
   exception handler. After any error, /health must still be 200.
2. Observability: confirm/extend structured logging of lifecycle events — extraction path chosen (LLM vs
   rule fallback), key-missing degradation, supersession events, recall tier decisions.
3. Final global tuning: run tests/fixture_runner.py, then do ONE careful pass over the central tunables
   (RRF k, per-source top-N, tier token splits, score thresholds, supersession threshold). Change one at a
   time, re-run, keep improvements that do not regress the noise/empty probe. STOP when two consecutive
   single-variable changes fail to improve the metric without regressing the noise/empty probe (cap at one
   full sweep of the tunable list). LOCK the best config and record the final metric. If the final metric
   is not ≥ phase 4's, explain why in the CHANGELOG — do NOT tune to hit a target or fabricate a number.

DOCS:
4. Write README.md with ALL EIGHT sections required by §6: (1) Architecture — an ASCII or Mermaid diagram
   + 1–2 paragraphs; (2) Backing-store choice and why (single Postgres+pgvector for vector + full-text +
   relational history); (3) Extraction pipeline — what you extract, how (hybrid Claude + rule fallback),
   what you miss and why; (4) Recall strategy — retrieval, RRF, token-budget priority tiers and the logic
   when budget is tight; (5) Fact evolution — contradictions, corrections, opinion arcs; (6) Tradeoffs —
   what you optimized for and gave up; (7) Failure modes — no data, slow disk, missing API keys, INCLUDING
   the measured recall latency and what you'd optimize; (8) How to run the tests. Also document the
   intentional CROSS-SESSION same-user fact sharing (per §5/§9) and add a short originality note (§11):
   this is our own design, not a clone of mem0/honcho/etc. Keep it skimmable — a reviewer should grasp the
   design in 5 minutes.
5. Finalize CHANGELOG.md: ensure ≥4 honest, substantive entries reading as a coherent iteration story
   (phase milestones + real tuning rounds with true before/after metrics). Do not fabricate numbers or
   invent fake mistakes — use the real ones already recorded plus this phase's final tuning result.

TESTS (run with `pytest`, fix until green):
- tests/test_robustness.py: one assertion per bad-input class above → 4xx/5xx and /health still 200;
  missing-auth-when-token-set → 401/403.
- A missing-key test: an app instance with ANTHROPIC_API_KEY unset (separately launched/restarted, not an
  in-process unset) runs the full fixture via the fallback with no crash.
- A restart-mid-write test (concrete): start ingesting N turns, kill/restart the app mid-batch against the
  same volume, then assert /health 200, previously-committed turns are recallable, and no half-written
  memory row exists (required columns non-null).
- Re-run the ENTIRE suite (contract, persistence, isolation, malformed, auth, delete, extraction incl.
  fallback, recall incl. noise/search/latency, evolution, multi-hop, robustness) AND the §7 smoke on a
  fresh, network-isolated `docker compose up`. Everything must be green.

VERIFY STRUCTURE: confirm the repo exactly matches §6 — README.md, CHANGELOG.md, docker-compose.yml,
Dockerfile, src/, tests/, fixtures/, .env.example all present and correct. Confirm the §8 clean-machine
flow works: `docker compose up -d` then `until curl -sf localhost:8080/health; do sleep 1; done`.

ACCEPTANCE CRITERIA:
- test_robustness.py green; missing-key and restart-mid-write tests green; the whole suite + the §7 smoke
  green on a clean, network-isolated boot.
- README.md has all 8 sections + the cross-session decision + originality note; CHANGELOG has ≥4
  substantive entries with real metrics; final metric recorded honestly.
- Repo structure matches §6 exactly.

DO NOT regress phases 1–4: every endpoint/shape/status code, Docker boot, persistence, isolation, auth,
deletes, extraction output, recall quality, noise resistance, /search, fact evolution, and multi-hop must
all keep working. Do not introduce a deliberate flaw.

FINISH by:
1. Updating CLAUDE.md "Status — done / next": mark Phase 5 done and the product SHIPPABLE; note the final
   locked config and final fixture metric.
2. Appending the final honest CHANGELOG entry "v1.0 — Hardening + final tuning + docs" with the final metric
   and what the final tuning pass changed. Confirm the CHANGELOG has ≥4 substantive entries.
````

---

## 3. Closing note — repo matches the required structure after Phase 5

After Phase 5 the repository matches the `CHALLENGE.md` §6 required structure exactly:

| §6 requirement | Produced by |
|---|---|
| `README.md` (8 mandatory sections + cross-session decision + originality note) | Phase 1 stub → **Phase 5 final** |
| `CHANGELOG.md` (≥4 honest substantive entries with real metrics) | Appended every phase + every real tuning round; **finalized Phase 5** |
| `docker-compose.yml` (app + `pgvector/pgvector:pg16`, named volume, port 8080, healthchecks) | **Phase 1**, unchanged after |
| `Dockerfile` (service image; bakes the bge-small model offline at build) | **Phase 1** |
| `src/` (api, db, embeddings, extraction, recall, search, evolution, entities, config) | Phases **1–4**, hardened **5** |
| `tests/` (test_contract, test_persistence, test_isolation, test_malformed, test_auth, test_delete, test_extraction, test_recall, test_evolution, test_multihop, test_robustness, fixture_runner) | Grown each phase |
| `fixtures/` (3–5 scripted conversations + probe queries with expected facts) | **Phase 2**, run after every later change |
| `.env.example` (`ANTHROPIC_API_KEY`, `MEMORY_AUTH_TOKEN`, DB vars) | **Phase 1** |
| `CLAUDE.md` (cross-session anchor; not required by §6 but central to this build) | **Phase 1**, updated every phase |

**Grading coverage.** Contract compliance (exact shapes/status codes for all 7 endpoints, optional bearer
auth, multi-message/tool turns, DELETE cascade + idempotency) + Docker boot + persistence + cross-session
scoping (same-user sharing, different-user isolation) + malformed input → **Phase 1**; extraction quality +
synchronous correctness + the two-metric measurement fixture → **Phase 2**; recall quality (the primary
signal) + budget assembly + noise resistance + structured search + recall latency → **Phase 3**; fact
evolution (supersession chain, history, "previously" rendering) + multi-hop (proven via a control
assertion) → **Phase 4**; robustness (malformed/oversized/unicode/missing-key/missing-auth/cold/restart-
mid-write) + observability + final tuning + README/CHANGELOG for the human review → **Phase 5**.

Every phase leaves the system working, re-runs all earlier tests plus the §7 smoke so it never regresses an
earlier phase, and the CHANGELOG records only real milestones and real measured tuning rounds — the 4+
substantive entries the challenge rewards arise naturally, with no fabricated numbers and no invented
mistakes. The architecture stays exactly as locked; the review only tightened test coverage, cross-phase
coherence, and the technical specifics a fresh session needs to implement it correctly on the first try.
