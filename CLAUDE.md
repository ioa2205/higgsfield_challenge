# CLAUDE.md — Memory Service

Cross-session anchor for any future session. Read this first, then `CHALLENGE.md`,
then the code on disk. Discover real names on disk; don't assume.

---

## (a) Architecture

A single **async FastAPI (Python 3.11)** monolith backed by **one Postgres 16 +
pgvector container** with **one named Docker volume**. Postgres is the single
backing store and will play three roles as the product grows: **pgvector** for
semantic search, native **full-text search** (`tsvector` + GIN) for keyword
search, and ordinary **relational tables** for typed memories, supersession
history, and a small entity graph. Embeddings come from a **local bge-small
model baked into the image** (fastembed/ONNX, 384-dim) — no API key on the hot
path, no network at runtime. Default port **8080**; `docker compose up` boots
everything with no manual setup; data survives `docker compose down && up`.

```
            ┌─────────────────────────── app (FastAPI, :8080) ───────────────────────────┐
            │ lifespan: load embedder (offline) → run migrations → open pool(register_vector)│
 client ───▶│ api/routes (7 endpoints) → extraction → embeddings → db/queries               │
            └───────────────────────────────────┬──────────────────────────────────────────┘
                                                 │ asyncpg (cosine <=>, tsvector)
                                       ┌─────────▼─────────┐
                                       │ db: pgvector/pg16 │  ← named volume: memory_pgdata
                                       └───────────────────┘
```

Phase 1 ships a **naive but honest** path: each turn becomes ≥1 typed `event`
memory (never a raw chunk), embedded and recalled by vector cosine. Later phases
upgrade extraction (hybrid LLM+rules), recall (hybrid + RRF + tiered assembly),
fact evolution (supersession), and multi-hop (entity layer) — **without
destructive migrations**, because the full schema already exists.

---

## (b) Conventions (pinned — inherit these, do not re-decide)

- **DB image:** `pgvector/pgvector:pg16` (stock `postgres:16` lacks the
  extension). Migrations run `CREATE EXTENSION IF NOT EXISTS vector` first.
- **Embedding:** `BAAI/bge-small-en-v1.5` → **384 dims**. Column is
  `embedding vector(384)`; `EMBED_DIM = 384` in `src/config.py` is shared by the
  embedder and the column so they cannot drift. The model is **downloaded into a
  fixed cache dir (`/models`) at Docker build time and loaded offline at
  runtime** (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `EMBED_CACHE_DIR`).
  Boot needs no network.
- **Embedder backends:** `EMBED_BACKEND=fastembed` (default, the real model, used
  in the container) or `EMBED_BACKEND=fake` (a deterministic, dependency-free
  hashing embedder used by the test suite so pytest needs no model/network).
  Both emit 384-dim unit vectors. See `src/embeddings/embedder.py`.
- **pgvector access:** **asyncpg** with the pgvector type adapter registered on
  **every pooled connection** (`pgvector.asyncpg.register_vector` via the pool
  `init` callback in `src/db/pool.py`). Without this, vectors round-trip as
  strings. The extension is created (in migrations) **before** the pool opens,
  because the codec needs the `vector` type to exist.
- **Distance / index:** cosine via `<=>`; vector index is
  `USING hnsw (embedding vector_cosine_ops)` (HNSW builds on an empty table at
  first boot; IVFFlat does not). GIN index on the `search` tsvector. At this
  scale an exact scan is fine — the index is for correctness/scale, not speed.
- **Single tunable surface:** `src/config.py`. Seeded with `EMBED_DIM=384`,
  `TOP_K`, `RECALL_MIN_SCORE`. Later phases add RRF `k`, per-source top-N, tier
  token fractions, thresholds **here** — tuning = change one named value.
- **Cross-session scoping (the load-bearing decision):** a user's stable facts
  are **shared across all of that user's sessions** (recall/search filter by
  `user_id`, never by `session_id`) — this is what makes the §7 smoke test's
  `smoke-1`→`smoke-2` Berlin recall work. Only the "recent conversation" tier
  (a later phase) is session-scoped. **Different users never bleed.** A null
  `user_id` is scoped to its session via an `anon:{session_id}` key.
- **Auth:** optional bearer. Enforced **iff** `MEMORY_AUTH_TOKEN` is set
  (missing header → 401, wrong → 403), ignored otherwise. `/health` is always
  open. Read live so it can be toggled. See `src/api/auth.py`.
- **Status codes:** `/health` 200, `/turns` 201, `/recall` 200, `/search` 200,
  `/users/{id}/memories` 200, `DELETE /sessions/{id}` 204, `DELETE /users/{id}`
  204 (deletes are idempotent). Malformed input → 4xx (FastAPI 422), never a
  crash.
- **Logging:** structured JSON lines (`src/logging_config.py`) for lifecycle +
  per-request events (startup, embedder load, migrations, auth state, ingest,
  recall, deletes).

### How to run

```bash
# Service (graders' path):
docker compose up -d            # boots db (pgvector) + app (:8080)
until curl -sf localhost:8080/health; do sleep 1; done
./smoke.sh                      # §7 Berlin smoke test

# Tests (host, in-process app against the dockerized db):
docker compose up -d db         # db is exposed on host :5433 (avoids local 5432)
python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest  # uses EMBED_BACKEND=fake — no model/network needed
```

---

## (c) File map

```
docker-compose.yml   app(:8080) + db(pgvector/pgvector:pg16); named volume memory_pgdata;
                     healthchecks on both; depends_on service_healthy; db on host :5433 (test-only)
Dockerfile           python:3.11-slim; pip install; BAKES bge-small into /models at build; offline env; uvicorn
.env.example         optional ANTHROPIC_API_KEY, MEMORY_AUTH_TOKEN, DB vars
requirements.txt     runtime deps (fastapi, asyncpg, pgvector, fastembed, numpy, uvicorn)
requirements-dev.txt host test deps (no fastembed needed — fake embedder)
smoke.sh             §7 smoke test (Berlin, smoke-1 → smoke-2)
pytest.ini           pytest config (pythonpath, testpaths)
conftest.py          repo-root sys.path shim so `import src` works

src/
  main.py            FastAPI app + lifespan (embedder → migrations → pool); mounts routers
  config.py          THE tunable surface: EMBED_DIM, TOP_K, RECALL_MIN_SCORE, DB/auth helpers
  logging_config.py  JSON-line structured logging + log_event()
  api/
    models.py        Pydantic request/response models mirroring §3 EXACTLY
    auth.py          optional bearer dependency (require_auth)
    routes.py        the 7 endpoints; naive store/recall orchestration
  db/
    pool.py          asyncpg pool + register_vector on init
    migrations.py    idempotent full schema (turns, messages, memories, entities, memory_entities) + indexes
    queries.py       all SQL (insert turn/message/memory, recall_by_vector, list, delete session/user)
  embeddings/
    embedder.py      FastEmbedEmbedder (real, offline) + FakeEmbedder (tests) + get_embedder()
  extraction/        HYBRID, synchronous (phase 2). draft.py = MemoryDraft + the
                     canonical slot-key vocabulary (single source for slot names);
                     rules.py = deterministic regex extractor (offline safety net + the
                     default path); llm.py = PROVIDER-AGNOSTIC LLM extractor (Gemini /
                     Anthropic / OpenAI over httpx, forced structured JSON); pipeline.py =
                     extract() = LLM-primary → rule fallback → event fallback
  recall/            naive.py — cosine top-k, grouped into "Known facts" / "recent
                     conversations" sections + {turn_id,score,snippet} citations (phase 3+
                     adds hybrid + RRF + token-budgeted tiering)
  search/            naive.py — §3 /search result formatting (phases 3+ add full-text/hybrid)

tests/
  conftest.py        in-process TestClient factory; fake embedder; db on :5433; LLM_PROVIDER=none
  fixture_runner.py  ingest fixtures/ → EXTRACTION + RECALL-CONTEXT metrics (iteration loop);
                     importable (run_fixtures) and standalone (python tests/fixture_runner.py)
  test_contract.py   shapes + status codes (incl. multi-message + tool msg, name null & set)
  test_extraction.py typed records, implicit pet, correction, synchronous correctness, rule
                     fallback when LLM errors, LLM-result-used branch, fixture-metric guard
  test_persistence.py write → restart db container → still recallable
  test_isolation.py  different users don't bleed; same user A→B cross-session sharing works
  test_malformed.py  bad JSON / missing fields / unicode → 4xx; /health still 200
  test_auth.py       token unset = open; token set = 401/403/normal; /health always open
  test_delete.py     session delete (scoped) / user delete (cascade) / idempotent

fixtures/
  conversations.json 6 scripted multi-session scenarios + probes (basic / implicit pet /
                     fact evolution / multi-hop / noise / opinion arc). Data only.
```

### Canonical slot keys (authoritative list in `src/extraction/draft.py`)
`employment`, `location`, `origin`, `pet.name`, `pet.species`, `family.<relation>`,
`diet`, `allergy`, `preference.favorite.<thing>`, `preference.answer_style`,
`preference`, `opinion.<subject>`, `event` (fallback). Keys are stable per-topic so a
later phase can supersede two memories about the same slot (and model an opinion arc).

### LLM extraction is provider-agnostic (not Claude-specific)
`config.resolve_llm()` picks a provider by `LLM_PROVIDER` (`auto` = whichever of
`GEMINI_API_KEY`/`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` is set; Gemini
is the documented primary). With no key it returns `None` and extraction uses the rule
path — so the service is **fully offline by default** and the eval can enable an LLM by
just providing a key. `llm.py` forces structured JSON per provider (Gemini
`responseSchema`, Anthropic/OpenAI forced tool call) and returns `None` on any
error/timeout (`LLM_TIMEOUT`, default 25s) so the pipeline falls back to rules.

### Schema (idempotent, full product schema created in Phase 1)

- `turns(id, user_id, session_id, created_at)`
- `messages(id, turn_id→turns ON DELETE CASCADE, user_id, session_id, position, role, name?, content, created_at)`
- `memories(id, user_id, type∈{fact,preference,opinion,event}, key, value, confidence,
  source_session, source_turn→turns ON DELETE SET NULL, created_at, updated_at,
  supersedes→memories ON DELETE SET NULL, active, embedding vector(384), search tsvector,
  provenance)` — GIN(search), HNSW(embedding vector_cosine_ops). `provenance` (phase 2,
  additive `ADD COLUMN IF NOT EXISTS`) records which path produced the memory
  ("llm:gemini" / "rule" / "rule:event"); `search` is now populated on insert
  (`to_tsvector(key||' '||value)`) ready for phase-3 keyword recall.
- `entities(id, user_id, entity_type∈{user,pet,employer,city}, name, created_at)` *(populated later)*
- `memory_entities(memory_id→memories CASCADE, entity_id→entities CASCADE, relation)` *(populated later)*

---

## (d) Status — done / next

**Phase 1 — Skeleton, Contract & Docker: DONE.**
- 7 endpoints with exact §3 shapes + status codes; optional bearer auth (both branches).
- Full product schema in idempotent migrations (no later destructive migration needed).
- Offline local bge-small embeddings baked into the image; asyncpg + register_vector.
- Naive-but-honest store/recall: one typed `event` memory per turn, cosine top-k recall,
  cross-session sharing for the same user, different users isolated.
- Docker compose (app + pgvector db, named volume, healthchecks); §7 Berlin smoke passes.
- Tests: contract, persistence, isolation, malformed, auth, delete.

**Phase 2 — Hybrid extraction + recall-quality fixture: DONE.**
- `extraction/naive.py` replaced by `extraction/{draft,rules,llm,pipeline}.py`. Provider-
  agnostic LLM (Gemini primary / Anthropic / OpenAI, forced structured JSON) + deterministic
  rule fallback → typed memories with canonical slot keys, confidence, and `provenance`.
  Handles personal facts, preferences, opinions, corrections, and implicit facts. `/turns`
  stays synchronous; `provenance` column + `search` tsvector populated on insert.
- `fixtures/conversations.json` (6 scenarios) + `tests/fixture_runner.py` (two metrics, per-
  probe breakdown, idempotent). Baseline EXTRACTION **8/8 (100%)** on the rule path; recall
  in-scope 5/5 (evolution/multi-hop/noise recall are known phase-3 gaps). See CHANGELOG v0.2.
- Phase 1 NOT regressed: all six phase-1 test files still green (25 tests total); §7 Berlin
  smoke still passes against the rebuilt image.

**Phase 3 — Hybrid recall + RRF + tiered context assembly: NEXT.**
- Add keyword (tsvector/BM25) retrieval alongside vector; fuse with RRF. Build token-budgeted
  tiered assembly (stable facts → query-relevant → recent, session-scoped). Add a recall score
  threshold so noise queries return empty context.
- Then supersession (fact evolution: mark old same-slot facts inactive, keep history) and the
  entity layer for multi-hop. These are why the evolution/multi-hop/noise recall probes fail today.
