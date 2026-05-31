# Memory Service

A Dockerized memory service for an AI agent: it ingests conversation turns,
extracts structured memories, and answers `/recall` queries that decide what
context the agent sees next. See `CHALLENGE.md` for the full spec and `CLAUDE.md`
for architecture/conventions.

> **Status:** Phase 1 (skeleton, HTTP contract, Docker, naive baseline). The
> recall pipeline is intentionally a naive cosine top-k baseline at this stage;
> hybrid retrieval, real extraction, fact evolution, and multi-hop land in later
> phases. The `CHANGELOG.md` tracks each iteration honestly.

## Architecture (short)

Single async **FastAPI** monolith over **one Postgres 16 + pgvector** container
(one named volume). Postgres is the sole backing store — vector search now,
full-text + relational fact history later. Embeddings are a **local
bge-small-en-v1.5 (384-dim)** model **baked into the image and loaded offline**,
so the hot path needs no API key and no network. `asyncpg` registers the
pgvector type adapter on every pooled connection; cosine via `<=>`.

```
client → FastAPI (:8080) → extraction → local embedder → Postgres(pgvector)
                                                            ^ named volume
```

## Backing store choice

One Postgres for everything removes cross-store consistency bugs and makes the
"if you wrote it, you can read it" guarantee fall out of a single transaction.
`pgvector/pgvector:pg16` is used because stock `postgres:16` lacks the extension.

## Cross-session scoping (intentional design decision)

A user's **stable facts are shared across all of that user's sessions** — recall
filters by `user_id`, not `session_id`. This is required by the smoke test (write
under `smoke-1`, recall under `smoke-2` for the same `user-1`, still expects
"Berlin"). Only the "recent conversation" tier (a later phase) is session-scoped.
**Different users never bleed.**

## Recall strategy (Phase 1 baseline)

Each turn produces at least one typed `event` memory (never a raw message
chunk), which is embedded and stored. `/recall` embeds the query and returns the
top-k memories by cosine distance, formatted as a readable context block with
`{turn_id, score, snippet}` citations. Cold/irrelevant queries return
`{"context": "", "citations": []}` with 200. This is a deliberate baseline —
the challenge notes vanilla cosine-top-k won't score well, and later phases add
hybrid retrieval + RRF + tiered, budget-aware assembly.

## Failure modes

- **No data / cold session:** `/recall` returns empty context, 200 — never errors.
- **Missing API keys:** none required for the core path; embeddings are local.
- **Malformed input:** validation errors return 4xx; the service never crashes
  and `/health` stays 200.

## Run it

```bash
docker compose up -d
until curl -sf localhost:8080/health; do sleep 1; done
./smoke.sh
```

## Run the tests

The suite drives the app **in-process** against the dockerized Postgres using a
deterministic "fake" embedder (no model download, no network):

```bash
docker compose up -d db          # db exposed on host :5433
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt   # (Linux/macOS: .venv/bin/pip)
.venv/Scripts/python -m pytest
```

Tests: `test_contract`, `test_persistence`, `test_isolation`, `test_malformed`,
`test_auth`, `test_delete`.
