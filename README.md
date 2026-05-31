# Memory Service

A Dockerized long-term memory service for an AI agent. It ingests completed
conversation turns, extracts typed memories synchronously, preserves fact
history, and returns prompt-ready context through `/recall`.

## 1. Architecture

```text
client
  |
  v
FastAPI :8080
  |-- request limits + optional bearer auth
  |-- /turns  -> hybrid extraction -> local embeddings -> one DB transaction
  |-- /recall -> vector + FTS -> RRF -> entity hop -> tiered assembly
  `-- /search -> structured hybrid results
  |
  v
Postgres 16 + pgvector
  |-- relational history + entities
  |-- pgvector HNSW cosine search
  `-- tsvector GIN full-text search
  |
  `-- named Docker volume: memory_pgdata
```

The service is one async FastAPI monolith and one `pgvector/pgvector:pg16`
container. Startup loads the baked local embedding model, applies idempotent
migrations, and opens an `asyncpg` pool with pgvector codecs registered on
every connection. `/turns` extracts, embeds, and writes the turn, messages,
memories, supersession updates, and entity links before returning `201`.

Stable facts intentionally share across sessions for the same `user_id`.
Recent-event context remains session-scoped. Different users never share
memories. A null `user_id` is scoped as `anon:<session_id>`.

## 2. Backing Store Choice

The service uses one Postgres database for vector search, full-text search, and
relational history. A single store makes synchronous correctness and atomic
writes straightforward: there is no cross-store indexing lag after `/turns`
returns. `pgvector` provides HNSW cosine search, Postgres `tsvector` provides a
GIN-backed keyword path, and ordinary tables preserve supersession chains and
entity links. A named Docker volume keeps data across `docker compose down`
and `docker compose up`.

## 3. Extraction Pipeline

Extraction is hybrid and synchronous:

1. If configured, Gemini, Anthropic, or OpenAI receives the turn and must emit
   structured JSON through provider-specific schema/tool forcing.
2. If no key exists, the provider fails, or the output is unusable, deterministic
   regex rules run offline.
3. If no typed fact matches, one `event` fallback keeps the turn queryable.

Both paths emit the same `MemoryDraft`: `type`, canonical slot `key`, concise
third-person `value`, confidence, and provenance. Rules cover employment,
location, origin, pets including implicit mentions, family, diet, allergies,
preferences, opinions, and correction cues. They intentionally miss arbitrary
paraphrases and nuanced relationships outside those patterns; the optional LLM
path is the broader extractor. The event fallback is labeled honestly and is
not presented as a structured fact.

## 4. Recall Strategy

`/recall` embeds the query locally with `BAAI/bge-small-en-v1.5` and retrieves
active memories through pgvector cosine search and Postgres full-text search.
Reciprocal Rank Fusion combines ranks without pretending cosine and `ts_rank`
are calibrated to the same scale. A conservative noise gate opens only for a
keyword hit or cosine score of at least `0.55`. Entity-name decomposition can
widen the candidate pool for multi-hop questions such as a pet name leading to
the user's city.

Assembly is greedy under `max_tokens`:

1. Active stable facts, preferences, and opinions win the budget.
2. Query-relevant event memories come next.
3. Recent events from the current session come last.

The estimator intentionally over-counts with `max(words * 1.3, chars / 4)` so
unicode-heavy context stays bounded. A cold store or irrelevant query returns
`{"context":"","citations":[]}` with `200`.

Locked v1.0 defaults: `SEM_TOP_N=20`, `KW_TOP_N=20`, `RRF_K=60`,
`RECALL_MIN_SCORE=0.55`, `TIER3_RECENT_N=5`, `RECALL_SNIPPET_MAX=240`, and
`SUPERSESSION_SIM_THRESHOLD=0.92`.

## 5. Fact Evolution

Canonical keys make mutable facts explicit: `employment`, `location`,
`opinion.typescript`, and similar slots. A new memory first supersedes an
active same-key memory. A conservative same-type embedding match is a fallback
for variant LLM keys. The old row is retained with `active=false`; the new row
points to it through `supersedes`. Recall renders the current value and can add
a `previously ...` annotation.

Corrections use the same mechanism. Opinion changes are modeled as a preserved
chain: the latest stance is active while earlier stances remain inspectable.
This captures an opinion arc without flattening history into one opaque blob.

## 6. Tradeoffs

The design optimizes for synchronous correctness, understandable ranking, and
offline operation. One database and a monolith are enough for this workload and
keep failure handling tractable. The local 384-dimensional model avoids a
network dependency on the recall hot path.

The rule extractor is deliberately bounded rather than pretending to understand
every phrasing. Entity traversal is also small and user-scoped rather than a
general graph engine. RRF and the noise gate favor predictable recall over a
larger reranker dependency. This is our own design, not a clone of mem0, Honcho,
or another public memory system.

## 7. Failure Modes

- No data or unrelated query: `/recall` returns empty context with `200`.
- Missing LLM API key: extraction logs degradation and uses offline rules.
- Missing or wrong bearer token when configured: protected endpoints return
  `401` or `403`; `/health` stays open.
- Malformed JSON, null required fields, NUL bytes, and oversized bodies:
  rejected as `4xx`; body size is capped at `1 MiB` before JSON parsing.
- Emoji, RTL markers, and long grapheme sequences: accepted within bounded
  field limits.
- Restart during `/turns`: the open transaction rolls back; startup migrations
  are idempotent and committed turns remain recallable.
- Slow disk or database pressure: request latency rises because writes are
  intentionally synchronous. The next optimization would be query-plan and
  pool instrumentation, then index and pool sizing, not eventual consistency.

Measured live-container `/recall` latency on the bundled fixture with the real
embedding model: **15.6 ms p50**, **17.5 ms p95**, **25.6 ms worst** across 35
requests on the development machine.

## 8. How To Run The Tests

Clean-machine service flow:

```bash
docker compose up -d
until curl -sf localhost:8080/health; do sleep 1; done
./smoke.sh
```

Host suite against the Dockerized database:

```bash
docker compose up -d db
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest
```

On Linux or macOS, use `.venv/bin/pip` and `.venv/bin/python`. The suite uses a
deterministic fake embedder for repeatability and includes process-level
missing-key and restart-mid-write tests. Run the live real-embedder fixture with:

```bash
.venv/Scripts/python tests/fixture_runner.py --live
```
