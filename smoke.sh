#!/usr/bin/env bash
# smoke.sh — CHALLENGE.md §7 smoke test (verbatim payloads).
# Run after: docker compose up -d && until curl -sf localhost:8080/health; do sleep 1; done
set -euo pipefail

BASE=${BASE:-http://localhost:8080}

echo "1. Health check..."
curl -s "$BASE/health" | grep -q '"ok"' || { echo "FAIL: health"; exit 1; }

echo "2. POST /turns (session smoke-1, user-1)..."
curl -s -X POST "$BASE/turns" \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "smoke-1",
    "user_id": "user-1",
    "messages": [
      {"role": "user", "content": "I just moved to Berlin from NYC last month. Loving it so far."},
      {"role": "assistant", "content": "That sounds exciting! Berlin is a great city. How are you settling in?"}
    ],
    "timestamp": "2025-03-15T10:30:00Z",
    "metadata": {}
  }' | grep -q '"id"' || { echo "FAIL: /turns did not return an id"; exit 1; }

echo "3. POST /recall (session smoke-2, SAME user-1 — cross-session)..."
RECALL=$(curl -s -X POST "$BASE/recall" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Where does this user live?",
    "session_id": "smoke-2",
    "user_id": "user-1",
    "max_tokens": 512
  }')
echo "$RECALL"
echo "$RECALL" | grep -qi "berlin" || { echo "FAIL: recall did not mention Berlin"; exit 1; }

echo "4. GET /users/user-1/memories (should be structured, typed)..."
MEM=$(curl -s "$BASE/users/user-1/memories")
echo "$MEM"
echo "$MEM" | grep -qi '"type"' || { echo "FAIL: memories not structured"; exit 1; }

echo "SMOKE TEST PASSED"
