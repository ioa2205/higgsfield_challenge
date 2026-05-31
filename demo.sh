#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-http://localhost:8080}
USER_ID=${DEMO_USER_ID:-reviewer-demo-user}
AUTH=()
if [[ -n "${MEMORY_AUTH_TOKEN:-}" ]]; then
  AUTH=(-H "Authorization: Bearer ${MEMORY_AUTH_TOKEN}")
fi

post_turn() {
  curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' \
    -X POST "$BASE/turns" -d "$1" >/dev/null
}

contains() {
  printf '%s' "$1" | grep -qi "$2" || {
    echo "FAIL: expected '$2' was absent"
    exit 1
  }
}

echo "Waiting for $BASE/health ..."
for _ in {1..60}; do
  if curl -fsS "$BASE/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -fsS "$BASE/health" >/dev/null

echo "Cleaning dedicated demo user ..."
curl -fsS "${AUTH[@]}" -X DELETE "$BASE/users/$USER_ID" >/dev/null

post_turn "{\"session_id\":\"demo-1\",\"user_id\":\"$USER_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"I work at Stripe.\"}]}"
post_turn "{\"session_id\":\"demo-2\",\"user_id\":\"$USER_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"I live in Lisbon.\"}]}"
post_turn "{\"session_id\":\"demo-3\",\"user_id\":\"$USER_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"My dog is named Biscuit.\"}]}"
post_turn "{\"session_id\":\"demo-4\",\"user_id\":\"$USER_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"I just joined Notion.\"}]}"

MEMORIES=$(curl -fsS "${AUTH[@]}" "$BASE/users/$USER_ID/memories")
RECALL=$(curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' \
  -X POST "$BASE/recall" \
  -d "{\"query\":\"What city does the user with the dog named Biscuit live in, and where do they work now?\",\"session_id\":\"demo-probe\",\"user_id\":\"$USER_ID\",\"max_tokens\":512}")

echo
echo "Stored memories:"
echo "$MEMORIES"
echo
echo "Multi-hop recall:"
echo "$RECALL"

for expected in Stripe Lisbon Biscuit Notion; do
  contains "$MEMORIES" "$expected"
done
for expected in Lisbon Biscuit Notion; do
  contains "$RECALL" "$expected"
done

echo
echo "DEMO PASSED"
