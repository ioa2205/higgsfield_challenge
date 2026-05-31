"""Recall-quality fixture runner — the project's iteration loop.

Ingests every fixture conversation (``fixtures/conversations.json``) via
``POST /turns``, then for each probe reports TWO metrics with a per-probe
breakdown:

  * **EXTRACTION** — of the facts a probe expects, how many appear as a typed
    memory value in ``GET /users/{id}/memories``. This is the metric Phase 2
    tunes (the LLM/rule extractor), independent of recall ranking.
  * **RECALL-CONTEXT** — of those facts, how many appear in the ``POST /recall``
    context string. ``recall_expected:false`` remains available for honest
    iteration tracking, although every shipped v1.0 probe is now in scope.

Idempotent: each scenario's user is DELETEd before ingest, so repeated runs (and
the test that wraps this) start clean. Importable (``run_fixtures(client)``) and
runnable standalone (``python tests/fixture_runner.py`` — spins up the in-process
app with the fake embedder against the dockerized db on :5433).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "conversations.json"
ADVERSARIAL_FIXTURES = (
    Path(__file__).resolve().parents[1] / "fixtures" / "adversarial_conversations.json"
)


def load_fixtures(path: Path = FIXTURES) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class ProbeResult:
    scenario: str
    category: str
    query: str
    recall_expected: bool
    expect_empty: bool
    extraction_total: int
    extraction_hits: int
    recall_total: int
    recall_hits: int
    missing_extraction: list[str] = field(default_factory=list)
    missing_recall: list[str] = field(default_factory=list)


@dataclass
class Report:
    probes: list[ProbeResult] = field(default_factory=list)

    def _sum(self, attr_hits: str, attr_total: str, only_recall_expected=False):
        hits = total = 0
        for p in self.probes:
            if only_recall_expected and not p.recall_expected:
                continue
            hits += getattr(p, attr_hits)
            total += getattr(p, attr_total)
        return hits, total

    @property
    def extraction(self):
        return self._sum("extraction_hits", "extraction_total")

    @property
    def recall(self):
        return self._sum("recall_hits", "recall_total")

    @property
    def recall_in_scope(self):
        """Recall over only the probes whose recall is expected to work now."""
        return self._sum("recall_hits", "recall_total", only_recall_expected=True)


def _found(substrings: list[str], haystack: str) -> tuple[int, list[str]]:
    hay = haystack.lower()
    hits, missing = 0, []
    for s in substrings:
        if s.lower() in hay:
            hits += 1
        else:
            missing.append(s)
    return hits, missing


def run_fixtures(client, data: dict | None = None) -> Report:
    data = data or load_fixtures()
    report = Report()

    for scenario in data["scenarios"]:
        user_id = scenario["user_id"]
        # Idempotency: wipe any prior run for this user first.
        client.delete(f"/users/{user_id}")

        for convo in scenario["conversations"]:
            r = client.post(
                "/turns",
                json={
                    "user_id": user_id,
                    "session_id": convo["session_id"],
                    "messages": convo["messages"],
                },
            )
            assert r.status_code == 201, f"ingest failed: {r.status_code} {r.text}"

        # EXTRACTION surface = STRUCTURED typed memories only. The raw `event`
        # fallback is excluded on purpose: a fact "present" only because its
        # words survive in an event blob is the raw-chunk anti-pattern §4 warns
        # about, not real extraction.
        mems = client.get(f"/users/{user_id}/memories").json()["memories"]
        mem_blob = "  ".join(m["value"] for m in mems if m["type"] != "event")

        for probe in scenario["probes"]:
            expect = probe.get("expect", [])
            extract_expect = probe.get("extract_expect", expect)
            expect_empty = probe.get("expect_empty", False)
            recall_expected = probe.get("recall_expected", True)

            ext_hits, ext_missing = _found(extract_expect, mem_blob)

            recall = client.post(
                "/recall",
                json={
                    "user_id": user_id,
                    "session_id": f"{scenario['id']}-probe",
                    "query": probe["query"],
                    "max_tokens": 512,
                },
            ).json()
            context = recall.get("context", "")

            if expect_empty:
                rec_total, rec_hits = 1, (1 if not context.strip() else 0)
                rec_missing = [] if rec_hits else ["<expected empty context>"]
                ext_total = 0  # nothing to extract for a noise probe
            else:
                rec_hits, rec_missing = _found(expect, context)
                rec_total = len(expect)
                ext_total = len(extract_expect)

            report.probes.append(
                ProbeResult(
                    scenario=scenario["id"],
                    category=probe.get("category", scenario.get("category", "")),
                    query=probe["query"],
                    recall_expected=recall_expected,
                    expect_empty=expect_empty,
                    extraction_total=ext_total,
                    extraction_hits=ext_hits if not expect_empty else 0,
                    recall_total=rec_total,
                    recall_hits=rec_hits,
                    missing_extraction=ext_missing if not expect_empty else [],
                    missing_recall=rec_missing,
                )
            )

    return report


def format_report(report: Report) -> str:
    lines = ["", "=" * 78, "RECALL-QUALITY FIXTURE — per-probe breakdown", "=" * 78]
    lines.append(
        f"{'scenario':<16} {'category':<10} {'extract':>8} {'recall':>8}  notes"
    )
    lines.append("-" * 78)
    for p in report.probes:
        ext = "n/a" if p.expect_empty else f"{p.extraction_hits}/{p.extraction_total}"
        rec = f"{p.recall_hits}/{p.recall_total}"
        notes = []
        if not p.recall_expected:
            notes.append("recall intentionally out of scope")
        if p.missing_extraction:
            notes.append("missing-ext:" + ",".join(p.missing_extraction))
        if p.missing_recall and not p.expect_empty:
            notes.append("missing-rec:" + ",".join(p.missing_recall))
        lines.append(
            f"{p.scenario:<16} {p.category:<10} {ext:>8} {rec:>8}  {'; '.join(notes)}"
        )
    lines.append("-" * 78)
    eh, et = report.extraction
    rh, rt = report.recall
    rih, rit = report.recall_in_scope
    pct = lambda h, t: f"{(100 * h / t):.0f}%" if t else "n/a"
    lines.append(f"EXTRACTION metric      : {eh}/{et}  ({pct(eh, et)})")
    lines.append(f"RECALL-CONTEXT metric  : {rh}/{rt}  ({pct(rh, rt)})  (all probes)")
    lines.append(
        f"RECALL (in-scope only) : {rih}/{rit}  ({pct(rih, rit)})  "
        f"(all shipped v1.0 probes are in scope)"
    )
    lines.append("=" * 78)
    return "\n".join(lines)


class _LiveClient:
    """Minimal TestClient-shaped adapter over a running service (real embedder),
    so ``run_fixtures`` can measure the dockerized container at :8080 — the true
    eval condition. Used by ``python tests/fixture_runner.py --live``."""

    def __init__(self, base_url: str, token: str | None = None) -> None:
        import requests

        self._base = base_url.rstrip("/")
        self._s = requests.Session()
        if token:
            self._s.headers["Authorization"] = f"Bearer {token}"

    def post(self, path, json=None):
        return self._s.post(self._base + path, json=json, timeout=60)

    def get(self, path):
        return self._s.get(self._base + path, timeout=60)

    def delete(self, path):
        return self._s.delete(self._base + path, timeout=60)


def main() -> None:
    import sys

    def print_reports(client) -> None:
        print("[original fixture]")
        print(format_report(run_fixtures(client)))
        print("\n[adversarial fixture]")
        print(format_report(run_fixtures(client, load_fixtures(ADVERSARIAL_FIXTURES))))

    if "--live" in sys.argv:
        base = os.environ.get("MEMORY_URL", "http://localhost:8080")
        token = os.environ.get("MEMORY_AUTH_TOKEN") or None
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
        print(f"[live @ {base}]")
        print_reports(_LiveClient(base, token))
        return

    # Spin up the in-process app exactly like the test suite.
    os.environ.setdefault("EMBED_BACKEND", "fake")
    os.environ.setdefault("PGHOST", "localhost")
    os.environ.setdefault("PGPORT", "5433")
    os.environ.setdefault("PGUSER", "memory")
    os.environ.setdefault("PGPASSWORD", "memory")
    os.environ.setdefault("PGDATABASE", "memory")
    os.environ.pop("MEMORY_AUTH_TOKEN", None)
    # Offline, deterministic baseline: force the rule path (override any host key).
    os.environ.setdefault("LLM_PROVIDER", "none")

    import sys

    # Make stdout tolerant of any non-ASCII in formatted output on Windows.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
    from conftest import build_client_cm  # tests/conftest.py

    with build_client_cm() as client:
        print_reports(client)


if __name__ == "__main__":
    main()
