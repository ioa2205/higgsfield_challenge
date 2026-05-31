"""Process-level resilience checks against the same durable Postgres volume."""
from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager

import asyncpg
import pytest
import requests

from fixture_runner import _LiveClient, run_fixtures


def _dsn() -> str:
    return (
        f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
        f"@{os.environ['PGHOST']}:{os.environ['PGPORT']}/{os.environ['PGDATABASE']}"
    )


def _db(sql: str, *args):
    async def run():
        conn = await asyncpg.connect(_dsn())
        try:
            if args or sql.lstrip().upper().startswith("SELECT"):
                return await conn.fetchval(sql, *args)
            return await conn.execute(sql)
        finally:
            await conn.close()

    return asyncio.run(run())


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_health(base: str, timeout: float = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(base + "/health", timeout=1).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.2)
    raise AssertionError(f"app failed to become healthy at {base}")


@contextmanager
def _app_process():
    port = _port()
    base = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(
        {
            "EMBED_BACKEND": "fake",
            "LLM_PROVIDER": "auto",
            "PGHOST": os.environ["PGHOST"],
            "PGPORT": os.environ["PGPORT"],
            "PGUSER": os.environ["PGUSER"],
            "PGPASSWORD": os.environ["PGPASSWORD"],
            "PGDATABASE": os.environ["PGDATABASE"],
        }
    )
    for name in (
        "MEMORY_AUTH_TOKEN",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ):
        env.pop(name, None)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_health(base)
        yield proc, base
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def _post_turn(base: str, user: str, session: str, content: str):
    return requests.post(
        base + "/turns",
        json={
            "user_id": user,
            "session_id": session,
            "messages": [{"role": "user", "content": content}],
        },
        timeout=60,
    )


def test_missing_api_key_separate_app_runs_full_fixture():
    """A separately launched app with every provider key unset uses rules."""
    try:
        with _app_process() as (_, base):
            report = run_fixtures(_LiveClient(base))
    except FileNotFoundError as exc:  # pragma: no cover - environment guard
        pytest.skip(f"separate app launch unavailable: {exc}")
    assert report.extraction == (8, 8)
    assert report.recall_in_scope == (9, 9)


def test_restart_mid_write_rolls_back_partial_turn():
    """Kill the app while Postgres is sleeping inside a message insert."""
    committed_user = "restart-committed-" + uuid.uuid4().hex[:8]
    interrupted_user = "restart-interrupted-" + uuid.uuid4().hex[:8]
    interrupted_session = "restart-mid-write-" + uuid.uuid4().hex[:8]

    trigger = """
    CREATE OR REPLACE FUNCTION phase5_sleep_mid_write() RETURNS trigger AS $$
    BEGIN
        IF NEW.session_id = '%s' THEN
            PERFORM pg_sleep(30);
        END IF;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    DROP TRIGGER IF EXISTS phase5_sleep_mid_write_trigger ON messages;
    CREATE TRIGGER phase5_sleep_mid_write_trigger
        BEFORE INSERT ON messages
        FOR EACH ROW EXECUTE FUNCTION phase5_sleep_mid_write();
    """ % interrupted_session

    try:
        with _app_process() as (proc, base):
            assert _post_turn(base, committed_user, "committed", "I live in Berlin.").status_code == 201
            _db(trigger)

            result: dict[str, object] = {}

            def ingest():
                try:
                    result["response"] = _post_turn(
                        base, interrupted_user, interrupted_session, "I work at Notion."
                    )
                except requests.RequestException as exc:
                    result["error"] = exc

            thread = threading.Thread(target=ingest, daemon=True)
            thread.start()
            deadline = time.time() + 10
            while time.time() < deadline:
                active = _db(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM pg_stat_activity
                        WHERE datname = current_database()
                          AND state = 'active'
                          AND query ILIKE '%%INSERT INTO messages%%'
                    )
                    """
                )
                if active:
                    break
                time.sleep(0.1)
            else:
                raise AssertionError("mid-write trigger did not become active")

            proc.terminate()
            proc.wait(timeout=10)
            thread.join(timeout=10)
    finally:
        _db("DROP TRIGGER IF EXISTS phase5_sleep_mid_write_trigger ON messages")
        _db("DROP FUNCTION IF EXISTS phase5_sleep_mid_write()")

    with _app_process() as (_, base):
        assert requests.get(base + "/health", timeout=5).status_code == 200
        recall = requests.post(
            base + "/recall",
            json={
                "user_id": committed_user,
                "session_id": "after-restart",
                "query": "Where does the user live?",
            },
            timeout=10,
        )
        assert recall.status_code == 200
        assert "berlin" in recall.json()["context"].lower()

    assert _db(
        "SELECT count(*) FROM turns WHERE session_id = $1", interrupted_session
    ) == 0
    assert _db(
        "SELECT count(*) FROM messages WHERE session_id = $1", interrupted_session
    ) == 0
    assert _db(
        "SELECT count(*) FROM memories WHERE source_session = $1", interrupted_session
    ) == 0
    assert _db(
        """
        SELECT count(*) FROM memories
        WHERE value IS NULL OR confidence IS NULL OR active IS NULL
        """
    ) == 0
