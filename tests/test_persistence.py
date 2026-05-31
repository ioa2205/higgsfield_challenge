"""Data survives a restart of the backing store (named volume)."""
from __future__ import annotations

import subprocess
import time
import uuid

import asyncpg
import pytest

from conftest import build_client_cm


def _wait_for_db(timeout=60):
    import asyncio

    async def _check():
        import os

        dsn = (
            f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
            f"@{os.environ['PGHOST']}:{os.environ['PGPORT']}/{os.environ['PGDATABASE']}"
        )
        conn = await asyncpg.connect(dsn=dsn)
        await conn.close()

    deadline = time.time() + timeout
    while True:
        try:
            asyncio.run(_check())
            return
        except Exception:
            if time.time() > deadline:
                raise
            time.sleep(1)


def test_data_survives_db_restart():
    user = "persist-" + uuid.uuid4().hex[:8]

    # Write a memory.
    with build_client_cm() as client:
        r = client.post(
            "/turns",
            json={
                "user_id": user,
                "session_id": "s1",
                "messages": [{"role": "user", "content": "I live in Berlin."}],
            },
        )
        assert r.status_code == 201

    # Restart the DB container (data lives in the named volume).
    try:
        subprocess.run(
            ["docker", "compose", "restart", "db"],
            check=True,
            capture_output=True,
            timeout=120,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:  # pragma: no cover
        pytest.skip(f"docker compose restart unavailable: {e}")

    _wait_for_db()

    # Data is still there.
    with build_client_cm() as client:
        r = client.get(f"/users/{user}/memories")
        assert r.status_code == 200
        values = [m["value"] for m in r.json()["memories"]]
        assert any("Berlin" in v for v in values)
