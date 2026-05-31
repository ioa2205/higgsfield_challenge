"""asyncpg pool with the pgvector type adapter registered on every connection.

Without ``register_vector`` on each pooled connection, ``vector`` columns
round-trip as strings and the ``<=>`` cosine operator can't be used with
Python lists. The init callback fixes that for the whole pool.
"""
from __future__ import annotations

import asyncpg
from pgvector.asyncpg import register_vector

from .. import config


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def create_pool() -> asyncpg.Pool:
    return await asyncpg.create_pool(
        dsn=config.database_url(),
        min_size=config.POOL_MIN,
        max_size=config.POOL_MAX,
        init=_init_connection,
    )
