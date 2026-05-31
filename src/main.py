"""FastAPI app + lifespan startup.

Startup order matters: run idempotent migrations (which create the pgvector
extension) BEFORE opening the pool, because the pool's init callback registers
the ``vector`` type codec and needs the type to already exist. The embedder is
loaded offline from its baked cache.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import config
from .api.routes import protected, router
from .db.migrations import run_migrations
from .db.pool import create_pool
from .embeddings import get_embedder
from .logging_config import log_event, setup_logging

logger = logging.getLogger("memory")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(config.LOG_LEVEL)
    log_event(
        logger,
        "startup.begin",
        embed_backend=config.EMBED_BACKEND,
        embed_model=config.EMBED_MODEL,
        embed_dim=config.EMBED_DIM,
        auth="enforced" if config.auth_token() else "disabled",
    )

    # Load the local embedder (offline for the real backend).
    app.state.embedder = get_embedder()
    log_event(
        logger,
        "embedder.loaded",
        backend=getattr(app.state.embedder, "backend", "unknown"),
        dim=app.state.embedder.dim,
    )

    # Idempotent migrations create the extension + full schema.
    await run_migrations(config.database_url())
    log_event(logger, "migrations.applied")

    # Pool with register_vector on every connection.
    app.state.pool = await create_pool()
    log_event(logger, "pool.ready")

    log_event(logger, "startup.complete")
    try:
        yield
    finally:
        await app.state.pool.close()
        log_event(logger, "shutdown.complete")


app = FastAPI(title="Memory Service", version="0.1.0", lifespan=lifespan)
app.include_router(router)
app.include_router(protected)
