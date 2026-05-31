"""FastAPI app + lifespan startup.

Startup order matters: run idempotent migrations (which create the pgvector
extension) BEFORE opening the pool, because the pool's init callback registers
the ``vector`` type codec and needs the type to already exist. The embedder is
loaded offline from its baked cache.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import config
from .api.middleware import MaxBodySizeMiddleware
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


app = FastAPI(title="Memory Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(MaxBodySizeMiddleware, max_bytes=config.MAX_REQUEST_BODY_BYTES)


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """Keep unexpected request failures contained and observable."""
    logger.exception(
        "request.unhandled",
        extra={
            "fields": {
                "method": request.method,
                "path": request.url.path,
                "exception": type(exc).__name__,
            }
        },
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(router)
app.include_router(protected)
