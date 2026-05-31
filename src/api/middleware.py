"""Small ASGI middleware used before FastAPI parses request bodies."""
from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ..logging_config import log_event

logger = logging.getLogger("memory.middleware")


class RequestBodyTooLarge(Exception):
    pass


class MaxBodySizeMiddleware:
    """Reject oversized request bodies even when Content-Length is absent."""

    def __init__(self, app: Callable[..., Awaitable[Any]], max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_bytes:
                    await self._reject(send, "content-length")
                    return
            except ValueError:
                pass

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestBodyTooLarge:
            await self._reject(send, "stream")

    async def _reject(self, send, source: str) -> None:
        log_event(
            logger,
            "request.rejected",
            reason="body_too_large",
            source=source,
            max_bytes=self.max_bytes,
        )
        body = json.dumps({"detail": "Request body too large"}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
