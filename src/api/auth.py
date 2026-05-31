"""Optional bearer-token auth.

Enforced iff ``MEMORY_AUTH_TOKEN`` is set; ignored otherwise. ``/health`` never
depends on this. Read live so it can be toggled between requests/tests.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import Header, HTTPException

from .. import config
from ..logging_config import log_event

logger = logging.getLogger("memory.auth")


async def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    token = config.auth_token()
    if not token:
        # Auth disabled: accept everything.
        return

    if authorization is None:
        log_event(logger, "auth.denied", reason="missing_header")
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        log_event(logger, "auth.denied", reason="malformed_header")
        raise HTTPException(status_code=401, detail="Malformed Authorization header")

    if parts[1] != token:
        log_event(logger, "auth.denied", reason="invalid_token")
        raise HTTPException(status_code=403, detail="Invalid token")
