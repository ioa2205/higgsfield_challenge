"""Structured (JSON-line) logging.

A §10 "well-logged" down-payment: lifecycle events (startup, embedder load,
auth enforced/skipped, extraction path) are emitted as single-line JSON so they
are greppable and machine-parseable.
"""
from __future__ import annotations

import json
import logging


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def log_event(logger: logging.Logger, event: str, **fields) -> None:
    """Emit a structured event with arbitrary key/value fields."""
    logger.info(event, extra={"fields": fields})
