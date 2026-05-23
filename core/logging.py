"""
Structured logging configuration for Aether-Kernel.

Uses the standard library logging module with JSON formatter hooks so that
operators can ingest logs into ELK / Datadog / CloudWatch without heavy
dependencies. structlog is optional and layered in when available.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

# Optional structlog for richer structured output
try:
    import structlog

    _HAS_STRUCTLOG = True
except ImportError:  # pragma: no cover
    _HAS_STRUCTLOG = False


class _JSONFormatter(logging.Formatter):
    """Lightweight JSON formatter for standard library logging.

    Avoids third-party JSON-logging packages to keep the dependency tree minimal.
    """

    def format(self, record: logging.LogRecord) -> str:
        import json

        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        if hasattr(record, "context"):
            payload["context"] = record.context  # type: ignore[attr-defined]
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with Aether-Kernel's formatting conventions.

    Callers may attach a ``context`` dict to log records via
    ``logger.info("msg", extra={"context": {"trace_id": "..."}})``.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent global logging setup; safe to call multiple times."""
    root = logging.getLogger("aether_kernel")
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JSONFormatter())
        root.addHandler(handler)
