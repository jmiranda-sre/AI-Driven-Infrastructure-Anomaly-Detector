"""Structured JSON logging with correlation ID propagation.

Uses structlog for JSON output, correlation context vars, and
PII masking for sensitive fields.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

_PII_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(password|passwd|pwd|secret|token|api[_-]?key)", re.IGNORECASE),
    re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"),  # CPF-like
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # email
]


def _mask_pii(value: str) -> str:
    """Mask potential PII values."""
    for pattern in _PII_PATTERNS:
        if pattern.search(str(value)):
            if "@" in str(value):
                parts = str(value).split("@")
                return f"{parts[0][:2]}***@{parts[1]}"
            return "***MASKED***"
    return value


def _filter_sensitive(
    logger: logging.Logger, method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Processor that masks PII in log events."""
    for key, value in event_dict.items():
        if isinstance(value, str):
            event_dict[key] = _mask_pii(value)
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, str):
                    event_dict[key][k] = _mask_pii(v)
    return event_dict


def _add_correlation_id(
    logger: logging.Logger, method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Ensure correlation_id is present; auto-generate if missing."""
    if "correlation_id" not in event_dict:
        from uuid import uuid4

        event_dict.setdefault("correlation_id", str(uuid4())[:8])
    return event_dict


def _safe_filter_by_level(
    logger: logging.Logger, method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Like filter_by_level but safe when logger is None (background threads)."""
    if logger is None:
        return event_dict
    return structlog.stdlib.filter_by_level(logger, method, event_dict)


def configure_logging(log_level: str = "info", json_format: bool = True) -> None:
    """Configure structlog for JSON structured logging."""
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _safe_filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        _add_correlation_id,
        _filter_sensitive,
    ]

    if json_format:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level.upper())


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger bound to a module name."""
    return structlog.get_logger(name)


def bind_correlation(correlation_id: str) -> None:
    """Bind correlation ID to all subsequent log calls in this context."""
    bind_contextvars(correlation_id=correlation_id)


def clear_correlation() -> None:
    """Clear correlation context (call at request end)."""
    clear_contextvars()
