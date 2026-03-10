"""Structured logging setup using structlog."""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(*, log_level: str = "INFO", json_output: bool = False) -> None:
    """Configure structlog for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        json_output: If True, output JSON lines (for containers). Otherwise, console.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper()))


def get_logger(component: str, **initial_binds: object) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger bound with component name.

    Args:
        component: Module/component name (e.g., "tick_loop", "kelly").
        **initial_binds: Additional initial context (e.g., match_id="12345").

    Returns:
        A bound structlog logger.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(component=component, **initial_binds)
    return logger
