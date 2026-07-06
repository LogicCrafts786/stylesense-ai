"""
Centralized logging configuration for StyleSense AI.

Uses loguru for structured, rotating file logs plus console output.
Import `get_logger` anywhere in the codebase to obtain a pre-configured
logger bound with the calling module's name.

Example:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Agent initialized")
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _loguru_logger

from src.utils.config import get_settings

_CONFIGURED = False


def _configure_logging() -> None:
    """Configure loguru sinks exactly once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()

    # Remove default handler to avoid duplicate console output.
    _loguru_logger.remove()

    # Console sink - human readable, colorized.
    _loguru_logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        backtrace=False,
        diagnose=False,
    )

    # File sink - rotating, retains 7 days, includes tracebacks for debugging.
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    _loguru_logger.add(
        log_dir / "stylesense_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="7 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
        enqueue=True,  # thread-safe writes
    )

    _CONFIGURED = True


def get_logger(name: str):
    """
    Return a loguru logger bound with a module name for contextual logging.

    Args:
        name: Typically `__name__` of the calling module.

    Returns:
        A loguru logger instance bound with the module context.
    """
    _configure_logging()
    return _loguru_logger.bind(module=name)
