"""
utils.py
─────────────────────────────────────────────────────────────────────────────
Shared utilities for the Football Analytics System.

Contains:
  • get_logger()   – project-wide Loguru logger factory
  • retry()        – decorator for retrying failed HTTP calls
  • timer()        – decorator that logs function execution time
  • safe_divide()  – division that returns 0 on ZeroDivisionError
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import functools
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from loguru import logger

from src.config import cfg

# ── Logger factory ────────────────────────────────────────────────────────────
_LOGGER_CONFIGURED = False


def get_logger(name: str = "football_analytics") -> "logger":
    """
    Return a Loguru logger that writes to both stdout and a rotating log file.

    Loguru uses a single global logger instance; this function configures it
    once (idempotent) then returns it.  Each call returns the same object.

    Args:
        name: Logical name embedded in the log file filename.

    Returns:
        Configured loguru logger.
    """
    global _LOGGER_CONFIGURED
    if not _LOGGER_CONFIGURED:
        # Remove the default handler so we control all output
        logger.remove()

        # ── Console sink ──────────────────────────────────────────────────────
        logger.add(
            sys.stderr,
            level=cfg.LOG_LEVEL,
            colorize=True,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
        )

        # ── File sink ─────────────────────────────────────────────────────────
        log_path = cfg.LOGS_DIR / f"{name}.log"
        logger.add(
            str(log_path),
            level=cfg.LOG_LEVEL,
            rotation=cfg.LOG_ROTATION,
            retention="30 days",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        )

        _LOGGER_CONFIGURED = True

    return logger


# ── Decorators ────────────────────────────────────────────────────────────────
F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    backoff: float = 1.5,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator: retry a function up to *max_attempts* times on *exceptions*.

    Each successive failure waits backoff^attempt seconds before retrying,
    implementing **exponential back-off** to avoid hammering rate-limited APIs.

    Args:
        max_attempts: Total tries (including the first attempt).
        backoff:      Multiplier for exponential wait time.
        exceptions:   Tuple of exception types that trigger a retry.

    Example:
        @retry(max_attempts=3, backoff=2.0, exceptions=(HTTPError,))
        def fetch_data(url: str) -> dict: ...
    """
    log = get_logger()

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        log.error(
                            f"[retry] {func.__name__} failed after {max_attempts} attempts: {exc}"
                        )
                        raise
                    wait = backoff ** attempt
                    log.warning(
                        f"[retry] {func.__name__} attempt {attempt}/{max_attempts} failed: {exc}. "
                        f"Retrying in {wait:.1f}s…"
                    )
                    time.sleep(wait)

        return wrapper  # type: ignore[return-value]

    return decorator


def timer(func: F) -> F:
    """
    Decorator: log how long a function takes to execute.

    Useful for profiling expensive data-loading or analysis steps.
    """
    log = get_logger()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        log.debug(f"[timer] {func.__qualname__} completed in {elapsed:.2f}s")
        return result

    return wrapper  # type: ignore[return-value]


# ── Math helpers ──────────────────────────────────────────────────────────────
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Divide *numerator* by *denominator*, returning *default* on division by zero.

    Args:
        numerator:   The top number.
        denominator: The bottom number.
        default:     Returned when denominator is 0.

    Returns:
        Float result of division, or *default* if denominator is zero.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def per_90(stat: float, minutes_played: float) -> float:
    """
    Normalise *stat* to a per-90-minutes rate.

    Args:
        stat:           Raw counting stat (e.g., total goals).
        minutes_played: Total minutes on the pitch.

    Returns:
        Stat per 90 minutes, or 0.0 if minutes_played is 0.
    """
    return safe_divide(stat * cfg.PER_90_DENOMINATOR, minutes_played)
