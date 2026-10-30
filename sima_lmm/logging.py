"""Logging helpers owned by LLiMa.

This module keeps the small logging interface used throughout the project while
avoiding a runtime dependency on the retired ``sima-utils`` package.
"""

from __future__ import annotations

from contextlib import ContextDecorator
from enum import Enum
import logging
from pathlib import Path
import sys


class LogLevel(Enum):
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET


def _caller_logger() -> logging.Logger:
    frame = sys._getframe(2)
    return logging.getLogger(frame.f_globals.get("__name__", ""))


def sima_log_dbg(msg: str, *args) -> None:
    _caller_logger().debug(msg, *args)


def sima_log_info(msg: str, *args) -> None:
    _caller_logger().info(msg, *args)


def sima_log_warning(msg: str, *args) -> None:
    _caller_logger().warning(msg, *args)


def sima_log_error(msg: str, *args) -> None:
    _caller_logger().error(msg, *args)


def sima_log_critical(msg: str, *args) -> None:
    _caller_logger().critical(msg, *args)


def sima_log_exception(msg: str, *args) -> None:
    _caller_logger().exception(msg, *args)


def sima_log(level: LogLevel, msg: str, *args) -> None:
    _caller_logger().log(level.value, msg, *args)


def set_logging_level(level: LogLevel) -> None:
    logging.getLogger().setLevel(level.value)


class ScopedLogLevel(ContextDecorator):
    """Temporarily override the root logger level."""

    def __init__(self, log_level: int):
        self._level = LogLevel(log_level)
        self._previous_level: int | None = None

    def __enter__(self):
        if self._level is not LogLevel.NOTSET:
            root = logging.getLogger()
            self._previous_level = root.level
            root.setLevel(self._level.value)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._previous_level is not None:
            logging.getLogger().setLevel(self._previous_level)
        return False


def scoped_log_level_decorator(log_level: int):
    def decorate(func):
        def wrapped(*args, **kwargs):
            with ScopedLogLevel(log_level):
                return func(*args, **kwargs)

        return wrapped

    return decorate


class UserFacingException(Exception):
    pass


def configure_runtime_logging(
    *, log_level: str | int | None = None, console: bool = False, log_file: str | Path = "run.log"
) -> None:
    """Configure the CLI runtime with a fresh file log and optional console output."""

    handler_level = logging.INFO if log_level is None else log_level
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handlers: list[logging.Handler] = [logging.FileHandler(log_file, mode="w")]
    if console:
        handlers.append(logging.StreamHandler(sys.stderr))
    for handler in handlers:
        handler.setLevel(handler_level)
        handler.setFormatter(formatter)

    logging.basicConfig(level=logging.DEBUG, handlers=handlers, force=True)

