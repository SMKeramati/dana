"""Structured logging with correlation ID propagation across services."""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    cid = correlation_id_var.get()
    if not cid:
        cid = uuid.uuid4().hex[:16]
        correlation_id_var.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    correlation_id_var.set(cid)


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "service": getattr(record, "service", "unknown"),
            "correlation_id": get_correlation_id(),
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        extra = getattr(record, "_structured_data", None)
        if extra:
            log_entry["data"] = extra
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


class StructuredLogger:
    """Wraps a stdlib logger to accept keyword arguments as structured data."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, log_level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        exc_info = kwargs.pop("exc_info", None)
        stack_info = kwargs.pop("stack_info", False)
        stacklevel = kwargs.pop("stacklevel", 1)
        extra = kwargs.pop("extra", None)
        if kwargs:
            # Remaining kwargs are structured data
            if extra is None:
                extra = {}
            extra["_structured_data"] = kwargs
        self._logger.log(
            log_level, msg, *args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel + 1,
            extra=extra or {},
        )

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    def isEnabledFor(self, level: int) -> bool:
        return self._logger.isEnabledFor(level)

    def setLevel(self, level: int) -> None:
        self._logger.setLevel(level)

    @property
    def handlers(self) -> list[logging.Handler]:
        return self._logger.handlers


def setup_logging(service_name: str, level: str = "INFO") -> StructuredLogger:
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return StructuredLogger(logger)


def get_logger(name: str, level: str = "INFO") -> StructuredLogger:
    """Convenience alias for setup_logging."""
    return setup_logging(name, level)
