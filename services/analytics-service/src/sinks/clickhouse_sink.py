"""ClickHouse event sink for Dana Analytics.

Writes all event types to the dana_analytics ClickHouse database using the
HTTP interface. Batches events for efficiency and retries on transient errors.

Daneshbonyan: Internal R&D - Custom Analytics Pipeline
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dana_common.logging import get_logger

logger = get_logger(__name__)

_CH_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse")
_CH_PORT = os.getenv("CLICKHOUSE_PORT", "8123")
_CH_DB = os.getenv("CLICKHOUSE_DB", "dana_analytics")
_CH_USER = os.getenv("CLICKHOUSE_USER", "default")
_CH_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")

_BASE_URL = f"http://{_CH_HOST}:{_CH_PORT}/"


# ---------------------------------------------------------------------------
# Event dataclasses — map 1:1 to ClickHouse table columns
# ---------------------------------------------------------------------------


@dataclass
class ApiRequestEvent:
    request_id: str
    user_id: str
    api_key_prefix: str
    endpoint: str
    method: str
    status_code: int
    response_time_ms: int
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tier: str
    country_code: str = ""
    error_code: str = ""
    streaming: int = 0
    cached: int = 0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


@dataclass
class UserEvent:
    user_id: str
    event_type: str
    properties: dict[str, Any]
    ip_address: str = ""
    user_agent: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


@dataclass
class InferenceEvent:
    job_id: str
    user_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    time_to_first_token_ms: int
    total_latency_ms: int
    tokens_per_second: float
    speculative_accepted: int = 0
    speculative_total: int = 0
    batch_size: int = 1
    gpu_memory_used_gb: float = 0.0
    worker_id: str = ""
    cache_hit: int = 0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


@dataclass
class BillingEvent:
    user_id: str
    org_id: str
    event_type: str
    amount_cents: int
    total_tokens: int
    model: str
    tier: str
    invoice_id: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


@dataclass
class ErrorEvent:
    service: str
    error_type: str
    error_code: str
    message: str
    user_id: str = ""
    request_id: str = ""
    stack_trace: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# ---------------------------------------------------------------------------
# ClickHouse HTTP client
# ---------------------------------------------------------------------------

_TABLE_MAP: dict[type, str] = {
    ApiRequestEvent: "api_requests",
    UserEvent: "user_events",
    InferenceEvent: "inference_events",
    BillingEvent: "billing_events",
    ErrorEvent: "error_events",
}


def _row_to_json_line(event: Any) -> str:
    d = asdict(event)
    # Serialize nested dicts (e.g. UserEvent.properties) to JSON strings
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = json.dumps(v, ensure_ascii=False)
    return json.dumps(d, ensure_ascii=False)


def _execute(query: str, data: bytes | None = None, max_retries: int = 3) -> None:
    """Send a query to ClickHouse HTTP interface with retry logic."""
    params = urlencode({
        "query": query,
        "user": _CH_USER,
        "password": _CH_PASSWORD,
        "database": _CH_DB,
    })
    url = _BASE_URL + "?" + params
    req = Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")

    delay = 1.0
    for attempt in range(max_retries):
        try:
            with urlopen(req, timeout=10) as resp:
                if resp.status not in (200, 204):
                    raise RuntimeError(f"ClickHouse error {resp.status}: {resp.read()[:200]}")
            return
        except (URLError, RuntimeError) as exc:
            if attempt == max_retries - 1:
                logger.error("clickhouse_write_failed", error=str(exc), query=query[:100])
                raise
            logger.warning("clickhouse_retry", attempt=attempt + 1, error=str(exc))
            time.sleep(delay)
            delay *= 2


class ClickHouseSink:
    """Writes analytics events to ClickHouse in micro-batches.

    Usage::

        sink = ClickHouseSink()
        sink.write(ApiRequestEvent(...))
        sink.flush()  # or let auto-flush handle it
    """

    def __init__(self, batch_size: int = 100, auto_flush_interval: float = 5.0) -> None:
        self._batch_size = batch_size
        self._auto_flush_interval = auto_flush_interval
        self._buffers: dict[str, list[str]] = {}
        self._last_flush = time.monotonic()
        self._written = 0
        self._errors = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, event: Any) -> None:
        table = _TABLE_MAP.get(type(event))
        if table is None:
            logger.warning("unknown_event_type", event_type=type(event).__name__)
            return
        self._buffers.setdefault(table, []).append(_row_to_json_line(event))
        if (
            len(self._buffers[table]) >= self._batch_size
            or time.monotonic() - self._last_flush > self._auto_flush_interval
        ):
            self.flush()

    def flush(self) -> None:
        for table, rows in list(self._buffers.items()):
            if not rows:
                continue
            payload = "\n".join(rows).encode("utf-8")
            query = f"INSERT INTO {_CH_DB}.{table} FORMAT JSONEachRow"
            try:
                _execute(query, data=payload)
                self._written += len(rows)
                logger.debug(
                    "clickhouse_flush",
                    table=table,
                    rows=len(rows),
                    total_written=self._written,
                )
            except Exception as exc:
                self._errors += 1
                logger.error("clickhouse_flush_error", table=table, rows=len(rows), error=str(exc))
            finally:
                self._buffers[table] = []
        self._last_flush = time.monotonic()

    def stats(self) -> dict[str, int]:
        return {"written": self._written, "errors": self._errors}

    def ping(self) -> bool:
        """Return True if ClickHouse is reachable."""
        try:
            _execute("SELECT 1")
            return True
        except Exception:
            return False


# Module-level default sink (shared across the analytics service)
_default_sink: ClickHouseSink | None = None


def get_sink() -> ClickHouseSink:
    global _default_sink
    if _default_sink is None:
        _default_sink = ClickHouseSink()
    return _default_sink
