"""Custom data aggregation pipeline with time-windowed rollups.

Daneshbonyan: Internal R&D -- A streaming aggregation pipeline that
incrementally rolls up raw usage events into configurable time windows
(minute / hour / day / month) with efficient in-memory accumulators and
periodic flush-to-store semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from dana_common.logging import get_logger

logger = get_logger(__name__)


class RollupWindow(StrEnum):
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    MONTH = "month"


def _bucket_key(ts: datetime, window: RollupWindow) -> str:
    """Derive a deterministic bucket key from a timestamp and window."""
    if window is RollupWindow.MINUTE:
        return ts.strftime("%Y-%m-%dT%H:%M")
    if window is RollupWindow.HOUR:
        return ts.strftime("%Y-%m-%dT%H")
    if window is RollupWindow.DAY:
        return ts.strftime("%Y-%m-%d")
    return ts.strftime("%Y-%m")


@dataclass
class UsageEvent:
    """A single raw usage event entering the pipeline."""

    tenant_id: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class RollupBucket:
    """Accumulator for a single (tenant, model, window, bucket_key) slice."""

    tenant_id: str
    model: str
    window: RollupWindow
    bucket_key: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    def accumulate(self, event: UsageEvent) -> None:
        self.total_input_tokens += event.input_tokens
        self.total_output_tokens += event.output_tokens
        self.total_requests += 1
        self.total_latency_ms += event.latency_ms
        self.min_latency_ms = min(self.min_latency_ms, event.latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, event.latency_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "model": self.model,
            "window": self.window.value,
            "bucket_key": self.bucket_key,
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
        }


class UsagePipeline:
    """Streaming aggregation pipeline with multi-window rollups.

    Events are ingested via :meth:`push` and accumulated into per-window
    buckets.  Call :meth:`flush` to drain all accumulated rollups for
    downstream persistence.
    """

    def __init__(self, windows: list[RollupWindow] | None = None) -> None:
        self._windows = windows or [RollupWindow.HOUR, RollupWindow.DAY, RollupWindow.MONTH]
        # {window: {(tenant, model, bucket_key): RollupBucket}}
        self._buckets: dict[RollupWindow, dict[tuple[str, str, str], RollupBucket]] = {
            w: {} for w in self._windows
        }
        self._events_ingested: int = 0

    def push(self, event: UsageEvent) -> None:
        """Ingest a single usage event into all configured window buckets."""
        for window in self._windows:
            bk = _bucket_key(event.timestamp, window)
            composite = (event.tenant_id, event.model, bk)
            bucket = self._buckets[window].get(composite)
            if bucket is None:
                bucket = RollupBucket(
                    tenant_id=event.tenant_id,
                    model=event.model,
                    window=window,
                    bucket_key=bk,
                )
                self._buckets[window][composite] = bucket
            bucket.accumulate(event)
        self._events_ingested += 1

    def push_batch(self, events: list[UsageEvent]) -> int:
        """Ingest a batch of events. Returns count ingested."""
        for e in events:
            self.push(e)
        return len(events)

    def query(
        self,
        tenant_id: str,
        window: RollupWindow,
        model: str | None = None,
    ) -> list[RollupBucket]:
        """Return current rollup buckets matching the query."""
        results: list[RollupBucket] = []
        for (tid, mdl, _bk), bucket in self._buckets[window].items():
            if tid != tenant_id:
                continue
            if model and mdl != model:
                continue
            results.append(bucket)
        return sorted(results, key=lambda b: b.bucket_key)

    def flush(self) -> list[RollupBucket]:
        """Drain all accumulated buckets and return them."""
        all_buckets: list[RollupBucket] = []
        for window_buckets in self._buckets.values():
            all_buckets.extend(window_buckets.values())
            window_buckets.clear()
        logger.info("usage_pipeline.flush", buckets=len(all_buckets), total_events=self._events_ingested)
        return all_buckets

    @property
    def events_ingested(self) -> int:
        return self._events_ingested
