"""Custom time-window aggregation for usage data.

Aggregates raw token-usage records into hourly, daily, and monthly rollups
that feed into billing invoices and analytics dashboards.

Daneshbonyan: Internal R&D
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from dana_common.logging import get_logger

logger = get_logger(__name__)


class TimeWindow(StrEnum):
    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"


@dataclass(frozen=True)
class UsageRecord:
    """A single raw usage event."""

    org_id: str
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: float  # unix epoch


@dataclass(frozen=True)
class AggregatedUsage:
    """A rolled-up usage bucket."""

    org_id: str
    model: str
    window: TimeWindow
    bucket_key: str  # e.g. "2026-03-06T14" for hourly
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    request_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "org_id": self.org_id,
            "model": self.model,
            "window": self.window.value,
            "bucket_key": self.bucket_key,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
        }


def _bucket_key(ts: float, window: TimeWindow) -> str:
    """Derive the bucket key for *ts* at the given *window* granularity."""
    dt = datetime.fromtimestamp(ts, tz=UTC)
    if window is TimeWindow.HOURLY:
        return dt.strftime("%Y-%m-%dT%H")
    if window is TimeWindow.DAILY:
        return dt.strftime("%Y-%m-%d")
    # monthly
    return dt.strftime("%Y-%m")


@dataclass
class UsageAggregator:
    """Incrementally aggregates :class:`UsageRecord` objects into time-windowed
    rollups.

    Keeps an in-memory accumulator that can be flushed on demand.  In
    production the flushed data is persisted to the billing database and
    forwarded to the analytics pipeline.
    """

    _accumulators: dict[
        tuple[str, str, TimeWindow, str],
        dict[str, int],
    ] = field(default_factory=lambda: defaultdict(lambda: {
        "input_tokens": 0,
        "output_tokens": 0,
        "request_count": 0,
    }))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, record: UsageRecord, windows: list[TimeWindow] | None = None) -> None:
        """Add *record* to the internal accumulators for each requested window."""
        if windows is None:
            windows = list(TimeWindow)

        for window in windows:
            key = (record.org_id, record.model, window, _bucket_key(record.timestamp, window))
            acc = self._accumulators[key]
            acc["input_tokens"] += record.input_tokens
            acc["output_tokens"] += record.output_tokens
            acc["request_count"] += 1

    def flush(self) -> list[AggregatedUsage]:
        """Flush all accumulated buckets and return them as a list."""
        results: list[AggregatedUsage] = []
        for (org_id, model, window, bucket_key), acc in self._accumulators.items():
            inp = acc["input_tokens"]
            out = acc["output_tokens"]
            results.append(
                AggregatedUsage(
                    org_id=org_id,
                    model=model,
                    window=window,
                    bucket_key=bucket_key,
                    total_input_tokens=inp,
                    total_output_tokens=out,
                    total_tokens=inp + out,
                    request_count=acc["request_count"],
                )
            )
        self._accumulators.clear()
        logger.info("usage_aggregator.flush", buckets=len(results))
        return results

    def peek(self, org_id: str, model: str, window: TimeWindow) -> list[AggregatedUsage]:
        """Return current (un-flushed) aggregated data for a given org/model/window."""
        results: list[AggregatedUsage] = []
        for (o, m, w, bk), acc in self._accumulators.items():
            if o == org_id and m == model and w == window:
                inp = acc["input_tokens"]
                out = acc["output_tokens"]
                results.append(
                    AggregatedUsage(
                        org_id=o,
                        model=m,
                        window=w,
                        bucket_key=bk,
                        total_input_tokens=inp,
                        total_output_tokens=out,
                        total_tokens=inp + out,
                        request_count=acc["request_count"],
                    )
                )
        return results
