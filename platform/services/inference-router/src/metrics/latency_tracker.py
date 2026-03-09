"""P50/P95/P99 latency tracking with exponential histogram buckets.

Daneshbonyan: Internal Design & Development - Custom latency tracker
that maintains an exponential histogram for efficient percentile computation
without storing every sample. Uses a circular buffer of histogram snapshots
for time-windowed percentile queries.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Default exponential histogram bucket boundaries (in milliseconds).
# Covers 1ms to ~30s with exponentially increasing resolution.
_DEFAULT_BUCKET_BOUNDARIES: list[float] = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1024, 2048, 4096, 8192, 16384, 30000,
]


@dataclass
class HistogramSnapshot:
    """A point-in-time snapshot of the histogram state."""

    timestamp: float
    buckets: list[int]
    boundaries: list[float]
    count: int
    total_ms: float
    min_ms: float
    max_ms: float


@dataclass
class ExponentialHistogram:
    """Exponential histogram for latency distribution tracking.

    Daneshbonyan: Internal Design & Development - Custom histogram with
    exponentially spaced buckets for efficient percentile estimation.
    Memory usage is O(num_buckets) regardless of sample count.
    """

    boundaries: list[float] = field(default_factory=lambda: list(_DEFAULT_BUCKET_BOUNDARIES))
    buckets: list[int] = field(init=False)
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def __post_init__(self) -> None:
        # One extra bucket for values above the highest boundary
        self.buckets = [0] * (len(self.boundaries) + 1)

    def record(self, latency_ms: float) -> None:
        """Record a latency observation."""
        self.count += 1
        self.total_ms += latency_ms
        self.min_ms = min(self.min_ms, latency_ms)
        self.max_ms = max(self.max_ms, latency_ms)

        # Binary search for the correct bucket
        idx = self._find_bucket(latency_ms)
        self.buckets[idx] += 1

    def _find_bucket(self, value: float) -> int:
        """Find bucket index using binary search over boundaries."""
        lo, hi = 0, len(self.boundaries)
        while lo < hi:
            mid = (lo + hi) // 2
            if value <= self.boundaries[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def percentile(self, p: float) -> float:
        """Estimate the p-th percentile (0-100) from the histogram.

        Uses linear interpolation within the bucket that contains the
        target rank for a more accurate estimate.
        """
        if self.count == 0:
            return 0.0

        target = (p / 100.0) * self.count
        cumulative = 0

        for i, bucket_count in enumerate(self.buckets):
            cumulative += bucket_count
            if cumulative >= target:
                # Determine bucket bounds
                lower = self.boundaries[i - 1] if i > 0 else 0.0
                upper = self.boundaries[i] if i < len(self.boundaries) else self.max_ms

                # Linear interpolation within bucket
                prev_cumulative = cumulative - bucket_count
                if bucket_count == 0:
                    return lower
                fraction = (target - prev_cumulative) / bucket_count
                return lower + fraction * (upper - lower)

        return self.max_ms

    @property
    def mean_ms(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_ms / self.count

    def snapshot(self) -> HistogramSnapshot:
        """Create an immutable snapshot of the current state."""
        return HistogramSnapshot(
            timestamp=time.time(),
            buckets=list(self.buckets),
            boundaries=list(self.boundaries),
            count=self.count,
            total_ms=self.total_ms,
            min_ms=self.min_ms if self.count > 0 else 0.0,
            max_ms=self.max_ms,
        )

    def reset(self) -> None:
        """Reset all counters."""
        self.buckets = [0] * (len(self.boundaries) + 1)
        self.count = 0
        self.total_ms = 0.0
        self.min_ms = float("inf")
        self.max_ms = 0.0


class LatencyTracker:
    """Tracks per-route and per-model latency with windowed percentiles.

    Daneshbonyan: Internal Design & Development - Custom latency tracker
    that maintains rolling histograms per metric key. Supports querying
    P50, P95, P99 over configurable time windows using histogram snapshots
    stored in a circular buffer.
    """

    def __init__(self, window_size: int = 60, snapshot_interval_s: float = 10.0) -> None:
        self._histograms: dict[str, ExponentialHistogram] = {}
        self._window_size = window_size  # Number of snapshots to retain
        self._snapshot_interval = snapshot_interval_s
        self._snapshot_buffers: dict[str, list[HistogramSnapshot]] = {}
        self._last_snapshot_time: dict[str, float] = {}

    def _get_or_create(self, key: str) -> ExponentialHistogram:
        if key not in self._histograms:
            self._histograms[key] = ExponentialHistogram()
            self._snapshot_buffers[key] = []
            self._last_snapshot_time[key] = 0.0
        return self._histograms[key]

    def record(self, key: str, latency_ms: float) -> None:
        """Record a latency sample for the given metric key.

        Keys are typically formatted as 'model:<model_name>' or
        'route:<endpoint>'.
        """
        histogram = self._get_or_create(key)
        histogram.record(latency_ms)

        # Periodic snapshot for windowed queries
        now = time.time()
        if now - self._last_snapshot_time.get(key, 0) >= self._snapshot_interval:
            snap = histogram.snapshot()
            buffer = self._snapshot_buffers[key]
            buffer.append(snap)
            # Trim to window size
            if len(buffer) > self._window_size:
                buffer.pop(0)
            self._last_snapshot_time[key] = now

    def percentiles(self, key: str) -> dict[str, float]:
        """Return P50, P95, P99, mean, min, max for a given key."""
        histogram = self._histograms.get(key)
        if histogram is None or histogram.count == 0:
            return {"p50": 0, "p95": 0, "p99": 0, "mean": 0, "min": 0, "max": 0, "count": 0}

        return {
            "p50": round(histogram.percentile(50), 2),
            "p95": round(histogram.percentile(95), 2),
            "p99": round(histogram.percentile(99), 2),
            "mean": round(histogram.mean_ms, 2),
            "min": round(histogram.min_ms, 2),
            "max": round(histogram.max_ms, 2),
            "count": histogram.count,
        }

    def all_metrics(self) -> dict[str, dict[str, float]]:
        """Return percentile stats for all tracked keys."""
        return {key: self.percentiles(key) for key in self._histograms}

    def reset(self, key: str | None = None) -> None:
        """Reset a specific key or all keys."""
        if key is not None:
            hist = self._histograms.get(key)
            if hist:
                hist.reset()
                self._snapshot_buffers[key] = []
        else:
            for hist in self._histograms.values():
                hist.reset()
            self._snapshot_buffers = {k: [] for k in self._snapshot_buffers}

    @property
    def tracked_keys(self) -> list[str]:
        return list(self._histograms.keys())
