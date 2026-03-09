"""AcceptanceTracker — rolling acceptance rate statistics.

Tracks speculative decoding acceptance rates to feed AdaptiveDraftLength.
Records per-depth statistics to understand tree utilization.
"""

from __future__ import annotations

from collections import deque


class AcceptanceTracker:
    """Track acceptance rates for speculative decoding.

    Usage:
        tracker = AcceptanceTracker(window=100)
        tracker.record(accepted=3, proposed=5)
        print(tracker.rate())   # → 0.6
    """

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self._accepted: deque[int] = deque()
        self._proposed: deque[int] = deque()
        self._per_depth: dict[int, list[float]] = {}  # depth → list of rates

    def record(self, accepted: int, proposed: int) -> None:
        """Record one speculative step's outcome."""
        assert 0 <= accepted <= proposed, \
            f"accepted ({accepted}) must be in [0, proposed ({proposed})]"
        self._accepted.append(accepted)
        self._proposed.append(proposed)

        # Trim to window
        while len(self._accepted) > self.window:
            self._accepted.popleft()
            self._proposed.popleft()

    def record_depth(self, depth: int, accepted: int, proposed: int) -> None:
        """Record acceptance for a specific tree depth."""
        self.record(accepted, proposed)
        if depth not in self._per_depth:
            self._per_depth[depth] = []
        rate = accepted / proposed if proposed > 0 else 0.0
        self._per_depth[depth].append(rate)
        # Keep only recent entries
        if len(self._per_depth[depth]) > self.window:
            self._per_depth[depth].pop(0)

    def rate(self) -> float:
        """Rolling acceptance rate over the last `window` steps."""
        total_proposed = sum(self._proposed)
        if total_proposed == 0:
            return 0.0
        return sum(self._accepted) / total_proposed

    def per_depth_rate(self) -> dict[int, float]:
        """Average acceptance rate per tree depth."""
        return {
            depth: sum(rates) / len(rates)
            for depth, rates in self._per_depth.items()
            if rates
        }

    def total_accepted(self) -> int:
        return sum(self._accepted)

    def total_proposed(self) -> int:
        return sum(self._proposed)

    def reset(self) -> None:
        self._accepted.clear()
        self._proposed.clear()
        self._per_depth.clear()
