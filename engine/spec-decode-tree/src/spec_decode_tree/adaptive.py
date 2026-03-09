"""AdaptiveDraftLength — dynamically tune tree depth and width.

Uses rolling acceptance rate (from AcceptanceTracker) to adaptively
adjust how deep and wide the speculation tree is:

  If acceptance_rate > high_threshold → increase depth/width (more speculation)
  If acceptance_rate < low_threshold  → decrease depth/width (less waste)
"""

from __future__ import annotations

from spec_decode_tree.acceptance import AcceptanceTracker


class AdaptiveDraftLength:
    """Dynamically adjust speculative decoding tree depth and width.

    Usage:
        adaptive = AdaptiveDraftLength(min_depth=1, max_depth=6)
        adaptive.update(tracker)
        depth = adaptive.next_depth()
        width = adaptive.next_width()
    """

    def __init__(
        self,
        min_depth: int = 1,
        max_depth: int = 6,
        initial_depth: int = 3,
        min_width: int = 1,
        max_width: int = 4,
        initial_width: int = 2,
        high_threshold: float = 0.8,
        low_threshold: float = 0.5,
    ) -> None:
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_width = min_width
        self.max_width = max_width
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self._depth = initial_depth
        self._width = initial_width

    def update(self, tracker: AcceptanceTracker) -> None:
        """Update depth/width based on current acceptance rate."""
        rate = tracker.rate()

        if rate > self.high_threshold:
            # High acceptance: speculate more aggressively
            self._depth = min(self._depth + 1, self.max_depth)
            self._width = min(self._width + 1, self.max_width)
        elif rate < self.low_threshold:
            # Low acceptance: speculate less to avoid wasted compute
            self._depth = max(self._depth - 1, self.min_depth)
            self._width = max(self._width - 1, self.min_width)
        # else: maintain current settings

    def next_depth(self) -> int:
        return self._depth

    def next_width(self) -> int:
        return self._width

    def set_depth(self, depth: int) -> None:
        self._depth = max(self.min_depth, min(self.max_depth, depth))

    def set_width(self, width: int) -> None:
        self._width = max(self.min_width, min(self.max_width, width))
