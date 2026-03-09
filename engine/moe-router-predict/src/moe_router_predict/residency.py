"""ExpertResidencyTracker — single source of truth for where each expert lives.

Tracks the current storage tier of every expert across the model.
Updated by AsyncExpertLoader when a load completes, and by the engine
when experts are evicted.
"""

from __future__ import annotations

import threading
from typing import Literal

Tier = Literal["hot", "ram", "ssd", "in_flight"]


class ExpertResidencyTracker:
    """Thread-safe tracker for expert storage location.

    Usage:
        tracker = ExpertResidencyTracker(num_experts=8, default_tier="ssd")
        tracker.mark(expert_id=3, tier="hot")
        print(tracker.where(3))   # → "hot"
        cold = tracker.cold_experts()
    """

    def __init__(
        self,
        num_experts: int,
        default_tier: Tier = "ssd",
    ) -> None:
        self.num_experts = num_experts
        self._residency: dict[int, Tier] = {
            i: default_tier for i in range(num_experts)
        }
        self._lock = threading.Lock()

    def mark(self, expert_id: int, tier: Tier) -> None:
        """Update where an expert is stored."""
        with self._lock:
            self._residency[expert_id] = tier

    def where(self, expert_id: int) -> Tier:
        """Return the current tier of an expert."""
        with self._lock:
            return self._residency.get(expert_id, "ssd")

    def mark_in_flight(self, expert_id: int) -> None:
        """Mark expert as currently being loaded (prevents duplicate loads)."""
        self.mark(expert_id, "in_flight")

    def is_in_flight(self, expert_id: int) -> bool:
        return self.where(expert_id) == "in_flight"

    def is_hot(self, expert_id: int) -> bool:
        return self.where(expert_id) == "hot"

    def cold_experts(self) -> list[int]:
        """Return all expert IDs currently on SSD."""
        with self._lock:
            return [eid for eid, tier in self._residency.items() if tier == "ssd"]

    def hot_experts(self) -> list[int]:
        """Return all expert IDs currently in VRAM."""
        with self._lock:
            return [eid for eid, tier in self._residency.items() if tier == "hot"]

    def ram_experts(self) -> list[int]:
        """Return all expert IDs currently in RAM."""
        with self._lock:
            return [eid for eid, tier in self._residency.items() if tier == "ram"]

    def snapshot(self) -> dict[int, Tier]:
        """Return a copy of the full residency map."""
        with self._lock:
            return dict(self._residency)

    def needs_load(self, expert_id: int) -> bool:
        """Return True if expert needs to be loaded (not hot, not in flight)."""
        tier = self.where(expert_id)
        return tier not in ("hot", "in_flight")

    def summary(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {"hot": 0, "ram": 0, "ssd": 0, "in_flight": 0}
            for tier in self._residency.values():
                counts[tier] = counts.get(tier, 0) + 1
            return counts
