"""PlacementOptimizer — frequency-based tier assignment.

Given per-key access counts and tier budgets, assigns each tensor to the
optimal tier (hot > ram > ssd) using a greedy algorithm: sort by access
frequency descending, fill hot tier first, then RAM, overflow to SSD.

This runs synchronously and is called by BackgroundPromoter periodically.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TierAssignment:
    key: str
    recommended_tier: str
    access_count: int
    size_bytes: int


class PlacementOptimizer:
    """Greedy tier optimizer: highest-frequency tensors get the fastest tier.

    Usage:
        optimizer = PlacementOptimizer(hot_budget=4*GB, ram_budget=32*GB)
        assignments = optimizer.optimize(
            access_counts={"expert_0": 100, "expert_1": 50, ...},
            size_map={"expert_0": 64_000_000, ...},
        )
    """

    def __init__(
        self,
        hot_budget_bytes: int = 4 * 1024**3,
        ram_budget_bytes: int = 32 * 1024**3,
    ) -> None:
        self.hot_budget = hot_budget_bytes
        self.ram_budget = ram_budget_bytes

    def optimize(
        self,
        access_counts: dict[str, int],
        size_map: dict[str, int],
    ) -> dict[str, str]:
        """Compute optimal tier assignment for all tracked tensors.

        Args:
            access_counts: {key: number_of_accesses}
            size_map: {key: size_in_bytes}

        Returns:
            {key: "hot" | "ram" | "ssd"}
        """
        # Sort by access frequency descending (ties broken by key for stability)
        sorted_keys = sorted(
            access_counts.keys(),
            key=lambda k: (-access_counts[k], k),
        )

        assignments: dict[str, str] = {}
        hot_used = 0
        ram_used = 0

        for key in sorted_keys:
            size = size_map.get(key, 0)
            if hot_used + size <= self.hot_budget:
                assignments[key] = "hot"
                hot_used += size
            elif ram_used + size <= self.ram_budget:
                assignments[key] = "ram"
                ram_used += size
            else:
                assignments[key] = "ssd"

        return assignments

    def optimize_store(self, store: object) -> dict[str, str]:
        """Convenience: optimize directly from a TieredTensorStore.

        Args:
            store: TieredTensorStore instance

        Returns:
            {key: recommended_tier}
        """
        from tiered_tensor_store.tier_manager import TieredTensorStore
        assert isinstance(store, TieredTensorStore)

        access_counts = store.access_counts()
        size_map: dict[str, int] = {}
        for key in access_counts:
            with store._lock:
                entry = store._entries.get(key)
                if entry:
                    size_map[key] = entry.size_bytes

        return self.optimize(access_counts, size_map)

    def delta(
        self,
        current: dict[str, str],
        recommended: dict[str, str],
    ) -> list[tuple[str, str, str]]:
        """Return (key, from_tier, to_tier) moves needed to reach recommended.

        Only returns moves that change tier (up or down).
        """
        moves = []
        for key, new_tier in recommended.items():
            old_tier = current.get(key, "ssd")
            if old_tier != new_tier:
                moves.append((key, old_tier, new_tier))
        return moves
