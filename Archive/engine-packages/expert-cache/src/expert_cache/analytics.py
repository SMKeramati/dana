"""CacheAnalytics — hit/miss tracking and self-tuning threshold suggestions.

Tracks per-expert and aggregate cache performance. Suggests updated hot/warm
thresholds based on observed frequency distributions.
"""

from __future__ import annotations

from collections import Counter


class CacheAnalytics:
    """Tracks cache hits, misses, and evictions per expert.

    Usage:
        analytics = CacheAnalytics()
        analytics.record_hit(expert_id=3)
        analytics.record_miss(expert_id=7)
        print(analytics.hit_rate())        # → 0.5
        thresholds = analytics.suggest_thresholds()
    """

    def __init__(self) -> None:
        self._hits: Counter[int] = Counter()
        self._misses: Counter[int] = Counter()
        self._evictions: Counter[int] = Counter()

    def record_hit(self, expert_id: int) -> None:
        self._hits[expert_id] += 1

    def record_miss(self, expert_id: int) -> None:
        self._misses[expert_id] += 1

    def record_eviction(self, expert_id: int) -> None:
        self._evictions[expert_id] += 1

    def hit_rate(self) -> float:
        """Overall cache hit rate (hits / (hits + misses))."""
        total_hits = sum(self._hits.values())
        total_misses = sum(self._misses.values())
        total = total_hits + total_misses
        if total == 0:
            return 0.0
        return total_hits / total

    def per_expert_hit_rate(self) -> dict[int, float]:
        """Hit rate per expert."""
        all_ids = set(self._hits.keys()) | set(self._misses.keys())
        result: dict[int, float] = {}
        for eid in all_ids:
            h = self._hits[eid]
            m = self._misses[eid]
            result[eid] = h / (h + m) if (h + m) > 0 else 0.0
        return result

    def total_accesses(self) -> int:
        return sum(self._hits.values()) + sum(self._misses.values())

    def hottest_experts(self, n: int = 10) -> list[tuple[int, int]]:
        """Return top-n experts by total access count (hits + misses)."""
        access = Counter()
        for eid in set(self._hits) | set(self._misses):
            access[eid] = self._hits[eid] + self._misses[eid]
        return access.most_common(n)

    def suggest_thresholds(self) -> dict[str, float]:
        """Suggest hot/warm thresholds based on observed frequency distribution.

        Strategy:
          hot_threshold  = 75th percentile of access counts
          warm_threshold = 25th percentile of access counts

        Returns:
            {"hot_threshold": float, "warm_threshold": float}
        """
        access = {
            eid: self._hits[eid] + self._misses[eid]
            for eid in set(self._hits) | set(self._misses)
        }
        if not access:
            return {"hot_threshold": 10.0, "warm_threshold": 1.0}

        counts = sorted(access.values())
        n = len(counts)
        p75 = counts[max(0, int(n * 0.75) - 1)]
        p25 = counts[max(0, int(n * 0.25) - 1)]
        return {
            "hot_threshold": max(1.0, float(p75)),
            "warm_threshold": max(0.0, float(p25)),
        }

    def reset(self) -> None:
        self._hits.clear()
        self._misses.clear()
        self._evictions.clear()

    def summary(self) -> dict:
        return {
            "hit_rate": self.hit_rate(),
            "total_accesses": self.total_accesses(),
            "total_hits": sum(self._hits.values()),
            "total_misses": sum(self._misses.values()),
            "total_evictions": sum(self._evictions.values()),
            "unique_experts_seen": len(set(self._hits) | set(self._misses)),
        }
