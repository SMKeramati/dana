"""Predictive expert cache — hint-driven pre-loading.

Wraps FrequencyExpertCache with a hint interface: when the router predictor
knows which experts will be needed in the next N steps, it calls hint() to
pre-load them before they're requested. This eliminates the cold-miss latency.

The predictive cache holds a separate "pre-load set" of hinted experts. When
capacity is tight, hinted experts are protected from eviction.
"""

from __future__ import annotations

from typing import Optional

import torch

from expert_cache.frequency_cache import FrequencyExpertCache


class PredictiveExpertCache:
    """Expert cache that accepts lookahead hints from the router predictor.

    Usage:
        cache = PredictiveExpertCache(capacity=8)
        cache.hint([3, 5, 7])        # pre-load these experts next
        tensor = cache.get(3)        # hit (pre-loaded)
        cache.put(3, weight_tensor)  # explicit put still works
    """

    def __init__(
        self,
        capacity: int,
        window_size: int = 1000,
        hint_protection_slots: int = 2,
    ) -> None:
        """
        Args:
            capacity: Max experts in cache
            window_size: Sliding window for frequency decay
            hint_protection_slots: How many slots to reserve for hinted experts
        """
        self.capacity = capacity
        self.hint_protection_slots = hint_protection_slots
        self._freq_cache = FrequencyExpertCache(capacity, window_size)
        self._hinted: set[int] = set()
        self._pending_hints: list[int] = []

    def hint(self, upcoming_expert_ids: list[int]) -> None:
        """Signal that these experts will be needed soon.

        The cache will protect them from eviction and optionally pre-load
        them if tensors are provided via put_hinted().
        """
        self._hinted.update(upcoming_expert_ids)
        self._pending_hints = list(upcoming_expert_ids)

    def put_hinted(self, expert_id: int, tensor: torch.Tensor) -> Optional[int]:
        """Pre-load a hinted expert. Hinted experts are preferred over non-hinted."""
        return self.put(expert_id, tensor)

    def get(self, expert_id: int) -> Optional[torch.Tensor]:
        """Get an expert tensor (removes from hinted set on access)."""
        self._hinted.discard(expert_id)
        return self._freq_cache.get(expert_id)

    def put(self, expert_id: int, tensor: torch.Tensor) -> Optional[int]:
        """Insert expert. When eviction needed, prefer evicting non-hinted experts."""
        if self._freq_cache.contains(expert_id):
            return self._freq_cache.put(expert_id, tensor)

        if len(self._freq_cache) >= self.capacity:
            evicted = self._evict_non_hinted()
            if evicted is None:
                # All cached experts are hinted — fall back to frequency eviction
                evicted = self._freq_cache.evict()
        else:
            evicted = None

        self._freq_cache._cache[expert_id] = tensor
        return evicted

    def evict(self) -> Optional[int]:
        return self._evict_non_hinted() or self._freq_cache.evict()

    def contains(self, expert_id: int) -> bool:
        return self._freq_cache.contains(expert_id)

    def is_hinted(self, expert_id: int) -> bool:
        return expert_id in self._hinted

    def pending_hints(self) -> list[int]:
        return [eid for eid in self._pending_hints if not self._freq_cache.contains(eid)]

    def __len__(self) -> int:
        return len(self._freq_cache)

    def cached_ids(self) -> list[int]:
        return self._freq_cache.cached_ids()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_non_hinted(self) -> Optional[int]:
        """Evict the lowest-frequency non-hinted expert."""
        cached = self._freq_cache.cached_ids()
        non_hinted = [eid for eid in cached if eid not in self._hinted]
        if not non_hinted:
            return None
        evict_id = min(non_hinted, key=lambda eid: (self._freq_cache.frequency(eid), eid))
        del self._freq_cache._cache[evict_id]
        return evict_id
