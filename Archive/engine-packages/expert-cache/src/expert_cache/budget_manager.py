"""VRAMBudgetManager — enforce VRAM capacity limits on the expert cache.

Tracks total bytes of tensors in the "hot" (VRAM) tier and evicts experts
when the budget is exceeded. Works in conjunction with any cache implementation.
"""

from __future__ import annotations

from typing import Union

import torch

from expert_cache.lru_cache import LRUExpertCache
from expert_cache.frequency_cache import FrequencyExpertCache
from expert_cache.predictive_cache import PredictiveExpertCache

AnyCache = Union[LRUExpertCache, FrequencyExpertCache, PredictiveExpertCache]


class VRAMBudgetManager:
    """Enforces a byte-level capacity budget for the hot expert cache.

    Usage:
        budget = VRAMBudgetManager(budget_bytes=4 * 1024**3)
        budget.register(tensor)                    # track when adding to cache
        budget.unregister(tensor)                  # track when evicting
        if not budget.can_fit(new_tensor):
            cache.evict()                          # make room first
    """

    def __init__(self, budget_bytes: int = 4 * 1024**3) -> None:
        self.budget = budget_bytes
        self._used: int = 0
        self._tracked: dict[int, int] = {}  # tensor_id → size_bytes

    def can_fit(self, tensor_or_bytes: Union[torch.Tensor, int]) -> bool:
        """Check if a tensor fits within remaining budget."""
        size = self._size(tensor_or_bytes)
        return self._used + size <= self.budget

    def register(self, tensor: torch.Tensor) -> None:
        """Mark a tensor as resident in VRAM (add to used budget)."""
        tid = id(tensor)
        if tid not in self._tracked:
            size = tensor.element_size() * tensor.numel()
            self._tracked[tid] = size
            self._used += size

    def unregister(self, tensor: torch.Tensor) -> None:
        """Mark a tensor as evicted from VRAM (remove from budget)."""
        tid = id(tensor)
        if tid in self._tracked:
            self._used -= self._tracked.pop(tid)

    def enforce(self, cache: AnyCache) -> int:
        """Evict experts from cache until used bytes fit within budget.

        Returns:
            Number of experts evicted
        """
        evicted_count = 0
        while self._used > self.budget and len(cache) > 0:
            evicted_id = cache.evict()
            if evicted_id is None:
                break
            evicted_count += 1
        return evicted_count

    def used_bytes(self) -> int:
        return self._used

    def available_bytes(self) -> int:
        return max(0, self.budget - self._used)

    def utilization(self) -> float:
        """Return fraction of budget used (0.0–1.0)."""
        if self.budget == 0:
            return 0.0
        return min(1.0, self._used / self.budget)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _size(tensor_or_bytes: Union[torch.Tensor, int]) -> int:
        if isinstance(tensor_or_bytes, int):
            return tensor_or_bytes
        return tensor_or_bytes.element_size() * tensor_or_bytes.numel()
