"""LRU expert cache — baseline Least-Recently-Used implementation.

Uses collections.OrderedDict to track insertion/access order.
Evicts the least-recently-used expert when capacity is exceeded.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch


class LRUExpertCache:
    """Least-Recently-Used expert cache with capacity limit.

    Usage:
        cache = LRUExpertCache(capacity=4)
        cache.put(0, tensor0)
        tensor = cache.get(0)   # returns tensor, marks as recently used
    """

    def __init__(self, capacity: int) -> None:
        assert capacity > 0, "Capacity must be positive"
        self.capacity = capacity
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()

    def get(self, expert_id: int) -> Optional[torch.Tensor]:
        """Get expert tensor, marking it as recently used.

        Returns None on cache miss.
        """
        if expert_id not in self._cache:
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(expert_id)
        return self._cache[expert_id]

    def put(self, expert_id: int, tensor: torch.Tensor) -> Optional[int]:
        """Insert or update an expert.

        Returns the evicted expert_id if capacity was exceeded, else None.
        """
        if expert_id in self._cache:
            self._cache.move_to_end(expert_id)
            self._cache[expert_id] = tensor
            return None

        evicted_id = None
        if len(self._cache) >= self.capacity:
            evicted_id, _ = self._cache.popitem(last=False)  # remove LRU (front)

        self._cache[expert_id] = tensor
        return evicted_id

    def evict(self) -> Optional[int]:
        """Evict the least-recently-used expert.

        Returns evicted expert_id or None if cache is empty.
        """
        if not self._cache:
            return None
        evicted_id, _ = self._cache.popitem(last=False)
        return evicted_id

    def contains(self, expert_id: int) -> bool:
        return expert_id in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def cached_ids(self) -> list[int]:
        return list(self._cache.keys())

    def is_full(self) -> bool:
        return len(self._cache) >= self.capacity
