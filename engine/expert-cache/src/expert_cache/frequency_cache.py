"""Frequency-aware expert cache.

Combines access frequency (LFU) with a sliding window to avoid keeping stale
experts that were hot in the distant past. Each access increments the expert's
count; the window decays counts older than `window_size` steps.

Eviction policy: evict the expert with the lowest sliding-window frequency.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Optional

import torch


class FrequencyExpertCache:
    """LFU expert cache with sliding-window frequency decay.

    Usage:
        cache = FrequencyExpertCache(capacity=4, window_size=1000)
        cache.put(0, tensor0)
        tensor = cache.get(0)   # increments frequency for expert 0
    """

    def __init__(self, capacity: int, window_size: int = 1000) -> None:
        assert capacity > 0
        self.capacity = capacity
        self.window_size = window_size

        self._cache: dict[int, torch.Tensor] = {}
        self._freq: Counter[int] = Counter()
        self._window: deque[int] = deque()  # expert_id per access step
        self._step: int = 0

    def get(self, expert_id: int) -> Optional[torch.Tensor]:
        """Get expert tensor and increment its access frequency."""
        if expert_id not in self._cache:
            return None
        self._record_access(expert_id)
        return self._cache[expert_id]

    def put(self, expert_id: int, tensor: torch.Tensor) -> Optional[int]:
        """Insert or update an expert.

        Returns the evicted expert_id if capacity was exceeded, else None.
        """
        if expert_id in self._cache:
            self._cache[expert_id] = tensor
            return None

        evicted_id = None
        if len(self._cache) >= self.capacity:
            evicted_id = self._evict_lowest_freq()

        self._cache[expert_id] = tensor
        return evicted_id

    def evict(self) -> Optional[int]:
        """Evict the lowest-frequency expert."""
        if not self._cache:
            return None
        return self._evict_lowest_freq()

    def frequency(self, expert_id: int) -> int:
        return self._freq[expert_id]

    def contains(self, expert_id: int) -> bool:
        return expert_id in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def cached_ids(self) -> list[int]:
        return list(self._cache.keys())

    def all_frequencies(self) -> dict[int, int]:
        return dict(self._freq)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_access(self, expert_id: int) -> None:
        self._freq[expert_id] += 1
        self._window.append(expert_id)
        self._step += 1

        # Decay: slide window
        while len(self._window) > self.window_size:
            old = self._window.popleft()
            if self._freq[old] > 0:
                self._freq[old] -= 1
            if self._freq[old] == 0:
                del self._freq[old]

    def _evict_lowest_freq(self) -> int:
        # Among cached experts, evict the one with lowest frequency
        # (tie-broken by smallest id for stability)
        cached = list(self._cache.keys())
        evict_id = min(cached, key=lambda eid: (self._freq.get(eid, 0), eid))
        del self._cache[evict_id]
        return evict_id
