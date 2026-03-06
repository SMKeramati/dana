"""Custom hybrid KV cache manager.

Daneshbonyan: Internal Design & Development

GPU/CPU split KV cache with LRU + frequency-based eviction.  Hot entries
stay on the GPU side while cold entries are demoted to CPU memory.  Eviction
combines recency (LRU timestamp) and access frequency into a single score so
that both recently-used *and* frequently-used cache lines are retained.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class CacheTier(Enum):
    """Storage tier for a KV cache entry."""

    GPU = "gpu"
    CPU = "cpu"


@dataclass
class CacheEntry:
    """A single cached key-value pair for one sequence position."""

    seq_id: str
    position: int
    key: np.ndarray
    value: np.ndarray
    tier: CacheTier = CacheTier.GPU
    access_count: int = 0
    last_access_ts: float = field(default_factory=time.monotonic)
    created_ts: float = field(default_factory=time.monotonic)

    @property
    def size_bytes(self) -> int:
        return self.key.nbytes + self.value.nbytes

    def touch(self) -> None:
        """Record an access."""
        self.access_count += 1
        self.last_access_ts = time.monotonic()


@dataclass
class CacheStats:
    """Aggregate statistics for the KV cache."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    promotions: int = 0
    demotions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class HybridKVCacheManager:
    """Hybrid GPU/CPU KV cache with LRU + frequency-based eviction.

    Daneshbonyan: Internal Design & Development

    The eviction score for each entry is::

        score = alpha * recency_norm + (1 - alpha) * frequency_norm

    where *recency_norm* maps the time-since-last-access to [0, 1] and
    *frequency_norm* maps the access count to [0, 1].  Entries with the
    **lowest** score are evicted first.
    """

    def __init__(
        self,
        gpu_capacity_bytes: int = 4 * 1024**3,  # 4 GiB default
        cpu_capacity_bytes: int = 16 * 1024**3,  # 16 GiB default
        num_heads: int = 32,
        head_dim: int = 128,
        alpha: float = 0.6,
    ) -> None:
        self._gpu_capacity = gpu_capacity_bytes
        self._cpu_capacity = cpu_capacity_bytes
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._alpha = alpha  # weight for recency vs frequency

        # Keyed by (seq_id, position)
        self._entries: dict[tuple[str, int], CacheEntry] = {}
        self._gpu_used: int = 0
        self._cpu_used: int = 0
        self.stats = CacheStats()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def gpu_used_bytes(self) -> int:
        return self._gpu_used

    @property
    def cpu_used_bytes(self) -> int:
        return self._cpu_used

    @property
    def total_entries(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def _make_kv_pair(self, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Create a KV pair shaped (num_heads, head_dim) in float16."""
        if rng is None:
            rng = np.random.default_rng()
        k = rng.standard_normal((self._num_heads, self._head_dim)).astype(np.float16)
        v = rng.standard_normal((self._num_heads, self._head_dim)).astype(np.float16)
        return k, v

    def put(
        self,
        seq_id: str,
        position: int,
        key: np.ndarray,
        value: np.ndarray,
    ) -> CacheTier:
        """Insert or update a KV cache entry.

        Returns the tier where the entry was placed.
        """
        cache_key = (seq_id, position)
        entry_size = key.nbytes + value.nbytes

        # If it already exists, update in-place
        if cache_key in self._entries:
            old = self._entries[cache_key]
            old_size = old.size_bytes
            self._adjust_usage(old.tier, -old_size)
            old.key = key
            old.value = value
            old.touch()
            self._adjust_usage(old.tier, entry_size)
            return old.tier

        # Try GPU first
        if self._gpu_used + entry_size <= self._gpu_capacity:
            tier = CacheTier.GPU
        elif self._cpu_used + entry_size <= self._cpu_capacity:
            tier = CacheTier.CPU
        else:
            # Need eviction
            freed = self._evict(entry_size, prefer_tier=CacheTier.CPU)
            if not freed:
                freed = self._evict(entry_size, prefer_tier=CacheTier.GPU)
            tier = CacheTier.CPU if self._cpu_used + entry_size <= self._cpu_capacity else CacheTier.GPU

        entry = CacheEntry(
            seq_id=seq_id,
            position=position,
            key=key,
            value=value,
            tier=tier,
        )
        self._entries[cache_key] = entry
        self._adjust_usage(tier, entry_size)
        return tier

    def get(self, seq_id: str, position: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Retrieve a cached KV pair. Returns None on miss."""
        cache_key = (seq_id, position)
        entry = self._entries.get(cache_key)
        if entry is None:
            self.stats.misses += 1
            return None
        entry.touch()
        self.stats.hits += 1
        return entry.key, entry.value

    def get_sequence(self, seq_id: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Get all cached positions for a sequence, keyed by position."""
        result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for (sid, pos), entry in self._entries.items():
            if sid == seq_id:
                entry.touch()
                self.stats.hits += 1
                result[pos] = (entry.key, entry.value)
        return result

    def evict_sequence(self, seq_id: str) -> int:
        """Remove all entries for a given sequence. Returns bytes freed."""
        to_remove = [k for k in self._entries if k[0] == seq_id]
        freed = 0
        for cache_key in to_remove:
            entry = self._entries.pop(cache_key)
            self._adjust_usage(entry.tier, -entry.size_bytes)
            freed += entry.size_bytes
            self.stats.evictions += 1
        return freed

    # ------------------------------------------------------------------
    # Eviction logic
    # ------------------------------------------------------------------

    def _compute_eviction_score(self, entry: CacheEntry) -> float:
        """Lower score = more likely to be evicted."""
        now = time.monotonic()
        age = now - entry.last_access_ts + 1e-6  # seconds since last access

        # Recency score: inverse of age, clamped and normalised via sigmoid-like
        recency = 1.0 / (1.0 + age)

        # Frequency score: log-scaled access count
        frequency = np.log1p(entry.access_count) / (np.log1p(entry.access_count) + 1.0)

        return float(self._alpha * recency + (1.0 - self._alpha) * frequency)

    def _evict(self, needed_bytes: int, prefer_tier: CacheTier) -> bool:
        """Evict entries from *prefer_tier* until *needed_bytes* are freed.

        Returns True if enough space was freed.
        """
        candidates = [
            (k, e) for k, e in self._entries.items() if e.tier == prefer_tier
        ]
        if not candidates:
            return False

        # Sort by eviction score ascending (lowest score evicted first)
        candidates.sort(key=lambda kv: self._compute_eviction_score(kv[1]))

        freed = 0
        for cache_key, entry in candidates:
            if freed >= needed_bytes:
                break
            freed += entry.size_bytes
            self._adjust_usage(entry.tier, -entry.size_bytes)
            del self._entries[cache_key]
            self.stats.evictions += 1

        return freed >= needed_bytes

    def promote(self, seq_id: str, position: int) -> bool:
        """Promote an entry from CPU to GPU tier if space is available."""
        cache_key = (seq_id, position)
        entry = self._entries.get(cache_key)
        if entry is None or entry.tier == CacheTier.GPU:
            return False

        if self._gpu_used + entry.size_bytes > self._gpu_capacity:
            # Try evicting from GPU first
            if not self._evict(entry.size_bytes, prefer_tier=CacheTier.GPU):
                return False

        self._adjust_usage(CacheTier.CPU, -entry.size_bytes)
        entry.tier = CacheTier.GPU
        self._adjust_usage(CacheTier.GPU, entry.size_bytes)
        self.stats.promotions += 1
        return True

    def demote(self, seq_id: str, position: int) -> bool:
        """Demote an entry from GPU to CPU tier."""
        cache_key = (seq_id, position)
        entry = self._entries.get(cache_key)
        if entry is None or entry.tier == CacheTier.CPU:
            return False

        if self._cpu_used + entry.size_bytes > self._cpu_capacity:
            return False

        self._adjust_usage(CacheTier.GPU, -entry.size_bytes)
        entry.tier = CacheTier.CPU
        self._adjust_usage(CacheTier.CPU, entry.size_bytes)
        self.stats.demotions += 1
        return True

    def rebalance(self) -> int:
        """Promote hot CPU entries to GPU and demote cold GPU entries.

        Returns the number of entries moved.
        """
        gpu_entries = [
            (k, e) for k, e in self._entries.items() if e.tier == CacheTier.GPU
        ]
        cpu_entries = [
            (k, e) for k, e in self._entries.items() if e.tier == CacheTier.CPU
        ]

        if not gpu_entries or not cpu_entries:
            return 0

        # Find the coldest GPU entry and hottest CPU entry
        gpu_entries.sort(key=lambda kv: self._compute_eviction_score(kv[1]))
        cpu_entries.sort(key=lambda kv: self._compute_eviction_score(kv[1]), reverse=True)

        moved = 0
        for (cpu_key, cpu_entry), (gpu_key, gpu_entry) in zip(
            cpu_entries, gpu_entries
        ):
            cpu_score = self._compute_eviction_score(cpu_entry)
            gpu_score = self._compute_eviction_score(gpu_entry)

            if cpu_score > gpu_score:
                # Swap: demote GPU entry, promote CPU entry
                if self.demote(gpu_key[0], gpu_key[1]):
                    if self.promote(cpu_key[0], cpu_key[1]):
                        moved += 2
                    else:
                        # Roll back demotion
                        self.promote(gpu_key[0], gpu_key[1])
            else:
                break

        return moved

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _adjust_usage(self, tier: CacheTier, delta: int) -> None:
        if tier == CacheTier.GPU:
            self._gpu_used += delta
        else:
            self._cpu_used += delta

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()
        self._gpu_used = 0
        self._cpu_used = 0
