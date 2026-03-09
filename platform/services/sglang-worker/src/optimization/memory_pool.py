"""Custom GPU memory allocation pool with fragmentation prevention.

Daneshbonyan: Internal Design & Development

Implements a buddy-allocator-inspired memory pool that pre-allocates a
contiguous block and hands out aligned sub-allocations.  Free blocks are
coalesced eagerly to prevent fragmentation.  All sizes are rounded up to
the nearest power-of-two multiple of the minimum block size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MemoryBlock:
    """A contiguous region within the pool."""

    offset: int  # byte offset from pool start
    size: int  # size in bytes
    is_free: bool = True
    tag: str = ""  # optional label for debugging


@dataclass
class PoolStats:
    """Allocation statistics."""

    total_allocs: int = 0
    total_frees: int = 0
    peak_used: int = 0
    current_used: int = 0
    fragmentation_events: int = 0
    coalesce_events: int = 0


class GPUMemoryPool:
    """Buddy-allocator-inspired GPU memory pool.

    Daneshbonyan: Internal Design & Development

    Parameters
    ----------
    total_bytes : int
        Total pool capacity.
    min_block_size : int
        Minimum allocation granularity.  All allocations are rounded up
        to a power-of-two multiple of this value.
    alignment : int
        Byte alignment for all returned offsets (must be a power of two).
    """

    def __init__(
        self,
        total_bytes: int = 4 * 1024**3,
        min_block_size: int = 256,
        alignment: int = 256,
    ) -> None:
        assert total_bytes > 0
        assert min_block_size > 0
        assert alignment > 0 and (alignment & (alignment - 1)) == 0, "alignment must be power of 2"

        self._total = total_bytes
        self._min_block = min_block_size
        self._alignment = alignment

        # Start with a single free block spanning the entire pool
        self._blocks: list[MemoryBlock] = [
            MemoryBlock(offset=0, size=total_bytes, is_free=True)
        ]
        self.stats = PoolStats()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _next_power_of_two(n: int) -> int:
        if n <= 0:
            return 1
        return 1 << (n - 1).bit_length()

    def _align_up(self, size: int) -> int:
        """Round *size* up to the nearest allocation granularity."""
        size = max(size, self._min_block)
        size = self._next_power_of_two(size)
        # Also align to self._alignment
        remainder = size % self._alignment
        if remainder:
            size += self._alignment - remainder
        return size

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate(self, requested_bytes: int, tag: str = "") -> int | None:
        """Allocate at least *requested_bytes* from the pool.

        Returns the byte offset of the allocation, or None if the pool
        is exhausted.
        """
        size = self._align_up(requested_bytes)

        # First-fit search
        for idx, block in enumerate(self._blocks):
            if not block.is_free or block.size < size:
                continue

            # Found a suitable free block -- split if much larger
            if block.size > size:
                remainder = MemoryBlock(
                    offset=block.offset + size,
                    size=block.size - size,
                    is_free=True,
                )
                self._blocks.insert(idx + 1, remainder)
                block.size = size

            block.is_free = False
            block.tag = tag

            self.stats.total_allocs += 1
            self.stats.current_used += size
            self.stats.peak_used = max(self.stats.peak_used, self.stats.current_used)

            return block.offset

        logger.warning("Allocation of %d bytes failed (pool exhausted)", requested_bytes)
        return None

    def free(self, offset: int) -> bool:
        """Free a previously allocated block at *offset*.

        Returns True on success, False if the offset was not found or
        already free.
        """
        for idx, block in enumerate(self._blocks):
            if block.offset == offset and not block.is_free:
                block.is_free = True
                block.tag = ""
                self.stats.total_frees += 1
                self.stats.current_used -= block.size

                # Coalesce with neighbours
                self._coalesce(idx)
                return True

        return False

    def _coalesce(self, idx: int) -> None:
        """Merge adjacent free blocks around index *idx*."""
        merged = True
        while merged:
            merged = False
            # Try merge with next
            if idx + 1 < len(self._blocks):
                current = self._blocks[idx]
                nxt = self._blocks[idx + 1]
                if current.is_free and nxt.is_free:
                    current.size += nxt.size
                    self._blocks.pop(idx + 1)
                    self.stats.coalesce_events += 1
                    merged = True
            # Try merge with previous
            if idx > 0:
                prev = self._blocks[idx - 1]
                current = self._blocks[idx]
                if prev.is_free and current.is_free:
                    prev.size += current.size
                    self._blocks.pop(idx)
                    idx -= 1
                    self.stats.coalesce_events += 1
                    merged = True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def total_bytes(self) -> int:
        return self._total

    @property
    def used_bytes(self) -> int:
        return self.stats.current_used

    @property
    def free_bytes(self) -> int:
        return self._total - self.stats.current_used

    @property
    def utilization(self) -> float:
        return self.stats.current_used / self._total if self._total > 0 else 0.0

    @property
    def num_free_blocks(self) -> int:
        return sum(1 for b in self._blocks if b.is_free)

    @property
    def num_used_blocks(self) -> int:
        return sum(1 for b in self._blocks if not b.is_free)

    def fragmentation_ratio(self) -> float:
        """Compute external fragmentation ratio.

        Returns 0.0 when all free space is contiguous, approaching 1.0
        when free space is highly fragmented.
        """
        free_blocks = [b for b in self._blocks if b.is_free]
        if not free_blocks:
            return 0.0
        total_free = sum(b.size for b in free_blocks)
        largest_free = max(b.size for b in free_blocks)
        if total_free == 0:
            return 0.0
        return 1.0 - (largest_free / total_free)

    def dump_layout(self) -> list[dict[str, object]]:
        """Return the current block layout for debugging."""
        return [
            {
                "offset": b.offset,
                "size": b.size,
                "is_free": b.is_free,
                "tag": b.tag,
            }
            for b in self._blocks
        ]

    def reset(self) -> None:
        """Reset pool to initial empty state."""
        self._blocks = [MemoryBlock(offset=0, size=self._total, is_free=True)]
        self.stats = PoolStats()
