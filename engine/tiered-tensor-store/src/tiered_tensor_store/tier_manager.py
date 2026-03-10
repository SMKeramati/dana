"""TieredTensorStore — central registry for VRAM/RAM/SSD tensor management.

Tiers (CPU simulation mode — GPU paths activated when torch.cuda.is_available()):
  "hot"  → CPU tensor (simulates VRAM; will use .cuda() in production)
  "ram"  → numpy.memmap-backed tensor (real RAM-tier I/O, works on CPU)
  "ssd"  → .pt file on disk (torch.save/load, simulates NVMe)

Usage:
    store = TieredTensorStore(base_dir="/tmp/dana_store")
    store.store("expert_0_w1", weight_tensor, tier="ram")
    tensor = store.load("expert_0_w1")   # promotes to hot on first access
    store.evict("expert_0_w1", to_tier="ssd")
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import torch


class Tier(str, Enum):
    HOT = "hot"    # VRAM (or CPU tensor in sim mode)
    RAM = "ram"    # System RAM via mmap
    SSD = "ssd"    # NVMe SSD via file I/O


@dataclass
class TensorEntry:
    tier: Tier
    tensor: Optional[torch.Tensor]     # in-memory tensor (hot/ram tiers)
    path: Optional[Path]               # file path (ram mmap file or ssd .pt file)
    shape: tuple
    dtype: torch.dtype
    access_count: int = 0
    size_bytes: int = 0

    def __post_init__(self) -> None:
        if self.size_bytes == 0 and self.tensor is not None:
            self.size_bytes = self.tensor.element_size() * self.tensor.numel()
        elif self.size_bytes == 0:
            # Estimate from shape + dtype
            elem = torch.empty(1, dtype=self.dtype).element_size()
            numel = 1
            for s in self.shape:
                numel *= s
            self.size_bytes = elem * numel


class TieredTensorStore:
    """Central registry managing tensors across VRAM / RAM / SSD.

    Thread-safe: uses a per-store lock for all mutations.
    """

    def __init__(
        self,
        base_dir: str = "/tmp/dana_store",
        hot_budget_bytes: int = 4 * 1024**3,   # 4GB
        ram_budget_bytes: int = 32 * 1024**3,  # 32GB
        auto_promote: bool = True,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.hot_budget = hot_budget_bytes
        self.ram_budget = ram_budget_bytes
        self.auto_promote = auto_promote

        self._entries: dict[str, TensorEntry] = {}
        self._lock = threading.Lock()
        self._hot_used: int = 0
        self._ram_used: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, key: str, tensor: torch.Tensor, tier: str = "ram") -> None:
        """Store a tensor in the specified tier.

        Args:
            key: Unique identifier (e.g. "layer0_expert3_w1")
            tensor: Float tensor to store
            tier: "hot", "ram", or "ssd"
        """
        t = Tier(tier)
        with self._lock:
            # Remove existing entry if present
            if key in self._entries:
                self._evict_locked(key, Tier.SSD)

            entry = self._write_to_tier(key, tensor, t)
            self._entries[key] = entry

    def load(self, key: str) -> torch.Tensor:
        """Load a tensor, promoting to hot tier on access.

        Args:
            key: Tensor key

        Returns:
            Tensor in hot tier (GPU if available, else CPU)

        Raises:
            KeyError: if key not found
        """
        with self._lock:
            if key not in self._entries:
                raise KeyError(f"Key '{key}' not found in store")

            entry = self._entries[key]
            entry.access_count += 1

            if entry.tier == Tier.HOT:
                assert entry.tensor is not None
                return entry.tensor

            # Promote to hot
            tensor = self._read_from_tier(entry)
            if self.auto_promote:
                self._make_hot(key, tensor, entry)
                return entry.tensor  # type: ignore[return-value]  # _make_hot guarantees not None
            return tensor

    def evict(self, key: str, to_tier: str = "ssd") -> None:
        """Manually evict a tensor to a lower tier."""
        t = Tier(to_tier)
        with self._lock:
            self._evict_locked(key, t)

    def tier_of(self, key: str) -> str:
        """Return the current tier of a stored tensor."""
        with self._lock:
            if key not in self._entries:
                raise KeyError(f"Key '{key}' not found in store")
            return self._entries[key].tier.value

    def access_count(self, key: str) -> int:
        with self._lock:
            if key not in self._entries:
                return 0
            return self._entries[key].access_count

    def all_keys(self) -> list[str]:
        with self._lock:
            return list(self._entries.keys())

    def access_counts(self) -> dict[str, int]:
        with self._lock:
            return {k: e.access_count for k, e in self._entries.items()}

    def hot_used_bytes(self) -> int:
        return self._hot_used

    def ram_used_bytes(self) -> int:
        return self._ram_used

    def stats(self) -> dict:
        with self._lock:
            tiers: dict[str, int] = {t.value: 0 for t in Tier}
            for e in self._entries.values():
                tiers[e.tier.value] += 1
            return {
                "total_keys": len(self._entries),
                "per_tier": tiers,
                "hot_used_bytes": self._hot_used,
                "ram_used_bytes": self._ram_used,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_to_tier(self, key: str, tensor: torch.Tensor, tier: Tier) -> TensorEntry:
        size = tensor.element_size() * tensor.numel()

        if tier == Tier.HOT:
            hot_tensor = self._to_hot(tensor)
            self._hot_used += size
            return TensorEntry(
                tier=Tier.HOT, tensor=hot_tensor, path=None,
                shape=tuple(tensor.shape), dtype=tensor.dtype, size_bytes=size,
            )

        elif tier == Tier.RAM:
            import numpy as np
            path = self.base_dir / f"{key}.mmap"
            arr = tensor.detach().cpu().numpy()
            mm = np.memmap(str(path), dtype=arr.dtype, mode="w+", shape=arr.shape)
            mm[:] = arr[:]
            mm.flush()
            # Keep a cpu tensor view for fast access
            cpu_tensor = torch.from_numpy(np.array(mm)).clone()
            self._ram_used += size
            return TensorEntry(
                tier=Tier.RAM, tensor=cpu_tensor, path=path,
                shape=tuple(tensor.shape), dtype=tensor.dtype, size_bytes=size,
            )

        else:  # SSD
            path = self.base_dir / f"{key}.pt"
            torch.save(tensor.detach().cpu(), path)
            return TensorEntry(
                tier=Tier.SSD, tensor=None, path=path,
                shape=tuple(tensor.shape), dtype=tensor.dtype, size_bytes=size,
            )

    def _read_from_tier(self, entry: TensorEntry) -> torch.Tensor:
        if entry.tier == Tier.HOT:
            assert entry.tensor is not None
            return entry.tensor
        elif entry.tier == Tier.RAM:
            if entry.tensor is not None:
                return entry.tensor
            import numpy as np
            mm = np.memmap(str(entry.path), dtype="float32", mode="r", shape=entry.shape)
            return torch.from_numpy(np.array(mm))
        else:  # SSD
            assert entry.path is not None
            return torch.load(entry.path, weights_only=True)

    def _make_hot(self, key: str, tensor: torch.Tensor, entry: TensorEntry) -> None:
        """Promote tensor to hot tier (in-place, called under lock)."""
        if entry.tier == Tier.HOT:
            return
        hot_tensor = self._to_hot(tensor)
        if entry.tier == Tier.RAM:
            self._ram_used -= entry.size_bytes
        self._hot_used += entry.size_bytes
        entry.tier = Tier.HOT
        entry.tensor = hot_tensor

    def _evict_locked(self, key: str, to_tier: Tier) -> None:
        """Evict key to lower tier (called under lock)."""
        if key not in self._entries:
            return
        entry = self._entries[key]
        if entry.tier == to_tier:
            return

        tensor = self._read_from_tier(entry)

        # Clean up old tier resources
        if entry.tier == Tier.HOT:
            self._hot_used -= entry.size_bytes
            entry.tensor = None
        elif entry.tier == Tier.RAM:
            self._ram_used -= entry.size_bytes
            entry.tensor = None
            if entry.path and to_tier == Tier.SSD:
                entry.path.unlink(missing_ok=True)
                entry.path = None

        # Write to new tier
        new_entry = self._write_to_tier(key, tensor, to_tier)
        new_entry.access_count = entry.access_count
        self._entries[key] = new_entry

    @staticmethod
    def _to_hot(tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to 'hot' storage (GPU if available, else CPU)."""
        if torch.cuda.is_available():
            return tensor.cuda()
        return tensor.detach().cpu()
