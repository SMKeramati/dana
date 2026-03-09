"""MmapPool — directory-based pool of memory-mapped numpy arrays.

Provides the RAM tier backend: tensors are stored as memory-mapped files,
giving true RAM-tier semantics (OS manages page cache, flushing, etc.).
Converts cleanly to/from torch.Tensor via torch.from_numpy().
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


class MmapPool:
    """A directory-backed pool of numpy.memmap arrays.

    Each array is stored as a separate file: <base_dir>/<key>.mmap
    Arrays are kept open for fast repeated access.
    """

    def __init__(self, base_dir: str = "/tmp/dana_mmap") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._open: dict[str, np.memmap] = {}  # key → open memmap

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def allocate(
        self,
        key: str,
        shape: tuple,
        dtype: np.dtype | str = np.float32,
        fill: float = 0.0,
    ) -> np.memmap:
        """Create a new memmap array (zero-initialised by default).

        Args:
            key: Unique identifier
            shape: Array shape
            dtype: Numpy dtype
            fill: Initial fill value (0.0 by default)

        Returns:
            numpy.memmap writable array
        """
        path = self._path(key)
        mm = np.memmap(str(path), dtype=dtype, mode="w+", shape=shape)
        if fill != 0.0:
            mm[:] = fill
        else:
            mm[:] = 0
        mm.flush()
        self._open[key] = mm
        return mm

    def write(self, key: str, tensor: torch.Tensor) -> np.memmap:
        """Write a torch.Tensor to a new memmap file.

        Args:
            key: Unique identifier
            tensor: Tensor to persist (will be moved to CPU)

        Returns:
            numpy.memmap containing the tensor data
        """
        arr = tensor.detach().cpu().numpy()
        path = self._path(key)
        mm = np.memmap(str(path), dtype=arr.dtype, mode="w+", shape=arr.shape)
        mm[:] = arr[:]
        mm.flush()
        self._open[key] = mm
        return mm

    def get(self, key: str) -> np.memmap:
        """Get an open memmap (must have been previously allocated or written).

        Raises:
            KeyError: if key not found
        """
        if key in self._open:
            return self._open[key]
        path = self._path(key)
        if not path.exists():
            raise KeyError(f"MmapPool: key '{key}' not found at {path}")
        # Re-open in readonly mode
        mm = np.memmap(str(path), mode="r")
        self._open[key] = mm
        return mm

    def to_tensor(self, key: str) -> torch.Tensor:
        """Load a memmap array as a torch.Tensor (copy to avoid aliasing issues)."""
        mm = self.get(key)
        return torch.from_numpy(np.array(mm))

    def free(self, key: str) -> None:
        """Close and delete a memmap entry."""
        if key in self._open:
            del self._open[key]
        path = self._path(key)
        path.unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        return key in self._open or self._path(key).exists()

    def keys(self) -> list[str]:
        return list(self._open.keys())

    def total_bytes(self) -> int:
        """Total bytes currently allocated across all open memmaps."""
        total = 0
        for mm in self._open.values():
            total += mm.nbytes
        return total

    def flush_all(self) -> None:
        """Flush all open memmaps to disk."""
        for mm in self._open.values():
            mm.flush()

    def close_all(self) -> None:
        """Close all open memmaps (data persists on disk)."""
        self._open.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("\\", "_")
        return self.base_dir / f"{safe}.mmap"
