"""SSDStore — SSD tier backend using torch.save/torch.load.

In production, this would use O_DIRECT to bypass the OS page cache for
predictable latency. On this CPU-only dev environment it uses standard I/O.

See GPU_TODO.md item 2 for adding O_DIRECT on Linux with a real NVMe drive.
"""

from __future__ import annotations

from pathlib import Path

import torch


class SSDStore:
    """File-backed tensor store for the SSD tier.

    Each tensor is stored as a separate .pt file.
    """

    def __init__(self, base_dir: str = "/tmp/dana_ssd") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, key: str, tensor: torch.Tensor) -> Path:
        """Persist a tensor to disk.

        Returns:
            Path where the tensor was saved
        """
        path = self._path(key)
        torch.save(tensor.detach().cpu(), path)
        return path

    def load(self, key: str) -> torch.Tensor:
        """Load a tensor from disk.

        Raises:
            FileNotFoundError: if key not found
        """
        path = self._path(key)
        if not path.exists():
            raise FileNotFoundError(f"SSDStore: key '{key}' not found at {path}")
        return torch.load(path, weights_only=True)

    def delete(self, key: str) -> None:
        """Delete a tensor from disk."""
        self._path(key).unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def path_of(self, key: str) -> Path:
        return self._path(key)

    def list_keys(self) -> list[str]:
        return [p.stem for p in self.base_dir.glob("*.pt")]

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.base_dir.glob("*.pt"))

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("\\", "_")
        return self.base_dir / f"{safe}.pt"
