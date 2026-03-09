"""GPU utilization, temperature, and memory tracking.

Provides a monitoring interface for GPU health metrics.  When running
on a machine without real GPUs (e.g., CI), the monitor returns simulated
values so the rest of the system can still be tested.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """Point-in-time metrics for a single GPU."""

    gpu_id: int
    utilization_pct: float  # 0..100
    memory_used_bytes: int
    memory_total_bytes: int
    temperature_c: float
    power_draw_w: float
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def memory_utilization_pct(self) -> float:
        if self.memory_total_bytes == 0:
            return 0.0
        return 100.0 * self.memory_used_bytes / self.memory_total_bytes

    @property
    def is_healthy(self) -> bool:
        return self.temperature_c < 90.0 and self.utilization_pct < 99.0


@dataclass
class GPUHealthSummary:
    """Aggregate health summary across all GPUs."""

    num_gpus: int
    all_healthy: bool
    avg_utilization_pct: float
    avg_temperature_c: float
    total_memory_used_bytes: int
    total_memory_bytes: int
    per_gpu: list[GPUMetrics]


class GPUMonitor:
    """Monitors GPU health via nvidia-smi or simulation.

    On systems without NVIDIA GPUs, the monitor generates simulated
    metrics based on a configurable load profile.
    """

    def __init__(
        self,
        num_gpus: int = 1,
        memory_per_gpu_bytes: int = 24 * 1024**3,
        simulate: bool = True,
    ) -> None:
        self._num_gpus = num_gpus
        self._memory_per_gpu = memory_per_gpu_bytes
        self._simulate = simulate
        self._rng = np.random.default_rng(42)
        self._history: list[GPUHealthSummary] = []
        self._max_history = 1000

        # Simulated base load (slowly varying)
        self._base_utilization = np.full(num_gpus, 40.0)
        self._base_temperature = np.full(num_gpus, 55.0)
        self._base_memory_frac = np.full(num_gpus, 0.5)

    def _simulate_metrics(self) -> list[GPUMetrics]:
        """Generate simulated GPU metrics."""
        metrics: list[GPUMetrics] = []
        for gid in range(self._num_gpus):
            # Add noise to base values
            util = float(np.clip(
                self._base_utilization[gid] + self._rng.normal(0, 5), 0, 100
            ))
            temp = float(np.clip(
                self._base_temperature[gid] + self._rng.normal(0, 2), 30, 100
            ))
            mem_frac = float(np.clip(
                self._base_memory_frac[gid] + self._rng.normal(0, 0.05), 0, 1
            ))
            mem_used = int(mem_frac * self._memory_per_gpu)
            power = float(np.clip(50 + util * 2.5 + self._rng.normal(0, 10), 30, 350))

            metrics.append(GPUMetrics(
                gpu_id=gid,
                utilization_pct=util,
                memory_used_bytes=mem_used,
                memory_total_bytes=self._memory_per_gpu,
                temperature_c=temp,
                power_draw_w=power,
            ))

            # Slowly drift the base values
            self._base_utilization[gid] += self._rng.normal(0, 1)
            self._base_utilization[gid] = float(np.clip(self._base_utilization[gid], 10, 95))
            self._base_temperature[gid] += self._rng.normal(0, 0.5)
            self._base_temperature[gid] = float(np.clip(self._base_temperature[gid], 40, 85))
            self._base_memory_frac[gid] += self._rng.normal(0, 0.01)
            self._base_memory_frac[gid] = float(np.clip(self._base_memory_frac[gid], 0.2, 0.95))

        return metrics

    def _read_nvidia_smi(self) -> list[GPUMetrics]:
        """Read real GPU metrics via nvidia-smi.

        Falls back to simulation if nvidia-smi is unavailable.
        """
        try:
            import subprocess
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError("nvidia-smi failed")

            metrics: list[GPUMetrics] = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    continue
                metrics.append(GPUMetrics(
                    gpu_id=int(parts[0]),
                    utilization_pct=float(parts[1]),
                    memory_used_bytes=int(float(parts[2]) * 1024 * 1024),
                    memory_total_bytes=int(float(parts[3]) * 1024 * 1024),
                    temperature_c=float(parts[4]),
                    power_draw_w=float(parts[5]),
                ))
            return metrics

        except Exception:
            logger.debug("nvidia-smi unavailable, falling back to simulation")
            return self._simulate_metrics()

    def collect(self) -> GPUHealthSummary:
        """Collect current GPU metrics and return a health summary."""
        if self._simulate:
            per_gpu = self._simulate_metrics()
        else:
            per_gpu = self._read_nvidia_smi()

        all_healthy = all(m.is_healthy for m in per_gpu)
        avg_util = float(np.mean([m.utilization_pct for m in per_gpu])) if per_gpu else 0.0
        avg_temp = float(np.mean([m.temperature_c for m in per_gpu])) if per_gpu else 0.0
        total_used = sum(m.memory_used_bytes for m in per_gpu)
        total_mem = sum(m.memory_total_bytes for m in per_gpu)

        summary = GPUHealthSummary(
            num_gpus=len(per_gpu),
            all_healthy=all_healthy,
            avg_utilization_pct=avg_util,
            avg_temperature_c=avg_temp,
            total_memory_used_bytes=total_used,
            total_memory_bytes=total_mem,
            per_gpu=per_gpu,
        )

        self._history.append(summary)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return summary

    def get_history(self, last_n: int = 100) -> list[GPUHealthSummary]:
        """Return the most recent *last_n* health summaries."""
        return self._history[-last_n:]
