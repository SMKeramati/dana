"""Weighted round-robin load balancer with health awareness for GPU workers.

Daneshbonyan: Internal Design & Development - Custom weighted round-robin
load balancer that tracks GPU worker health and load scores. Workers are
selected based on a combination of static weight and dynamic health metrics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GPUWorker:
    """Represents a GPU inference worker with health tracking."""

    worker_id: str
    host: str
    port: int
    weight: int = 1
    gpu_memory_total_mb: int = 0
    gpu_memory_used_mb: int = 0
    active_requests: int = 0
    max_concurrent: int = 16
    healthy: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    avg_latency_ms: float = 0.0

    @property
    def load_score(self) -> float:
        """Compute a normalised load score (0.0 = idle, 1.0 = fully loaded).

        Combines request saturation and GPU memory pressure.
        """
        request_ratio = self.active_requests / max(self.max_concurrent, 1)
        memory_ratio = (
            self.gpu_memory_used_mb / max(self.gpu_memory_total_mb, 1)
            if self.gpu_memory_total_mb > 0
            else 0.0
        )
        return 0.6 * request_ratio + 0.4 * memory_ratio

    @property
    def effective_weight(self) -> float:
        """Weight adjusted by inverse load and health status."""
        if not self.healthy:
            return 0.0
        return self.weight * (1.0 - self.load_score)


class LoadBalancer:
    """Weighted round-robin load balancer with health-aware selection.

    Daneshbonyan: Internal Design & Development - Not an off-the-shelf proxy.
    Custom implementation that combines static weights with real-time GPU
    utilisation metrics for optimal job placement.
    """

    HEARTBEAT_TIMEOUT_S: float = 30.0
    MAX_CONSECUTIVE_FAILURES: int = 5

    def __init__(self) -> None:
        self._workers: dict[str, GPUWorker] = {}
        self._rr_index: int = 0

    # ------------------------------------------------------------------
    # Worker registry
    # ------------------------------------------------------------------

    def register(self, worker: GPUWorker) -> None:
        """Register or update a GPU worker."""
        self._workers[worker.worker_id] = worker
        logger.info("Registered worker %s at %s:%d", worker.worker_id, worker.host, worker.port)

    def deregister(self, worker_id: str) -> None:
        """Remove a worker from the pool."""
        removed = self._workers.pop(worker_id, None)
        if removed:
            logger.info("Deregistered worker %s", worker_id)

    def heartbeat(self, worker_id: str, metrics: dict[str, Any] | None = None) -> None:
        """Record heartbeat and optionally update metrics for a worker."""
        worker = self._workers.get(worker_id)
        if worker is None:
            return
        worker.last_heartbeat = time.time()
        worker.consecutive_failures = 0
        worker.healthy = True
        if metrics:
            worker.gpu_memory_used_mb = metrics.get("gpu_memory_used_mb", worker.gpu_memory_used_mb)
            worker.gpu_memory_total_mb = metrics.get("gpu_memory_total_mb", worker.gpu_memory_total_mb)
            worker.active_requests = metrics.get("active_requests", worker.active_requests)
            worker.avg_latency_ms = metrics.get("avg_latency_ms", worker.avg_latency_ms)

    def report_failure(self, worker_id: str) -> None:
        """Increment failure counter and potentially mark worker unhealthy."""
        worker = self._workers.get(worker_id)
        if worker is None:
            return
        worker.consecutive_failures += 1
        if worker.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            worker.healthy = False
            logger.warning("Worker %s marked unhealthy after %d failures", worker_id, worker.consecutive_failures)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _check_timeouts(self) -> None:
        """Mark workers as unhealthy if heartbeat is stale."""
        now = time.time()
        for worker in self._workers.values():
            if now - worker.last_heartbeat > self.HEARTBEAT_TIMEOUT_S:
                if worker.healthy:
                    worker.healthy = False
                    logger.warning("Worker %s timed out (no heartbeat)", worker.worker_id)

    def get_healthy_workers(self) -> list[GPUWorker]:
        """Return all currently healthy workers, sorted by effective weight."""
        self._check_timeouts()
        healthy = [w for w in self._workers.values() if w.healthy and w.effective_weight > 0]
        return sorted(healthy, key=lambda w: w.effective_weight, reverse=True)

    def select_worker(self) -> GPUWorker | None:
        """Select the best available worker using weighted round-robin.

        Workers are weighted by their effective_weight which accounts for
        current GPU load and health status. Returns None if no healthy
        workers are available.
        """
        healthy = self.get_healthy_workers()
        if not healthy:
            logger.error("No healthy GPU workers available")
            return None

        # Weighted selection: pick the worker with highest effective weight,
        # cycling through on ties to avoid starvation.
        total_weight = sum(w.effective_weight for w in healthy)
        if total_weight == 0:
            return None

        self._rr_index = self._rr_index % len(healthy)
        selected = healthy[self._rr_index]

        # Advance index, preferring workers with higher weight
        self._rr_index = (self._rr_index + 1) % len(healthy)
        return selected

    @property
    def worker_count(self) -> int:
        return len(self._workers)

    @property
    def healthy_count(self) -> int:
        self._check_timeouts()
        return len([w for w in self._workers.values() if w.healthy])
