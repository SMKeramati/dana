"""Custom continuous batching scheduler.

Daneshbonyan: Internal Design & Development

Dynamic batch sizing based on GPU memory pressure and request priority.
The scheduler maintains a priority queue of pending requests and
continuously forms batches that fit within the current GPU memory budget.
Running sequences can be preempted if a higher-priority request arrives
and memory is tight.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Request priority levels (lower value = higher priority)."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class RequestState(IntEnum):
    PENDING = 0
    RUNNING = 1
    PREEMPTED = 2
    COMPLETED = 3


@dataclass
class InferenceRequest:
    """A single inference request tracked by the scheduler."""

    request_id: str
    token_ids: np.ndarray
    max_new_tokens: int = 256
    priority: Priority = Priority.NORMAL
    state: RequestState = RequestState.PENDING
    generated_tokens: int = 0
    arrival_ts: float = field(default_factory=time.monotonic)
    start_ts: float | None = None
    end_ts: float | None = None

    @property
    def total_len(self) -> int:
        """Current total sequence length (prompt + generated)."""
        return len(self.token_ids) + self.generated_tokens

    @property
    def remaining_tokens(self) -> int:
        return self.max_new_tokens - self.generated_tokens

    @property
    def is_done(self) -> bool:
        return self.generated_tokens >= self.max_new_tokens


@dataclass
class Batch:
    """A batch of requests to be processed together."""

    batch_id: int
    requests: list[InferenceRequest]
    created_ts: float = field(default_factory=time.monotonic)

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_len for r in self.requests)

    @property
    def max_seq_len(self) -> int:
        if not self.requests:
            return 0
        return max(r.total_len for r in self.requests)


class ContinuousBatchScheduler:
    """Continuous batching scheduler with dynamic sizing.

    Daneshbonyan: Internal Design & Development

    Parameters
    ----------
    gpu_memory_budget_bytes : int
        Total GPU memory available for KV cache of active batches.
    bytes_per_token : int
        Estimated KV cache cost per token per sequence (both K and V,
        all layers).
    max_batch_size : int
        Hard upper limit on batch size regardless of memory.
    preemption_enabled : bool
        Whether lower-priority running requests can be preempted.
    """

    def __init__(
        self,
        gpu_memory_budget_bytes: int = 4 * 1024**3,
        bytes_per_token: int = 2048,
        max_batch_size: int = 64,
        preemption_enabled: bool = True,
    ) -> None:
        self._memory_budget = gpu_memory_budget_bytes
        self._bytes_per_token = bytes_per_token
        self._max_batch_size = max_batch_size
        self._preemption_enabled = preemption_enabled

        self._pending: list[InferenceRequest] = []
        self._running: list[InferenceRequest] = []
        self._preempted: list[InferenceRequest] = []
        self._completed: list[InferenceRequest] = []
        self._batch_counter: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_request(self, request: InferenceRequest) -> None:
        """Submit a new request to the scheduler."""
        request.state = RequestState.PENDING
        self._pending.append(request)
        logger.debug("Request %s added (priority=%s)", request.request_id, request.priority.name)

    def _estimate_memory(self, requests: list[InferenceRequest]) -> int:
        """Estimate total GPU memory needed for a set of requests."""
        return sum(
            r.total_len * self._bytes_per_token for r in requests
        )

    def _memory_used(self) -> int:
        """Memory consumed by currently running requests."""
        return self._estimate_memory(self._running)

    def _memory_available(self) -> int:
        return self._memory_budget - self._memory_used()

    def _sort_pending(self) -> None:
        """Sort pending queue: higher priority first, then FIFO."""
        self._pending.sort(key=lambda r: (r.priority, r.arrival_ts))

    def schedule(self) -> Batch | None:
        """Form the next batch from pending and preempted requests.

        Returns None if no requests can be scheduled.
        """
        # Re-queue preempted requests with boosted priority
        for req in self._preempted:
            req.state = RequestState.PENDING
            self._pending.append(req)
        self._preempted.clear()

        self._sort_pending()

        if not self._pending and not self._running:
            return None

        # Continuous batching: add new requests to the running set
        available = self._memory_available()
        added: list[InferenceRequest] = []

        for req in list(self._pending):
            req_mem = req.total_len * self._bytes_per_token
            if len(self._running) + len(added) >= self._max_batch_size:
                break
            if req_mem <= available:
                added.append(req)
                available -= req_mem

        # If a high-priority request cannot fit, try preempting
        if self._preemption_enabled:
            for req in list(self._pending):
                if req in added:
                    continue
                if req.priority >= Priority.NORMAL:
                    continue  # only preempt for HIGH / CRITICAL
                req_mem = req.total_len * self._bytes_per_token
                freed = self._try_preempt(req_mem - available)
                if freed > 0:
                    available += freed
                    if req_mem <= available:
                        added.append(req)
                        available -= req_mem

        for req in added:
            self._pending.remove(req)
            req.state = RequestState.RUNNING
            if req.start_ts is None:
                req.start_ts = time.monotonic()
            self._running.append(req)

        if not self._running:
            return None

        self._batch_counter += 1
        batch = Batch(batch_id=self._batch_counter, requests=list(self._running))
        return batch

    def step_completed(self, request_ids: list[str] | None = None) -> None:
        """Mark that one decoding step has been completed.

        If *request_ids* is provided, only those requests are advanced;
        otherwise all running requests are advanced by one token.
        """
        for req in list(self._running):
            if request_ids is not None and req.request_id not in request_ids:
                continue
            req.generated_tokens += 1
            if req.is_done:
                req.state = RequestState.COMPLETED
                req.end_ts = time.monotonic()
                self._running.remove(req)
                self._completed.append(req)

    def _try_preempt(self, needed_bytes: int) -> int:
        """Preempt lowest-priority running requests to free memory.

        Returns total bytes freed.
        """
        if not self._running:
            return 0

        # Sort running by priority descending (lowest priority first for preemption)
        candidates = sorted(self._running, key=lambda r: (-r.priority, r.arrival_ts))
        freed = 0
        for req in candidates:
            if freed >= needed_bytes:
                break
            mem = req.total_len * self._bytes_per_token
            req.state = RequestState.PREEMPTED
            self._running.remove(req)
            self._preempted.append(req)
            freed += mem
            logger.info("Preempted request %s (priority=%s)", req.request_id, req.priority.name)
        return freed

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    def compute_dynamic_batch_size(self, avg_seq_len: int) -> int:
        """Compute the max batch size that fits in memory for a given avg seq len."""
        per_request = avg_seq_len * self._bytes_per_token
        if per_request <= 0:
            return self._max_batch_size
        return min(self._max_batch_size, max(1, self._memory_budget // per_request))

    def get_throughput_stats(self) -> dict[str, float]:
        """Return throughput statistics for completed requests."""
        if not self._completed:
            return {"avg_latency_s": 0.0, "avg_tokens_per_s": 0.0, "total_completed": 0}

        latencies = []
        tps_list = []
        for req in self._completed:
            if req.start_ts and req.end_ts:
                lat = req.end_ts - req.start_ts
                latencies.append(lat)
                if lat > 0:
                    tps_list.append(req.generated_tokens / lat)

        return {
            "avg_latency_s": float(np.mean(latencies)) if latencies else 0.0,
            "avg_tokens_per_s": float(np.mean(tps_list)) if tps_list else 0.0,
            "total_completed": len(self._completed),
        }
