"""ExpertAwareBatchScheduler — schedule inference requests with expert-aware grouping.

Wraps ExpertGrouper to form batches that maximise expert cache reuse.
Requests with overlapping expert sets are processed together, amortising
the expert loading cost across the batch.

CPU mode: all logic runs synchronously; no GPU memory tracking needed.
TODO(gpu): integrate torch.cuda.memory_allocated() for real budget tracking.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dana_engine.batch.expert_grouper import ExpertGrouper, InferenceRequest, RequestGroup

if TYPE_CHECKING:
    from dana_engine.model.transformer import TinyMoETransformer


@dataclass
class SchedulerConfig:
    """Configuration for the batch scheduler."""
    max_batch_size: int = 4
    overlap_threshold: float = 0.4
    max_queue_size: int = 64
    max_wait_ms: float = 50.0  # max time to wait for batch to fill


class ExpertAwareBatchScheduler:
    """Schedules inference requests into expert-overlap-aware batches.

    Usage:
        scheduler = ExpertAwareBatchScheduler(model)
        scheduler.submit(request)
        groups = scheduler.next_batch()
        for group in groups:
            # execute group.requests together
    """

    def __init__(
        self,
        model: "TinyMoETransformer",
        config: SchedulerConfig | None = None,
    ) -> None:
        self.config = config or SchedulerConfig()
        self._grouper = ExpertGrouper(
            model,
            overlap_threshold=self.config.overlap_threshold,
            max_group_size=self.config.max_batch_size,
        )
        self._queue: list[InferenceRequest] = []

    def submit(self, request: InferenceRequest) -> None:
        """Add a request to the scheduling queue."""
        if len(self._queue) >= self.config.max_queue_size:
            # Drop oldest on overflow (in production: back-pressure)
            self._queue.pop(0)
        self._queue.append(request)

    def next_batch(self) -> list[RequestGroup]:
        """Form the next expert-aware batch from queued requests.

        Returns:
            List of RequestGroups. Each group should be executed together
            to maximise expert cache reuse.
        """
        if not self._queue:
            return []

        # Take up to max_batch_size requests
        batch_requests = self._queue[:self.config.max_batch_size]
        self._queue = self._queue[self.config.max_batch_size:]

        return self._grouper.group(batch_requests)

    def pending_count(self) -> int:
        """Number of requests waiting in the queue."""
        return len(self._queue)

    def queue_snapshot(self) -> list[InferenceRequest]:
        """Return a copy of the current queue (for inspection/debugging)."""
        return list(self._queue)
