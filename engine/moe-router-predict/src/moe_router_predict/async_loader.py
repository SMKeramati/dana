"""AsyncExpertLoader — priority-queue-based expert loader.

Uses asyncio.PriorityQueue to load experts from lower tiers to hot tier
asynchronously while the current inference step runs. This is the
double-buffering mechanism: while expert N is being used, expert N+1
is being loaded.

Priority: lower number = higher priority (loaded sooner).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Callable

from moe_router_predict.residency import ExpertResidencyTracker

logger = logging.getLogger(__name__)


@dataclass(order=True)
class LoadRequest:
    priority: float
    expert_id: int = field(compare=False)
    target_tier: str = field(compare=False, default="hot")


class AsyncExpertLoader:
    """Async double-buffered expert loader.

    Usage:
        loader = AsyncExpertLoader(tracker, load_fn=my_load_function)
        await loader.start()
        await loader.enqueue(expert_id=3, target_tier="hot", priority=1.0)
        # loader will call load_fn(3, "hot") in background
        await loader.stop()

    The load_fn is called with (expert_id, target_tier) and should move
    the expert weights to the requested tier. It can be a sync or async fn.
    """

    def __init__(
        self,
        tracker: ExpertResidencyTracker,
        load_fn: Optional[Callable] = None,
        max_concurrent: int = 2,
        cuda_stream: Optional[Any] = None,
    ) -> None:
        self.tracker = tracker
        self.load_fn = load_fn or self._noop_load
        self.max_concurrent = max_concurrent
        self._cuda_stream = cuda_stream  # torch.cuda.Stream for non-blocking H2D
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._workers: list[asyncio.Task] = []
        self._running = False
        self.loads_completed: int = 0
        self.loads_enqueued: int = 0

    @classmethod
    def create_with_cuda_stream(
        cls,
        tracker: ExpertResidencyTracker,
        load_fn: Optional[Callable] = None,
        max_concurrent: int = 2,
    ) -> "AsyncExpertLoader":
        """Create loader with a dedicated CUDA stream for non-blocking H2D transfers.

        Each expert load runs inside ``torch.cuda.stream(stream)`` so H2D
        copies overlap with GPU compute on the default stream. Falls back to
        CPU mode (no stream) when CUDA is not available.
        """
        try:
            import torch
            if torch.cuda.is_available():
                stream = torch.cuda.Stream()
                return cls(tracker, load_fn, max_concurrent, cuda_stream=stream)
        except ImportError:
            pass
        return cls(tracker, load_fn, max_concurrent)

    async def start(self) -> None:
        """Start background worker tasks."""
        self._running = True
        for _ in range(self.max_concurrent):
            task = asyncio.create_task(self._worker())
            self._workers.append(task)
        logger.debug("AsyncExpertLoader started (%d workers)", self.max_concurrent)

    async def stop(self) -> None:
        """Stop all workers gracefully."""
        self._running = False
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.debug("AsyncExpertLoader stopped. Loads: %d", self.loads_completed)

    async def enqueue(
        self,
        expert_id: int,
        target_tier: str = "hot",
        priority: float = 1.0,
    ) -> None:
        """Enqueue an expert load request.

        Args:
            expert_id: Which expert to load
            target_tier: Destination tier ("hot" or "ram")
            priority: Lower = higher priority (loaded sooner)
        """
        if self.tracker.is_hot(expert_id) or self.tracker.is_in_flight(expert_id):
            return  # already available or being loaded

        self.tracker.mark_in_flight(expert_id)
        req = LoadRequest(priority=priority, expert_id=expert_id, target_tier=target_tier)
        await self._queue.put(req)
        self.loads_enqueued += 1

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _worker(self) -> None:
        while self._running:
            try:
                req: LoadRequest = await asyncio.wait_for(
                    self._queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                if asyncio.iscoroutinefunction(self.load_fn):
                    await self.load_fn(req.expert_id, req.target_tier)
                elif self._cuda_stream is not None:
                    # GPU: run load_fn inside a dedicated CUDA stream so the
                    # H2D transfer overlaps with compute on the default stream
                    # (non-blocking double-buffer behaviour).
                    import torch
                    cuda_stream = self._cuda_stream

                    def _load_in_stream() -> None:
                        with torch.cuda.stream(cuda_stream):
                            self.load_fn(req.expert_id, req.target_tier)

                    await asyncio.get_event_loop().run_in_executor(None, _load_in_stream)
                else:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.load_fn, req.expert_id, req.target_tier
                    )
                self.tracker.mark(req.expert_id, req.target_tier)  # type: ignore
                self.loads_completed += 1
            except Exception as exc:
                logger.warning(
                    "Failed to load expert %d → %s: %s",
                    req.expert_id, req.target_tier, exc
                )
                # Revert in_flight status
                self.tracker.mark(req.expert_id, "ssd")
            finally:
                self._queue.task_done()

    @staticmethod
    async def _noop_load(expert_id: int, target_tier: str) -> None:
        """No-op load function used when no real loader is configured."""
        await asyncio.sleep(0)  # simulate async work
