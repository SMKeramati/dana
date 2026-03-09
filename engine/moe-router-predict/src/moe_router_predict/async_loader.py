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
from typing import Optional, Callable

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
    ) -> None:
        self.tracker = tracker
        self.load_fn = load_fn or self._noop_load
        self.max_concurrent = max_concurrent
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._workers: list[asyncio.Task] = []
        self._running = False
        self.loads_completed: int = 0
        self.loads_enqueued: int = 0

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
