"""BackgroundPromoter — async task that re-optimises tier placement.

Runs as an asyncio background task. Every `interval_seconds` it:
1. Calls PlacementOptimizer.optimize_store() to get recommended assignments
2. Computes the delta (moves needed)
3. Executes the moves by calling store.evict() or reloading

This simulates the background DMA movement that the engine performs to
pre-stage experts for upcoming inference steps.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from tiered_tensor_store.placement_optimizer import PlacementOptimizer

logger = logging.getLogger(__name__)


class BackgroundPromoter:
    """Asyncio background task that rebalances tier placement.

    Usage:
        promoter = BackgroundPromoter(store, interval_seconds=5.0)
        await promoter.start()
        # ... inference runs ...
        await promoter.stop()
    """

    def __init__(
        self,
        store: object,
        interval_seconds: float = 5.0,
        hot_budget_bytes: int = 4 * 1024**3,
        ram_budget_bytes: int = 32 * 1024**3,
    ) -> None:
        self.store = store
        self.interval = interval_seconds
        self.optimizer = PlacementOptimizer(hot_budget_bytes, ram_budget_bytes)
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.moves_executed: int = 0

    async def start(self) -> None:
        """Start the background rebalancing loop."""
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.debug("BackgroundPromoter started (interval=%.1fs)", self.interval)

    async def stop(self) -> None:
        """Stop the background loop gracefully."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.debug("BackgroundPromoter stopped. Total moves: %d", self.moves_executed)

    async def run_once(self) -> list[tuple[str, str, str]]:
        """Run a single rebalance pass. Returns list of (key, from, to) moves."""
        from tiered_tensor_store.tier_manager import TieredTensorStore
        store: TieredTensorStore = self.store

        recommended = self.optimizer.optimize_store(store)
        current = {k: store.tier_of(k) for k in store.all_keys()}
        moves = self.optimizer.delta(current, recommended)

        for key, from_tier, to_tier in moves:
            try:
                # Promotion (SSD→RAM or RAM→HOT): load then re-store at new tier
                if _tier_rank(to_tier) > _tier_rank(from_tier):
                    tensor = store.load(key)
                    store.store(key, tensor, tier=to_tier)
                else:
                    # Demotion
                    store.evict(key, to_tier=to_tier)
                self.moves_executed += 1
                logger.debug("Moved '%s': %s → %s", key, from_tier, to_tier)
            except Exception as exc:
                logger.warning("Failed to move '%s': %s", key, exc)

        return moves

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while self._running:
            try:
                await self.run_once()
            except Exception as exc:
                logger.warning("BackgroundPromoter error: %s", exc)
            await asyncio.sleep(self.interval)


def _tier_rank(tier: str) -> int:
    return {"ssd": 0, "ram": 1, "hot": 2}.get(tier, 0)
