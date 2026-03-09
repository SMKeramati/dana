"""PrefetchScheduler — wires RouterPredictor → AsyncExpertLoader.

Called at each inference step. Converts router predictions into
prioritized AsyncExpertLoader.enqueue() calls.

Priority mapping:
  step 1 (next token) → priority 0.0 (highest, load immediately)
  step 2             → priority 1.0
  step N             → priority N-1.0 (lower priority, load when idle)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, TYPE_CHECKING

from moe_router_predict.async_loader import AsyncExpertLoader
from moe_router_predict.residency import ExpertResidencyTracker

if TYPE_CHECKING:
    import torch
    from moe_router_predict.predictor import RouterPredictor

logger = logging.getLogger(__name__)


class PrefetchScheduler:
    """Coordinates prediction → prefetch pipeline.

    Usage:
        scheduler = PrefetchScheduler(predictor, loader)
        # At each inference step:
        scheduler.on_step(last_hidden_state)   # sync, fast
        # Loader runs in background
    """

    def __init__(
        self,
        predictor: "RouterPredictor",
        loader: AsyncExpertLoader,
        num_steps: int = 3,
    ) -> None:
        self.predictor = predictor
        self.loader = loader
        self.num_steps = num_steps
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.step_count: int = 0
        self.enqueued_total: int = 0

    def on_step(self, last_hidden_state: "torch.Tensor") -> list[int]:
        """Called at each inference step with the last hidden state.

        Predicts future expert needs and enqueues loads asynchronously.

        Args:
            last_hidden_state: (1, 1, hidden_dim) last token hidden state

        Returns:
            List of expert IDs that were enqueued for prefetching
        """
        self.step_count += 1
        predictions = self.predictor.predict(last_hidden_state, self.num_steps)

        enqueued: list[int] = []
        for step_pred in predictions:
            priority = float(step_pred.step - 1)  # step 1 → 0.0, step 2 → 1.0
            for layer_pred in step_pred.per_layer:
                for expert_id in layer_pred.expert_ids:
                    if self.loader.tracker.needs_load(expert_id):
                        asyncio.ensure_future(
                            self.loader.enqueue(expert_id, "hot", priority)
                        )
                        enqueued.append(expert_id)

        self.enqueued_total += len(enqueued)
        return enqueued

    def stats(self) -> dict:
        return {
            "steps_processed": self.step_count,
            "total_enqueued": self.enqueued_total,
            "loads_completed": self.loader.loads_completed,
        }
