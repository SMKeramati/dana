"""RouterPredictor — predict expert activations N steps ahead.

Uses the model's router modules (cheap linear layers) to predict which experts
the next N tokens will activate, without running the full forward pass.

This is the core innovation for PCIe prefetching: we run ~1% of the model's
compute (just the router) to predict which experts to preload before they're
needed.

The prediction is a proxy: we use the current last hidden state as an
approximation for future hidden states. Good enough for prefetching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from dana_engine.model.transformer import TinyMoETransformer


@dataclass
class ExpertPrediction:
    """Predicted expert activations for a future step."""
    layer: int                   # which transformer layer
    step: int                    # how many steps ahead (1 = next step)
    expert_ids: list[int]        # predicted active expert indices
    scores: list[float]          # router scores for each predicted expert


@dataclass
class StepPrediction:
    """All layer predictions for a single future step."""
    step: int
    per_layer: list[ExpertPrediction]

    def all_expert_ids(self) -> set[int]:
        """All predicted expert IDs across all layers."""
        ids: set[int] = set()
        for pred in self.per_layer:
            ids.update(pred.expert_ids)
        return ids


class RouterPredictor:
    """Predicts future expert activations using only router forward passes.

    Usage:
        predictor = RouterPredictor(model)
        # After each forward pass, call:
        predictions = predictor.predict(last_hidden_state, num_steps=3)
        for step_pred in predictions:
            for layer_pred in step_pred.per_layer:
                # layer_pred.expert_ids → prefetch these experts
    """

    def __init__(self, model: "TinyMoETransformer") -> None:
        self.model = model
        self.num_layers = model.config.num_layers
        self.num_active = model.config.num_active

    def predict(
        self,
        last_hidden_state: torch.Tensor,
        num_steps: int = 3,
    ) -> list[StepPrediction]:
        """Predict expert activations for the next `num_steps` forward passes.

        Args:
            last_hidden_state: (1, 1, hidden_dim) — last token's hidden state
                from a recent forward pass.
            num_steps: How many steps ahead to predict.

        Returns:
            List of StepPrediction, one per future step.
        """
        assert last_hidden_state.dim() == 3, \
            f"Expected (1, 1, H), got {last_hidden_state.shape}"

        predictions: list[StepPrediction] = []

        # Use the same hidden state as a proxy for all future steps
        # (future hidden states are unknown; current state is a reasonable proxy)
        x = last_hidden_state   # (1, 1, H)

        with torch.no_grad():
            for step in range(1, num_steps + 1):
                per_layer: list[ExpertPrediction] = []

                for layer_idx, block in enumerate(self.model.blocks):
                    router = block.moe.router
                    logits = router.gate(x)                          # (1, 1, num_experts)
                    top_k_logits, indices = torch.topk(
                        logits[0, 0], self.num_active
                    )                                                # (num_active,)
                    weights = F.softmax(top_k_logits, dim=-1)

                    per_layer.append(ExpertPrediction(
                        layer=layer_idx,
                        step=step,
                        expert_ids=indices.tolist(),
                        scores=weights.tolist(),
                    ))

                predictions.append(StepPrediction(step=step, per_layer=per_layer))

        return predictions

    def predict_flat(
        self,
        last_hidden_state: torch.Tensor,
        num_steps: int = 3,
    ) -> list[int]:
        """Flat list of all predicted expert IDs (deduplicated, all layers/steps)."""
        all_ids: set[int] = set()
        for step_pred in self.predict(last_hidden_state, num_steps):
            all_ids.update(step_pred.all_expert_ids())
        return sorted(all_ids)
