"""ExpertGrouper — group inference requests by predicted expert overlap.

Key insight: requests that activate the same experts can be batched together
at zero extra expert-loading cost. Expert loads are amortized across the batch.

Algorithm:
  1. Run router-only forward on each request's prompt
  2. Compute Jaccard similarity of predicted expert sets
  3. Greedy grouping: pair requests with similarity > threshold
  4. Output groups sorted by overlap (maximize cache reuse between groups)

CPU mode: router-only forward uses RouterPredictor (cheap linear layer).
TODO(gpu): run router-only forward in a separate CUDA stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from dana_engine.model.transformer import TinyMoETransformer


@dataclass
class InferenceRequest:
    """A pending inference request."""
    request_id: str
    input_ids: torch.Tensor   # (1, T)
    max_new_tokens: int = 32
    temperature: float = 1.0


@dataclass
class RequestGroup:
    """A batch of requests grouped by expert overlap."""
    requests: list[InferenceRequest]
    predicted_experts: set[int]
    overlap_score: float  # average pairwise Jaccard similarity


def jaccard(a: set[int], b: set[int]) -> float:
    """Jaccard similarity between two expert sets."""
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


class ExpertGrouper:
    """Groups inference requests by predicted expert activation overlap.

    Usage:
        grouper = ExpertGrouper(model, overlap_threshold=0.5)
        groups = grouper.group(requests)
        for group in groups:
            # batch group.requests together — they share experts
    """

    def __init__(
        self,
        model: "TinyMoETransformer",
        overlap_threshold: float = 0.4,
        max_group_size: int = 4,
    ) -> None:
        self.model = model
        self.overlap_threshold = overlap_threshold
        self.max_group_size = max_group_size

    def group(self, requests: list[InferenceRequest]) -> list[RequestGroup]:
        """Group requests by predicted expert overlap.

        Args:
            requests: List of pending inference requests

        Returns:
            List of RequestGroups, sorted by descending overlap score.
        """
        if not requests:
            return []

        if len(requests) == 1:
            experts = self._predict_experts(requests[0])
            return [RequestGroup(
                requests=requests,
                predicted_experts=experts,
                overlap_score=1.0,
            )]

        # Predict experts for each request
        expert_sets = [self._predict_experts(r) for r in requests]

        # Greedy grouping
        ungrouped = list(range(len(requests)))
        groups: list[RequestGroup] = []

        while ungrouped:
            seed_idx = ungrouped.pop(0)
            group_indices = [seed_idx]
            group_experts = set(expert_sets[seed_idx])

            remaining = list(ungrouped)
            for idx in remaining:
                if len(group_indices) >= self.max_group_size:
                    break
                sim = jaccard(group_experts, expert_sets[idx])
                if sim >= self.overlap_threshold:
                    group_indices.append(idx)
                    ungrouped.remove(idx)
                    group_experts |= expert_sets[idx]

            # Compute average pairwise similarity for the group
            group_reqs = [requests[i] for i in group_indices]
            group_expert_lists = [expert_sets[i] for i in group_indices]
            overlap = self._avg_pairwise_jaccard(group_expert_lists)

            groups.append(RequestGroup(
                requests=group_reqs,
                predicted_experts=group_experts,
                overlap_score=overlap,
            ))

        # Sort by overlap score descending (best batches first)
        groups.sort(key=lambda g: g.overlap_score, reverse=True)
        return groups

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _predict_experts(self, request: InferenceRequest) -> set[int]:
        """Run router-only forward to predict which experts this request needs."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(request.input_ids)
            # Collect top-active experts from all layers at last position
            experts: set[int] = set()
            for router_logits in out.all_router_logits:
                # router_logits: (1, T, num_experts)
                last_pos = router_logits[0, -1, :]  # (num_experts,)
                k = self.model.config.num_active
                top_ids = torch.topk(last_pos, k).indices.tolist()
                experts.update(top_ids)
        return experts

    def _avg_pairwise_jaccard(self, expert_sets: list[set[int]]) -> float:
        """Average pairwise Jaccard similarity for a group."""
        n = len(expert_sets)
        if n <= 1:
            return 1.0
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += jaccard(expert_sets[i], expert_sets[j])
                count += 1
        return total / count if count > 0 else 0.0
