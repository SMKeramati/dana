"""MoE layer: Expert FFN + top-k router.

This is the core MoE building block. Token-choice routing: each token independently
selects its top-k experts. Router logits are returned so they can be used by
moe-router-predict and moe-self-draft.

Architecture:
    input (batch, seq, hidden)
        → MoERouter → top-k expert indices + gating weights
        → dispatch tokens to selected experts
        → each ExpertFFN: Linear(hidden→ffn) → ReLU → Linear(ffn→hidden)
        → weighted sum of expert outputs
    output (batch, seq, hidden), router_logits (batch, seq, num_experts)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dana_engine.model.config import TinyMoEConfig


class ExpertFFN(nn.Module):
    """Single expert: a two-layer feed-forward network."""

    def __init__(self, hidden_dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)))


class RouterOutput(NamedTuple):
    indices: torch.Tensor    # (batch, seq, num_active) — chosen expert indices
    weights: torch.Tensor    # (batch, seq, num_active) — softmax-normalised gate weights
    logits: torch.Tensor     # (batch, seq, num_experts) — raw router logits (for prediction)


class MoERouter(nn.Module):
    """Token-choice router: each token picks top-k experts.

    Returns raw logits alongside indices and weights so they can be captured
    externally (by moe-router-predict and moe-self-draft).
    """

    def __init__(self, hidden_dim: int, num_experts: int, num_active: int) -> None:
        super().__init__()
        self.num_active = num_active
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """
        Args:
            x: (batch, seq, hidden_dim)
        Returns:
            RouterOutput with indices, weights, logits
        """
        logits = self.gate(x)                                    # (B, T, E)
        top_k_logits, indices = torch.topk(logits, self.num_active, dim=-1)  # (B, T, k)
        weights = F.softmax(top_k_logits, dim=-1)                # (B, T, k)
        return RouterOutput(indices=indices, weights=weights, logits=logits)


class MoELayer(nn.Module):
    """Full MoE layer: route → dispatch → expert FFNs → aggregate.

    Uses a simple loop dispatch (correct but not optimised for throughput).
    The production implementation in dana-engine uses batched expert execution,
    but this reference implementation prioritises clarity.
    """

    def __init__(self, config: TinyMoEConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.num_active = config.num_active
        self.router = MoERouter(config.hidden_dim, config.num_experts, config.num_active)
        self.experts = nn.ModuleList([
            ExpertFFN(config.hidden_dim, config.ffn_dim)
            for _ in range(config.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq, hidden_dim)
        Returns:
            output: (batch, seq, hidden_dim)
            router_logits: (batch, seq, num_experts) — raw gate logits
        """
        B, T, H = x.shape
        router_out = self.router(x)

        # Dispatch: for each position, compute weighted sum of top-k expert outputs
        output = torch.zeros_like(x)

        for k_idx in range(self.num_active):
            expert_indices = router_out.indices[:, :, k_idx]   # (B, T)
            gate_weights = router_out.weights[:, :, k_idx]      # (B, T)

            # Group tokens by which expert they're going to
            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id)             # (B, T) bool
                if not mask.any():
                    continue
                expert_input = x[mask]                           # (N_tokens, H)
                expert_output = self.experts[expert_id](expert_input)  # (N_tokens, H)
                output[mask] += gate_weights[mask].unsqueeze(-1) * expert_output

        return output, router_out.logits
