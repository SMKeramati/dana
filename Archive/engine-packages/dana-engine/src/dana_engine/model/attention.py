"""Causal multi-head self-attention.

Standard pre-norm causal attention. No KV-cache (added in Phase 7 dana-engine
integration). Returns attention weights alongside output so they can be inspected
in benchmarks and tests.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dana_engine.model.config import TinyMoEConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (no KV-cache)."""

    def __init__(self, config: TinyMoEConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq, hidden_dim)
            attn_mask: optional additive mask (batch, num_heads, seq, seq)
        Returns:
            output: (batch, seq, hidden_dim)
            attn_weights: (batch, num_heads, seq, seq)
        """
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        # Causal mask: upper-triangular = -inf
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        scores = scores + causal

        if attn_mask is not None:
            scores = scores + attn_mask

        attn_weights = F.softmax(scores, dim=-1)                   # (B, H, T, T)
        out = torch.matmul(attn_weights, v)                        # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)
        out = self.out_proj(out)
        return out, attn_weights
