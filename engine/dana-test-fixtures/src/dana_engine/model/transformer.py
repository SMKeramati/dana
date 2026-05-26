"""Tiny MoE Transformer: the synthetic test model for all engine packages.

Architecture:
    token_embedding (vocab_size, hidden_dim)
    N x TransformerBlock:
        LayerNorm → CausalSelfAttention → residual
        LayerNorm → MoELayer → residual
    LayerNorm
    lm_head (hidden_dim, vocab_size)

Returns logits and all router logits (one per layer), so downstream libraries
(moe-router-predict, moe-self-draft) can access routing patterns.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from dana_engine.model.attention import CausalSelfAttention
from dana_engine.model.config import TinyMoEConfig
from dana_engine.model.moe_layer import MoELayer


class TransformerBlock(nn.Module):
    def __init__(self, config: TinyMoEConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.moe = MoELayer(config)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: (B, T, H) after attention + MoE
            attn_weights: (B, num_heads, T, T)
            router_logits: (B, T, num_experts)
        """
        attn_out, attn_weights = self.attn(self.ln1(x))
        x = x + attn_out

        moe_out, router_logits = self.moe(self.ln2(x))
        x = x + moe_out

        return x, attn_weights, router_logits


@dataclass
class ForwardOutput:
    logits: torch.Tensor                  # (batch, seq, vocab_size)
    all_router_logits: list[torch.Tensor]  # one (batch, seq, num_experts) per layer
    all_hidden_states: list[torch.Tensor]  # one (batch, seq, hidden_dim) per layer


class TinyMoETransformer(nn.Module):
    """The synthetic tiny MoE model used in all Dana engine tests.

    This is the foundation model. Every other engine package (tiered-tensor-store,
    expert-cache, moe-router-predict, moe-quant, spec-decode-tree, moe-self-draft)
    uses this model as the test harness.

    To make expert weights accessible for caching/quantization tests:
        model.blocks[layer_idx].moe.experts[expert_id]
    """

    def __init__(self, config: TinyMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_final = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying (standard practice)
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden_states: bool = False,
    ) -> ForwardOutput:
        """
        Args:
            input_ids: (batch, seq) integer token IDs
            return_hidden_states: if True, include hidden states for all layers
        Returns:
            ForwardOutput with logits, all_router_logits, all_hidden_states
        """
        x = self.embed(input_ids)                   # (B, T, H)

        all_router_logits: list[torch.Tensor] = []
        all_hidden_states: list[torch.Tensor] = []

        for block in self.blocks:
            x, _, router_logits = block(x)
            all_router_logits.append(router_logits)
            if return_hidden_states:
                all_hidden_states.append(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)                    # (B, T, vocab_size)

        return ForwardOutput(
            logits=logits,
            all_router_logits=all_router_logits,
            all_hidden_states=all_hidden_states,
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def expert_weights(self, layer: int, expert_id: int) -> list[torch.Tensor]:
        """Return weight tensors for a specific expert (for caching/quantization tests)."""
        expert = self.blocks[layer].moe.experts[expert_id]
        return [expert.w1.weight, expert.w2.weight]
