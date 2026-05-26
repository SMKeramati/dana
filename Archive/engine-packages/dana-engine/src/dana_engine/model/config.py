"""Configuration dataclass for the tiny MoE test model."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class TinyMoEConfig:
    """Configuration for the synthetic tiny MoE model used in all engine tests.

    Defaults produce a ~5MB model that runs a 10-token generation in <1s on CPU,
    making it suitable for fast unit tests across all engine packages.
    """

    # Architecture
    num_layers: int = 4
    num_experts: int = 8
    num_active: int = 2          # top-k experts selected per token
    hidden_dim: int = 256
    ffn_dim: int = 512
    num_heads: int = 8
    vocab_size: int = 1000
    max_seq_len: int = 128

    # Derived (computed from hidden_dim / num_heads)
    head_dim: int = field(init=False)

    # Runtime
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        assert self.hidden_dim % self.num_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        )
        assert self.num_active <= self.num_experts, (
            f"num_active ({self.num_active}) must be <= num_experts ({self.num_experts})"
        )
        self.head_dim = self.hidden_dim // self.num_heads

    @classmethod
    def tiny(cls) -> "TinyMoEConfig":
        """4-layer, 8-expert, 256-dim model. ~5MB. Tests complete in <1s on CPU."""
        return cls()

    @classmethod
    def micro(cls) -> "TinyMoEConfig":
        """Smallest possible config for near-instant tests."""
        return cls(num_layers=2, num_experts=4, num_active=1, hidden_dim=64,
                   ffn_dim=128, num_heads=4, vocab_size=256, max_seq_len=32)
