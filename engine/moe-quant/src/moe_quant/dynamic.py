"""DynamicRequantizer — re-quantize on tier promotion/demotion.

When an expert moves between tiers, its quantization level changes:
  cold (SSD, Q2) → warm (RAM, Q4): dequantize Q2, requantize to Q4
  warm (RAM, Q4) → hot (VRAM, Q8): dequantize Q4, requantize to Q8 or return FP16

This avoids storing multiple copies of the same expert at different precisions.
"""

from __future__ import annotations

import torch

from moe_quant.quantize import quantize, QuantizedTensor
from moe_quant.dequantize import dequantize
from moe_quant.tier_assigner import TIER_DEFAULT_BITS


class DynamicRequantizer:
    """Re-quantize expert weights when they change tiers.

    Usage:
        requantizer = DynamicRequantizer()
        # Expert moving from SSD (Q2) to RAM (Q4):
        new_qt = requantizer.on_tier_change(expert_id=3, old_tier="ssd",
                                             new_tier="ram", tensor=weight)
    """

    def __init__(
        self,
        tier_bits: dict[str, int] = None,
        group_size: int = 128,
    ) -> None:
        self.tier_bits = tier_bits or dict(TIER_DEFAULT_BITS)
        self.group_size = group_size

    def on_tier_change(
        self,
        expert_id: int,
        old_tier: str,
        new_tier: str,
        tensor: torch.Tensor,
    ) -> QuantizedTensor:
        """Re-quantize a tensor for its new tier.

        Args:
            expert_id: For logging/tracking
            old_tier: Where the expert was stored
            new_tier: Where the expert is being moved to
            tensor: The expert weight tensor (float or already dequantized)

        Returns:
            QuantizedTensor at the new tier's bit-width
        """
        target_bits = self.tier_bits.get(new_tier, 8)
        group_size = min(self.group_size, tensor.numel())
        return quantize(tensor, bits=target_bits, group_size=group_size)

    def requantize(
        self,
        qt: QuantizedTensor,
        new_tier: str,
    ) -> QuantizedTensor:
        """Re-quantize from an existing QuantizedTensor to a new tier.

        Dequantizes first, then requantizes at the new level.
        """
        tensor = dequantize(qt)
        target_bits = self.tier_bits.get(new_tier, 8)
        group_size = min(self.group_size, tensor.numel())
        return quantize(tensor, bits=target_bits, group_size=group_size)

    def upcast_to_float(self, qt: QuantizedTensor) -> torch.Tensor:
        """Dequantize to float for hot-tier use (highest quality)."""
        return dequantize(qt)
