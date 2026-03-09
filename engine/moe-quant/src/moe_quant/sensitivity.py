"""ExpertSensitivityProfiler — measure quality loss at each bit-width.

For each expert and each bit-width, runs the expert forward pass with
quantized weights and measures the output similarity to the full-precision
baseline using cosine similarity.

Returns a profile: {bits: quality_score} where 1.0 = perfect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from moe_quant.quantize import quantize
from moe_quant.dequantize import dequantize

if TYPE_CHECKING:
    from dana_engine.model.moe_layer import ExpertFFN


@dataclass
class SensitivityProfile:
    expert_id: int
    scores: dict[int, float]  # {bits: cosine_similarity}

    def recommended_bits(self, min_quality: float = 0.99) -> int:
        """Return lowest bit-width that meets minimum quality threshold."""
        for bits in sorted(self.scores.keys()):
            if self.scores[bits] >= min_quality:
                return bits
        return max(self.scores.keys())  # fallback: highest precision


class ExpertSensitivityProfiler:
    """Measure per-expert quality degradation at each quantization level.

    Usage:
        profiler = ExpertSensitivityProfiler(bits_list=[2, 4, 8])
        profile = profiler.profile(expert, expert_id=3, calibration_data=x)
        # profile.scores = {2: 0.85, 4: 0.97, 8: 0.999}
    """

    def __init__(
        self,
        bits_list: list[int] = None,
        group_size: int = 128,
        num_calibration_samples: int = 16,
    ) -> None:
        self.bits_list = bits_list or [2, 4, 8]
        self.group_size = group_size
        self.num_calibration_samples = num_calibration_samples

    def profile(
        self,
        expert: "ExpertFFN",
        expert_id: int,
        calibration_data: torch.Tensor,
    ) -> SensitivityProfile:
        """Profile an expert's sensitivity at each bit-width.

        Args:
            expert: ExpertFFN module
            expert_id: Integer ID (for tracking)
            calibration_data: (N, hidden_dim) input activations

        Returns:
            SensitivityProfile with cosine similarity scores per bit-width
        """
        expert.eval()
        x = calibration_data[:self.num_calibration_samples].detach()

        with torch.no_grad():
            # Baseline: full precision output
            baseline_out = expert(x)  # (N, hidden_dim)

            scores: dict[int, float] = {}
            for bits in self.bits_list:
                # Quantize and dequantize each weight matrix
                quant_expert = self._quantize_expert(expert, bits)
                quant_out = quant_expert(x)

                # Cosine similarity between baseline and quantized outputs
                sim = F.cosine_similarity(
                    baseline_out.flatten(),
                    quant_out.flatten(),
                    dim=0,
                ).item()
                scores[bits] = float(sim)

        return SensitivityProfile(expert_id=expert_id, scores=scores)

    def profile_all(
        self,
        experts: list["ExpertFFN"],
        calibration_data: torch.Tensor,
    ) -> list[SensitivityProfile]:
        """Profile all experts in a list."""
        return [
            self.profile(expert, i, calibration_data)
            for i, expert in enumerate(experts)
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _quantize_expert(self, expert: "ExpertFFN", bits: int) -> "ExpertFFN":
        """Return a copy of the expert with quantized (then dequantized) weights."""
        import copy
        quant_expert = copy.deepcopy(expert)

        with torch.no_grad():
            for param in quant_expert.parameters():
                qt = quantize(param.data, bits=bits, group_size=min(self.group_size, param.numel()))
                param.data = dequantize(qt).to(param.dtype)

        return quant_expert
