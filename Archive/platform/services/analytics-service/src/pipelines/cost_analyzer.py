"""GPU cost attribution per request.

Attributes infrastructure cost to individual requests based on GPU time,
model size, and cluster pricing tiers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dana_common.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# GPU pricing table (cost per GPU-second in micro-cents)
# ---------------------------------------------------------------------------

_GPU_COST_PER_SEC: dict[str, float] = {
    "a100-80gb": 550.0,
    "a100-40gb": 420.0,
    "h100": 850.0,
    "l4": 120.0,
    "t4": 55.0,
    "default": 200.0,
}

# Approximate GPU-seconds per 1000 tokens by model family
_GPU_SECS_PER_1K_TOKENS: dict[str, float] = {
    "dana-xl": 0.45,
    "dana-large": 0.25,
    "dana-base": 0.10,
    "dana-code": 0.30,
    "dana-embed": 0.02,
    "default": 0.15,
}


def _resolve(table: dict[str, float], key: str) -> float:
    lower = key.lower()
    for k, v in table.items():
        if k in lower:
            return v
    return table["default"]


@dataclass(frozen=True)
class CostBreakdown:
    """Per-request cost attribution result."""

    model: str
    gpu_type: str
    total_tokens: int
    gpu_seconds: float
    cost_microcents: float
    cost_cents: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "gpu_type": self.gpu_type,
            "total_tokens": self.total_tokens,
            "gpu_seconds": round(self.gpu_seconds, 6),
            "cost_cents": round(self.cost_cents, 6),
        }


class CostAnalyzer:
    """Computes per-request GPU cost attribution."""

    def attribute(
        self,
        model: str,
        total_tokens: int,
        gpu_type: str = "a100-80gb",
    ) -> CostBreakdown:
        """Return the infrastructure cost breakdown for a single request."""
        secs_per_1k = _resolve(_GPU_SECS_PER_1K_TOKENS, model)
        cost_per_sec = _resolve(_GPU_COST_PER_SEC, gpu_type)

        gpu_seconds = (total_tokens / 1000.0) * secs_per_1k
        cost_micro = gpu_seconds * cost_per_sec
        cost_cents = cost_micro / 1_000_000

        breakdown = CostBreakdown(
            model=model,
            gpu_type=gpu_type,
            total_tokens=total_tokens,
            gpu_seconds=gpu_seconds,
            cost_microcents=cost_micro,
            cost_cents=cost_cents,
        )
        logger.debug("cost_attributed", model=model, gpu_type=gpu_type, cost_cents=round(cost_cents, 6))
        return breakdown

    def attribute_batch(
        self,
        requests: list[dict[str, Any]],
        default_gpu: str = "a100-80gb",
    ) -> list[CostBreakdown]:
        """Attribute cost for a batch of requests.

        Each dict must contain 'model' and 'total_tokens' keys.
        """
        return [
            self.attribute(
                model=r["model"],
                total_tokens=r["total_tokens"],
                gpu_type=r.get("gpu_type", default_gpu),
            )
            for r in requests
        ]
