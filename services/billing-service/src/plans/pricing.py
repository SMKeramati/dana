"""Dynamic pricing engine with tier multipliers.

Computes per-request cost by combining a base per-token rate with tier-level
multipliers, model complexity factors, and optional volume discounts.
"""
from __future__ import annotations

from dataclasses import dataclass

from dana_common.logging import get_logger

from .subscription import PlanTier

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Rate tables (cost in micro-cents per token, i.e. 1e-6 cents)
# ------------------------------------------------------------------

_BASE_RATES: dict[str, dict[str, float]] = {
    "gpt-4": {"prompt": 30.0, "completion": 60.0},
    "gpt-3.5-turbo": {"prompt": 1.5, "completion": 2.0},
    "claude": {"prompt": 15.0, "completion": 50.0},
    "llama": {"prompt": 0.8, "completion": 1.0},
    "default": {"prompt": 5.0, "completion": 10.0},
}

_TIER_MULTIPLIERS: dict[PlanTier, float] = {
    PlanTier.FREE: 1.0,
    PlanTier.PRO: 0.85,
    PlanTier.ENTERPRISE: 0.65,
}

# Volume discount tiers: (threshold_tokens, discount_factor)
_VOLUME_DISCOUNTS: list[tuple[int, float]] = [
    (50_000_000, 0.80),
    (10_000_000, 0.90),
    (1_000_000, 0.95),
]


def _resolve_base_rate(model: str) -> dict[str, float]:
    lower = model.lower()
    for key, rates in _BASE_RATES.items():
        if key in lower:
            return rates
    return _BASE_RATES["default"]


def _volume_discount(monthly_tokens: int) -> float:
    """Return the discount factor based on cumulative monthly token usage."""
    for threshold, factor in _VOLUME_DISCOUNTS:
        if monthly_tokens >= threshold:
            return factor
    return 1.0


@dataclass(frozen=True)
class PriceBreakdown:
    prompt_cost_microcents: float
    completion_cost_microcents: float
    total_microcents: float
    total_cents: float
    tier_multiplier: float
    volume_discount: float
    model: str


class PricingEngine:
    """Computes request cost from token counts, plan tier, and volume."""

    def compute(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        tier: PlanTier,
        monthly_tokens_so_far: int = 0,
    ) -> PriceBreakdown:
        base = _resolve_base_rate(model)
        tier_mult = _TIER_MULTIPLIERS.get(tier, 1.0)
        vol_disc = _volume_discount(monthly_tokens_so_far)

        prompt_cost = prompt_tokens * base["prompt"] * tier_mult * vol_disc
        completion_cost = completion_tokens * base["completion"] * tier_mult * vol_disc
        total_micro = prompt_cost + completion_cost
        total_cents = total_micro / 1_000_000

        breakdown = PriceBreakdown(
            prompt_cost_microcents=prompt_cost,
            completion_cost_microcents=completion_cost,
            total_microcents=total_micro,
            total_cents=total_cents,
            tier_multiplier=tier_mult,
            volume_discount=vol_disc,
            model=model,
        )
        logger.debug(
            "price_computed",
            model=model,
            tier=tier.value,
            total_cents=round(total_cents, 6),
        )
        return breakdown
