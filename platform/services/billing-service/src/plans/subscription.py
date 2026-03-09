"""Plan management (free / pro / enterprise).

Defines available subscription tiers, their token quotas, and helpers for
upgrading / downgrading organisations between plans.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from dana_common.logging import get_logger

logger = get_logger(__name__)


class PlanTier(StrEnum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass(frozen=True)
class PlanDefinition:
    tier: PlanTier
    display_name: str
    monthly_token_limit: int
    rate_limit_rpm: int  # requests per minute
    max_context_length: int
    price_cents: int  # monthly price in cents (USD)
    priority: int  # scheduling priority (higher = better)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "display_name": self.display_name,
            "monthly_token_limit": self.monthly_token_limit,
            "rate_limit_rpm": self.rate_limit_rpm,
            "max_context_length": self.max_context_length,
            "price_cents": self.price_cents,
            "priority": self.priority,
        }


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------
PLAN_CATALOGUE: dict[PlanTier, PlanDefinition] = {
    PlanTier.FREE: PlanDefinition(
        tier=PlanTier.FREE,
        display_name="Free",
        monthly_token_limit=500_000,
        rate_limit_rpm=20,
        max_context_length=4096,
        price_cents=0,
        priority=0,
    ),
    PlanTier.PRO: PlanDefinition(
        tier=PlanTier.PRO,
        display_name="Pro",
        monthly_token_limit=10_000_000,
        rate_limit_rpm=120,
        max_context_length=16384,
        price_cents=4900,
        priority=5,
    ),
    PlanTier.ENTERPRISE: PlanDefinition(
        tier=PlanTier.ENTERPRISE,
        display_name="Enterprise",
        monthly_token_limit=500_000_000,
        rate_limit_rpm=1000,
        max_context_length=65536,
        price_cents=49900,
        priority=10,
    ),
}


@dataclass
class Subscription:
    """Represents an organisation's active subscription."""

    org_id: str
    tier: PlanTier
    started_at: datetime
    expires_at: datetime | None = None
    cancelled: bool = False

    @property
    def plan(self) -> PlanDefinition:
        return PLAN_CATALOGUE[self.tier]

    @property
    def is_active(self) -> bool:
        if self.cancelled:
            return False
        if self.expires_at and datetime.now(UTC) > self.expires_at:
            return False
        return True


class SubscriptionManager:
    """In-memory subscription manager (production version delegates to DB)."""

    def __init__(self) -> None:
        self._subscriptions: dict[str, Subscription] = {}

    def get(self, org_id: str) -> Subscription:
        if org_id not in self._subscriptions:
            sub = Subscription(
                org_id=org_id,
                tier=PlanTier.FREE,
                started_at=datetime.now(UTC),
            )
            self._subscriptions[org_id] = sub
        return self._subscriptions[org_id]

    def upgrade(self, org_id: str, tier: PlanTier) -> Subscription:
        current = self.get(org_id)
        if tier.value == current.tier.value:
            return current
        logger.info("subscription.upgrade", org_id=org_id, from_tier=current.tier.value, to_tier=tier.value)
        new_sub = Subscription(
            org_id=org_id,
            tier=tier,
            started_at=datetime.now(UTC),
        )
        self._subscriptions[org_id] = new_sub
        return new_sub

    def cancel(self, org_id: str) -> Subscription:
        sub = self.get(org_id)
        sub.cancelled = True
        logger.info("subscription.cancel", org_id=org_id)
        return sub

    def list_all(self) -> list[Subscription]:
        return list(self._subscriptions.values())
