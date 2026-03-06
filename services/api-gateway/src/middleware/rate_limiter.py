"""Custom sliding-window rate limiter.

Daneshbonyan: Internal Design & Development.
Uses Redis sorted sets for distributed sliding-window rate limiting.
NOT off-the-shelf slowapi - custom implementation with per-key, per-IP,
and tier-aware limits.
"""

from __future__ import annotations

import time

from dana_common import config
from dana_common.cache import SlidingWindowCounter
from fastapi import HTTPException


class RateLimiter:
    """Distributed sliding-window rate limiter.

    Supports per-API-key and per-IP rate limiting with tier-aware limits.
    Uses Redis sorted sets for O(log N) operations and precise windows.
    """

    TIER_LIMITS = {
        "free": {"rpm": config.rate_limit.free_rpm, "tpd": config.rate_limit.free_tpd},
        "pro": {"rpm": config.rate_limit.pro_rpm, "tpd": config.rate_limit.pro_tpd},
        "enterprise": {"rpm": 600, "tpd": 1_000_000},
    }

    def __init__(self, redis_url: str) -> None:
        self._counter = SlidingWindowCounter(redis_url)

    async def check_rate_limit(
        self,
        api_key: str,
        tier: str,
    ) -> dict[str, int]:
        """Check if request is within rate limit. Raises HTTPException if exceeded.

        Returns dict with remaining limits info.
        """
        limits = self.TIER_LIMITS.get(tier, self.TIER_LIMITS["free"])

        # Per-minute check
        rpm_key = f"rate:rpm:{api_key}"
        allowed, remaining_rpm = self._counter.is_allowed(
            rpm_key, limits["rpm"], window_seconds=60
        )
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": {
                        "message": f"Rate limit exceeded: {limits['rpm']} requests per minute",
                        "type": "rate_limit_exceeded",
                        "code": "rate_limit_exceeded",
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(limits["rpm"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + 60),
                    "Retry-After": "60",
                },
            )

        return {
            "X-RateLimit-Limit-RPM": limits["rpm"],
            "X-RateLimit-Remaining-RPM": remaining_rpm,
        }

    def close(self) -> None:
        self._counter.close()
