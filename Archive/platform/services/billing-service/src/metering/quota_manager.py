"""Real-time quota enforcement with Redis distributed counters.

Uses Redis INCRBY with TTL-based expiry to enforce per-org, per-model
token quotas across multiple service instances.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import redis.asyncio as redis
from dana_common.logging import get_logger

logger = get_logger(__name__)


class QuotaWindow(StrEnum):
    MINUTELY = "minutely"
    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"


_WINDOW_TTL: dict[QuotaWindow, int] = {
    QuotaWindow.MINUTELY: 120,
    QuotaWindow.HOURLY: 7200,
    QuotaWindow.DAILY: 172800,
    QuotaWindow.MONTHLY: 2764800,
}


@dataclass(frozen=True)
class QuotaResult:
    allowed: bool
    current_usage: int
    limit: int
    remaining: int
    window: QuotaWindow

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "current_usage": self.current_usage,
            "limit": self.limit,
            "remaining": self.remaining,
            "window": self.window.value,
        }


class QuotaManager:
    """Distributed quota enforcement backed by Redis.

    Each quota is tracked as a Redis key of the form
    ``quota:{org_id}:{model}:{window}:{bucket}`` with an appropriate TTL.
    """

    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check_and_increment(
        self,
        *,
        org_id: str,
        model: str,
        tokens: int,
        limit: int,
        window: QuotaWindow = QuotaWindow.MONTHLY,
        bucket_key: str = "",
    ) -> QuotaResult:
        """Atomically check whether *tokens* can be consumed and, if so,
        increment the counter.

        Returns a :class:`QuotaResult` indicating whether the request was
        allowed.
        """
        rkey = self._key(org_id, model, window, bucket_key)
        ttl = _WINDOW_TTL[window]

        # Lua script for atomic check-and-increment
        lua = """
        local current = tonumber(redis.call('GET', KEYS[1]) or '0')
        local requested = tonumber(ARGV[1])
        local cap = tonumber(ARGV[2])
        local ttl = tonumber(ARGV[3])
        if current + requested <= cap then
            redis.call('INCRBY', KEYS[1], requested)
            redis.call('EXPIRE', KEYS[1], ttl)
            return current + requested
        else
            return -1
        end
        """
        raw = await self._redis.eval(  # type: ignore[misc]
            lua, 1, rkey, str(tokens), str(limit), str(ttl),
        )
        result = int(raw)

        if result == -1:
            current = int(await self._redis.get(rkey) or 0)
            logger.warning(
                "quota_exceeded",
                org_id=org_id,
                model=model,
                window=window.value,
                current=current,
                requested=tokens,
                limit=limit,
            )
            return QuotaResult(
                allowed=False,
                current_usage=current,
                limit=limit,
                remaining=max(0, limit - current),
                window=window,
            )

        logger.debug(
            "quota_ok",
            org_id=org_id,
            model=model,
            window=window.value,
            current=result,
            limit=limit,
        )
        return QuotaResult(
            allowed=True,
            current_usage=result,
            limit=limit,
            remaining=max(0, limit - result),
            window=window,
        )

    async def get_usage(
        self,
        *,
        org_id: str,
        model: str,
        window: QuotaWindow = QuotaWindow.MONTHLY,
        bucket_key: str = "",
    ) -> int:
        """Return the current token count for the given key."""
        rkey = self._key(org_id, model, window, bucket_key)
        val = await self._redis.get(rkey)
        return int(val) if val else 0

    async def reset(
        self,
        *,
        org_id: str,
        model: str,
        window: QuotaWindow = QuotaWindow.MONTHLY,
        bucket_key: str = "",
    ) -> None:
        """Reset the counter (admin operation)."""
        rkey = self._key(org_id, model, window, bucket_key)
        await self._redis.delete(rkey)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _key(org_id: str, model: str, window: QuotaWindow, bucket_key: str) -> str:
        suffix = bucket_key or "current"
        return f"quota:{org_id}:{model}:{window.value}:{suffix}"
