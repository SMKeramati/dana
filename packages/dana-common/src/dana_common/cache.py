"""Custom Redis cache with invalidation strategy.

Daneshbonyan: Internal Design & Development - Custom caching with
sliding-window tracking and hierarchical invalidation.
"""

from __future__ import annotations

import json
import time
from typing import Any

import redis


class CacheManager:
    """Custom cache manager with hierarchical invalidation and TTL tracking."""

    def __init__(self, redis_url: str) -> None:
        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._default_ttl = 300  # 5 minutes

    def get(self, namespace: str, key: str) -> Any | None:
        full_key = f"{namespace}:{key}"
        raw = self._redis.get(full_key)
        if raw is None:
            return None
        return json.loads(raw)

    def set(
        self, namespace: str, key: str, value: Any, ttl: int | None = None
    ) -> None:
        full_key = f"{namespace}:{key}"
        self._redis.set(full_key, json.dumps(value), ex=ttl or self._default_ttl)
        # Track key in namespace set for bulk invalidation
        self._redis.sadd(f"_ns:{namespace}", full_key)

    def delete(self, namespace: str, key: str) -> None:
        full_key = f"{namespace}:{key}"
        self._redis.delete(full_key)
        self._redis.srem(f"_ns:{namespace}", full_key)

    def invalidate_namespace(self, namespace: str) -> int:
        """Invalidate all keys in a namespace. Returns count of deleted keys."""
        ns_key = f"_ns:{namespace}"
        keys = self._redis.smembers(ns_key)
        if not keys:
            return 0
        count: int = self._redis.delete(*keys)
        self._redis.delete(ns_key)
        return count

    def close(self) -> None:
        self._redis.close()


class SlidingWindowCounter:
    """Custom sliding-window rate counter using Redis sorted sets.

    Daneshbonyan: Internal Design & Development - Not off-the-shelf slowapi.
    Uses sorted sets with timestamp scores for precise sliding-window counting.
    """

    def __init__(self, redis_url: str) -> None:
        self._redis = redis.from_url(redis_url, decode_responses=True)

    def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        """Check if request is within rate limit.

        Returns (allowed, remaining_count).
        """
        now = time.time()
        window_start = now - window_seconds
        pipe = self._redis.pipeline()
        # Remove expired entries
        pipe.zremrangebyscore(key, 0, window_start)
        # Count current window
        pipe.zcard(key)
        # Add current request (optimistic)
        pipe.zadd(key, {f"{now}:{id(key)}": now})
        # Set expiry on the key
        pipe.expire(key, window_seconds + 1)
        results = pipe.execute()
        current_count = results[1]
        if current_count >= limit:
            # Over limit - remove the entry we just added
            self._redis.zremrangebyscore(key, now, now)
            return False, 0
        remaining = limit - current_count - 1
        return True, remaining

    def get_count(self, key: str, window_seconds: int) -> int:
        """Get current count in the sliding window."""
        now = time.time()
        window_start = now - window_seconds
        self._redis.zremrangebyscore(key, 0, window_start)
        result: int = self._redis.zcard(key)
        return result

    def close(self) -> None:
        self._redis.close()
