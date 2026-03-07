"""Automatic key rotation with zero-downtime migration.

Daneshbonyan: Internal Design & Development.
Maintains multiple active signing keys so tokens signed with old keys
remain valid during the rotation window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from dana_common.auth import TokenEngine


@dataclass
class KeyVersion:
    key: str
    engine: TokenEngine
    created_at: float
    expired_at: float | None = None


class KeyRotationManager:
    """Manages signing key versions for zero-downtime rotation."""

    def __init__(self, initial_key: str, algorithm: str = "HS512") -> None:
        self._algorithm = algorithm
        self._versions: list[KeyVersion] = []
        self._add_version(initial_key)

    @property
    def current_engine(self) -> TokenEngine:
        return self._versions[-1].engine

    def rotate(self, new_key: str, grace_period_hours: int = 24) -> None:
        """Rotate to a new key. Old key remains valid for grace_period."""
        now = time.time()
        for v in self._versions:
            if v.expired_at is None:
                v.expired_at = now + (grace_period_hours * 3600)
        self._add_version(new_key)

    def verify_token(self, token: str) -> dict[str, Any] | None:
        """Try all active key versions (newest first)."""
        now = time.time()
        for version in reversed(self._versions):
            if version.expired_at is not None and version.expired_at < now:
                continue
            payload = version.engine.verify_token(token)
            if payload is not None:
                return payload
        return None

    def cleanup_expired(self) -> int:
        """Remove expired key versions. Returns count of removed."""
        now = time.time()
        before = len(self._versions)
        self._versions = [
            v for v in self._versions
            if v.expired_at is None or v.expired_at >= now
        ]
        # Always keep at least the current version
        if not self._versions:
            self._versions = [self._versions[-1]] if before > 0 else []
        return before - len(self._versions)

    def _add_version(self, key: str) -> None:
        self._versions.append(
            KeyVersion(
                key=key,
                engine=TokenEngine(key, self._algorithm),
                created_at=time.time(),
            )
        )
