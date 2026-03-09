"""Custom HMAC-SHA512 token engine.

This is an INTERNAL R&D module - custom token format, NOT standard JWT.
Uses HMAC-SHA512 with embedded permission scopes and expiry.

Token format: base64(header).base64(payload).base64(signature)
Where signature = HMAC-SHA512(header.payload, secret_key)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any


class TokenEngine:
    """Custom token engine using HMAC-SHA512.

    Daneshbonyan: Internal Design & Development - Custom security implementation.
    """

    def __init__(self, secret_key: str, algorithm: str = "HS512") -> None:
        self._secret = secret_key.encode("utf-8")
        self._algorithm = algorithm

    def create_token(
        self,
        user_id: int,
        email: str,
        tier: str,
        permissions: list[str],
        expiry_minutes: int = 60,
    ) -> str:
        header = {"alg": self._algorithm, "typ": "DANA"}
        payload = {
            "sub": user_id,
            "email": email,
            "tier": tier,
            "perms": permissions,
            "iat": int(time.time()),
            "exp": int(time.time()) + (expiry_minutes * 60),
            "jti": secrets.token_hex(16),
        }
        header_b64 = self._encode(header)
        payload_b64 = self._encode(payload)
        signing_input = f"{header_b64}.{payload_b64}"
        signature = self._sign(signing_input)
        return f"{header_b64}.{payload_b64}.{signature}"

    def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify token and return payload, or None if invalid/expired."""
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, signature = parts
        signing_input = f"{header_b64}.{payload_b64}"
        expected_sig = self._sign(signing_input)
        if not hmac.compare_digest(signature, expected_sig):
            return None
        payload = self._decode(payload_b64)
        if payload is None:
            return None
        if payload.get("exp", 0) < time.time():
            return None
        return payload

    def _sign(self, data: str) -> str:
        sig = hmac.new(self._secret, data.encode("utf-8"), hashlib.sha512).digest()
        return base64.urlsafe_b64encode(sig).rstrip(b"=").decode("ascii")

    def _encode(self, data: dict[str, Any]) -> str:
        raw = json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    def _decode(self, data: str) -> dict[str, Any] | None:
        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        try:
            raw = base64.urlsafe_b64decode(data)
            result: dict[str, Any] = json.loads(raw)
            return result
        except Exception:
            return None


class APIKeyGenerator:
    """Custom API key generator with embedded tier/permission encoding.

    Key format: dk-{tier_code}{permission_bits}_{random_hex}
    Daneshbonyan: Internal Design & Development.
    """

    TIER_CODES = {"free": "f", "pro": "p", "enterprise": "e"}
    PERM_BITS = {"chat": "1", "models": "2", "admin": "4"}

    def __init__(self, salt: str) -> None:
        self._salt = salt.encode("utf-8")

    def generate(self, tier: str, permissions: list[str]) -> tuple[str, str]:
        """Generate API key. Returns (key, key_hash)."""
        tier_code = self.TIER_CODES.get(tier, "f")
        perm_bits = "".join(self.PERM_BITS.get(p, "0") for p in sorted(permissions))
        random_part = secrets.token_hex(24)
        key = f"dk-{tier_code}{perm_bits}_{random_part}"
        key_hash = self._hash_key(key)
        return key, key_hash

    def hash_key(self, key: str) -> str:
        return self._hash_key(key)

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(self._salt + key.encode("utf-8")).hexdigest()
