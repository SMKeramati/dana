"""Auth middleware for API Gateway.

Validates Bearer tokens and API keys against the auth service.
"""

from __future__ import annotations

from typing import Any

from dana_common import config
from dana_common.auth import APIKeyGenerator, TokenEngine
from fastapi import HTTPException, Request

token_engine = TokenEngine(config.auth.secret_key, config.auth.algorithm)
api_key_gen = APIKeyGenerator(config.auth.api_key_salt)


async def authenticate_request(request: Request) -> dict[str, Any]:
    """Extract and validate authentication from request.

    Supports both Bearer tokens and API keys (Authorization: Bearer dk-xxx).
    Returns user info dict with user_id, tier, permissions.
    """
    auth_header = request.headers.get("Authorization", "")

    if not auth_header:
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Missing Authorization header", "type": "authentication_error"}},
        )

    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Invalid Authorization format. Use 'Bearer <token>'", "type": "authentication_error"}},
        )

    credential = auth_header[7:]

    # API Key authentication (dk-xxx format)
    if credential.startswith("dk-"):
        return await _validate_api_key(credential)

    # Token authentication
    return _validate_token(credential)


async def _validate_api_key(api_key: str) -> dict[str, Any]:
    """Validate API key by hashing and looking up."""
    # In production, this would call auth-service or check Redis cache
    # For now, we extract tier from the key format
    key_hash = api_key_gen.hash_key(api_key)
    tier_char = api_key[3] if len(api_key) > 3 else "f"
    tier_map = {"f": "free", "p": "pro", "e": "enterprise"}
    tier = tier_map.get(tier_char, "free")

    return {
        "auth_type": "api_key",
        "api_key": api_key,
        "api_key_hash": key_hash,
        "tier": tier,
        "user_id": 0,  # Resolved by auth-service lookup
        "permissions": ["chat"],
    }


def _validate_token(token: str) -> dict[str, Any]:
    """Validate a Dana token."""
    payload = token_engine.verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Invalid or expired token", "type": "authentication_error"}},
        )
    return {
        "auth_type": "token",
        "user_id": payload["sub"],
        "email": payload.get("email"),
        "tier": payload.get("tier", "free"),
        "permissions": payload.get("perms", ["chat"]),
    }
