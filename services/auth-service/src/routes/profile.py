"""User profile and token refresh endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException

from ..crypto.token_engine import token_engine
from ..db import repository

router = APIRouter()


async def _get_payload(authorization: str) -> dict[str, Any]:
    """Extract full payload from Bearer token."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization[7:]
    payload = token_engine.verify_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


@router.get("/auth/me")
async def get_profile(authorization: str = Header(...)) -> dict[str, Any]:
    payload = await _get_payload(authorization)
    user = await repository.get_user_by_id(payload["sub"])
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": user.id,
        "email": user.email,
        "tier": user.tier,
        "created_at": str(user.created_at),
    }


@router.post("/auth/refresh")
async def refresh_token(authorization: str = Header(...)) -> dict[str, Any]:
    payload = await _get_payload(authorization)
    user = await repository.get_user_by_id(payload["sub"])
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    new_token = token_engine.create_token(
        user_id=user.id,
        email=user.email,
        tier=user.tier,
    )
    return {"access_token": new_token, "token_type": "bearer"}
