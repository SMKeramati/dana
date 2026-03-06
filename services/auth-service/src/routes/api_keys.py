"""API key management endpoints."""

from __future__ import annotations

from dana_common.models import APIKeyCreate, APIKeyResponse
from fastapi import APIRouter, Header, HTTPException

from ..crypto.api_key_gen import api_key_generator
from ..crypto.token_engine import token_engine
from ..db import repository

router = APIRouter()


async def _get_user_id(authorization: str) -> int:
    """Extract user_id from Bearer token."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization[7:]
    payload = token_engine.verify_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload["sub"]


@router.post("/auth/api-keys", response_model=APIKeyResponse, status_code=201)
async def create_api_key(
    body: APIKeyCreate,
    authorization: str = Header(...),
) -> APIKeyResponse:
    user_id = await _get_user_id(authorization)
    user = await repository.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    key, key_hash = api_key_generator.generate(user.tier, body.permissions)
    key_prefix = key[:10]

    api_key = await repository.create_api_key(
        user_id=user_id,
        name=body.name,
        key_hash=key_hash,
        key_prefix=key_prefix,
        permissions=",".join(body.permissions),
    )

    return APIKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key=key,  # Only shown once
        permissions=body.permissions,
        created_at=api_key.created_at,
    )


@router.get("/auth/api-keys", response_model=list[dict])
async def list_api_keys(authorization: str = Header(...)) -> list[dict]:
    user_id = await _get_user_id(authorization)
    keys = await repository.get_user_api_keys(user_id)
    return [
        {
            "id": k.id,
            "name": k.name,
            "prefix": k.key_prefix,
            "permissions": k.permissions.split(","),
            "created_at": str(k.created_at),
            "last_used": str(k.last_used) if k.last_used else None,
        }
        for k in keys
    ]
