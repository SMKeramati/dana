"""User login endpoint."""

from __future__ import annotations

from dana_common import config
from dana_common.models import TokenResponse, UserCreate
from fastapi import APIRouter, HTTPException
from passlib.hash import bcrypt

from ..crypto.token_engine import token_engine
from ..db import repository

router = APIRouter()


@router.post("/auth/login", response_model=TokenResponse)
async def login(body: UserCreate) -> TokenResponse:
    user = await repository.get_user_by_email(body.email)
    if user is None or not bcrypt.verify(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = token_engine.create_token(
        user_id=user.id,
        email=user.email,
        tier=user.tier,
        permissions=["chat", "models"],
        expiry_minutes=config.auth.token_expiry_minutes,
    )

    return TokenResponse(
        access_token=token,
        expires_in=config.auth.token_expiry_minutes * 60,
    )
