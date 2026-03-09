"""User registration endpoint."""

from __future__ import annotations

from dana_common.models import UserCreate, UserResponse, UserTier
from fastapi import APIRouter, HTTPException
from passlib.hash import bcrypt

from ..db import repository

router = APIRouter()


@router.post("/auth/register", response_model=UserResponse, status_code=201)
async def register(body: UserCreate) -> UserResponse:
    existing = await repository.get_user_by_email(body.email)
    if existing is not None:
        raise HTTPException(status_code=409, detail="Email already registered")

    password_hash = bcrypt.hash(body.password)
    user = await repository.create_user(body.email, password_hash)

    return UserResponse(
        id=user.id,
        email=user.email,
        tier=UserTier(user.tier),
        created_at=user.created_at,
    )
