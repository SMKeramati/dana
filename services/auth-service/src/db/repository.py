"""Database repository for auth operations."""

from __future__ import annotations

from datetime import datetime

from dana_common import config
from sqlalchemy import and_, delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .models import APIKey, Base, User

engine = create_async_engine(config.db.url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def create_user(email: str, password_hash: str) -> User:
    async with async_session() as session:
        user = User(email=email, password_hash=password_hash)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


async def get_user_by_email(email: str) -> User | None:
    async with async_session() as session:
        result = await session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()


async def get_user_by_id(user_id: int) -> User | None:
    async with async_session() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()


async def create_api_key(
    user_id: int, name: str, key_hash: str, key_prefix: str, permissions: str
) -> APIKey:
    async with async_session() as session:
        api_key = APIKey(
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            permissions=permissions,
        )
        session.add(api_key)
        await session.commit()
        await session.refresh(api_key)
        return api_key


async def get_api_key_by_hash(key_hash: str) -> APIKey | None:
    async with async_session() as session:
        result = await session.execute(
            select(APIKey).where(APIKey.key_hash == key_hash)
        )
        return result.scalar_one_or_none()


async def get_user_api_keys(user_id: int) -> list[APIKey]:
    async with async_session() as session:
        result = await session.execute(
            select(APIKey).where(APIKey.user_id == user_id)
        )
        return list(result.scalars().all())


async def update_api_key_last_used(key_hash: str) -> None:
    async with async_session() as session:
        await session.execute(
            update(APIKey)
            .where(APIKey.key_hash == key_hash)
            .values(last_used=datetime.utcnow())
        )
        await session.commit()


async def delete_api_key(key_id: int, user_id: int) -> bool:
    async with async_session() as session:
        result = await session.execute(
            delete(APIKey).where(
                and_(APIKey.id == key_id, APIKey.user_id == user_id)
            )
        )
        await session.commit()
        return result.rowcount > 0


async def update_user_tier(user_id: int, tier: str) -> User | None:
    async with async_session() as session:
        await session.execute(
            update(User).where(User.id == user_id).values(tier=tier)
        )
        await session.commit()
        return await get_user_by_id(user_id)
