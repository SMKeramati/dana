"""Dana Auth Service - Custom authentication and API key management."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dana_common.models import HealthResponse
from fastapi import FastAPI

from .db.repository import init_db
from .routes.api_keys import router as api_keys_router
from .routes.login import router as login_router
from .routes.register import router as register_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await init_db()
    yield


app = FastAPI(title="Dana Auth Service", version="0.1.0", lifespan=lifespan)

app.include_router(register_router, tags=["auth"])
app.include_router(login_router, tags=["auth"])
app.include_router(api_keys_router, tags=["api-keys"])


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="auth-service")
