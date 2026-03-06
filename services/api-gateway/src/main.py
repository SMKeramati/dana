"""Dana API Gateway - OpenAI-compatible API entry point."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .middleware.request_logger import RequestLoggerMiddleware
from .routes.health import router as health_router
from .routes.v1_chat import router as chat_router
from .routes.v1_models import router as models_router

app = FastAPI(
    title="Dana AI API",
    version="0.1.0",
    description="Frontier AI on Iranian Infrastructure. OpenAI-compatible API.",
)

# Middleware
app.add_middleware(RequestLoggerMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(chat_router, tags=["chat"])
app.include_router(models_router, tags=["models"])
app.include_router(health_router, tags=["health"])
