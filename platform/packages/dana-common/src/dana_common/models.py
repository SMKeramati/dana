"""Shared Pydantic models used across Dana services."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# === User & Auth Models ===

class UserTier(StrEnum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class UserCreate(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    tier: UserTier
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class APIKeyCreate(BaseModel):
    name: str
    permissions: list[str] = Field(default_factory=lambda: ["chat"])


class APIKeyResponse(BaseModel):
    id: int
    name: str
    key: str  # Only shown once at creation
    permissions: list[str]
    created_at: datetime


# === OpenAI-Compatible Chat Models ===

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "qwen3-235b-moe"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logprobs: bool = False
    top_logprobs: int | None = None  # 0-20, OpenAI-compatible


class TopLogprob(BaseModel):
    """Top alternative token with its log probability."""
    token: str
    logprob: float
    bytes: list[int] | None = None


class TokenLogprob(BaseModel):
    """Log probability for a single token (OpenAI-compatible)."""
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprob] = Field(default_factory=list)


class ChoiceLogprobs(BaseModel):
    """Logprobs container for a choice (OpenAI-compatible)."""
    content: list[TokenLogprob] | None = None


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"
    logprobs: ChoiceLogprobs | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[dict[str, Any]]


# === Model Info ===

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "dana"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# === Inference Job ===

class InferenceJob(BaseModel):
    job_id: str
    api_key_id: int
    user_id: int
    user_tier: UserTier
    request: ChatCompletionRequest
    created_at: datetime = Field(default_factory=datetime.utcnow)


class InferenceResult(BaseModel):
    job_id: str
    response: ChatCompletionResponse | None = None
    error: str | None = None


# === Usage & Billing ===

class UsageRecord(BaseModel):
    user_id: int
    api_key_id: int
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UsageEvent(BaseModel):
    """Published to RabbitMQ after each inference."""
    event_type: str = "inference_completed"
    user_id: int
    api_key_id: int
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# === Health ===

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    version: str = "0.1.0"


# === Error ===

class ErrorResponse(BaseModel):
    error: ErrorDetail


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: str | None = None
