"""Shared request/response types for all engine adapters.

These are OpenAI-compatible by design. Any engine that speaks OpenAI API
can be wrapped by an adapter with minimal translation work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CompletionRequest:
    model: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    stop: list[str] = field(default_factory=list)
    # Dana-engine-specific hints (ignored by sglang adapter)
    preferred_engine: Optional[str] = None  # force a specific engine


@dataclass
class CompletionResponse:
    text: str
    model: str
    engine: str          # which adapter actually served this
    tokens_generated: int
    tokens_per_second: float
    finish_reason: str   # "stop" | "length" | "error"


@dataclass
class StreamChunk:
    delta: str           # new text fragment
    done: bool = False
    engine: str = ""


@dataclass
class EngineHealth:
    healthy: bool
    engine: str
    latency_ms: float
    active_requests: int
    message: str = ""


@dataclass
class ModelInfo:
    id: str
    engine: str
    quantization: str
    context_length: int
