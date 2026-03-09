"""
InferenceEngine — the contract every engine adapter must satisfy.

Both dana-engine and sglang-worker implement this interface.
The inference-gateway loads the correct adapter at startup via ENGINE env var.
Switching engines = changing one environment variable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

from .types import CompletionRequest, CompletionResponse, StreamChunk, EngineHealth, ModelInfo


@dataclass
class EngineCapabilities:
    """Declares what a given engine supports. Gateway uses this to reject
    unsupported requests early rather than letting them fail downstream."""
    streaming: bool = True
    batching: bool = False          # True only for dana-engine
    speculative_decoding: bool = False  # True only for dana-engine
    expert_aware_batching: bool = False  # True only for dana-engine
    max_concurrent_requests: int = 1
    supported_quantizations: list[str] = None  # e.g. ["fp16", "q4", "q2"]

    def __post_init__(self):
        if self.supported_quantizations is None:
            self.supported_quantizations = ["fp16"]


class InferenceEngine(ABC):
    """Abstract base class for all inference engine adapters.

    Implementations:
      - DanaEngineAdapter   → engine/dana-engine  (the new custom MoE engine)
      - SGLangAdapter       → platform/services/sglang-worker  (old engine)
      - MockEngineAdapter   → for testing without a running engine
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name, e.g. 'dana-engine', 'sglang'."""

    @property
    @abstractmethod
    def capabilities(self) -> EngineCapabilities:
        """Declare what this engine supports."""

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Non-streaming completion."""

    @abstractmethod
    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Streaming completion, yields tokens as they are generated."""

    @abstractmethod
    async def health(self) -> EngineHealth:
        """Return current health/readiness of the engine."""

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Return the models this engine instance can serve."""

    async def startup(self) -> None:
        """Called once at gateway startup. Override for connection setup."""

    async def shutdown(self) -> None:
        """Called once at gateway shutdown. Override for cleanup."""
