"""Mock engine adapter for testing.

Returns deterministic fake responses. No running engine needed.
Use in CI, unit tests, and local development without GPU.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from inference_interface import (
    CompletionRequest,
    CompletionResponse,
    EngineCapabilities,
    EngineHealth,
    InferenceEngine,
    ModelInfo,
    StreamChunk,
)

_FAKE_RESPONSE = "This is a mock response for testing purposes."


class MockEngineAdapter(InferenceEngine):
    @property
    def name(self) -> str:
        return "mock"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            streaming=True,
            batching=True,
            speculative_decoding=False,
            expert_aware_batching=False,
            max_concurrent_requests=100,
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(
            text=_FAKE_RESPONSE,
            model=request.model,
            engine=self.name,
            tokens_generated=len(_FAKE_RESPONSE.split()),
            tokens_per_second=999.0,
            finish_reason="stop",
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        for word in _FAKE_RESPONSE.split():
            yield StreamChunk(delta=word + " ", done=False, engine=self.name)
        yield StreamChunk(delta="", done=True, engine=self.name)

    async def health(self) -> EngineHealth:
        return EngineHealth(healthy=True, engine=self.name, latency_ms=0.1,
                            active_requests=0)

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id="mock-model", engine=self.name,
                          quantization="fp16", context_length=4096)]
