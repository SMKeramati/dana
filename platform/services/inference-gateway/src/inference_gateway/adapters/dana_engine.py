"""Adapter for dana-engine — the custom MoE inference engine.

Connects to dana-engine's OpenAI-compatible HTTP API.
Engine URL configured via DANA_ENGINE_URL env var (default: http://dana-engine:8000).

Dana-specific capabilities:
  - Expert-aware batching (5-10x load reduction at concurrency)
  - MoE-self-draft speculative decoding (~85% acceptance)
  - Tree speculation (~7 tokens/step)
  - Per-expert hybrid quantization (Q2/Q4/Q8 per tier)
"""
from __future__ import annotations

import os
from typing import AsyncIterator

import httpx

from inference_interface import (
    InferenceEngine,
    EngineCapabilities,
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    EngineHealth,
    ModelInfo,
)


class DanaEngineAdapter(InferenceEngine):
    def __init__(self):
        self._url = os.environ.get("DANA_ENGINE_URL", "http://dana-engine:8000")
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "dana-engine"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            streaming=True,
            batching=True,
            speculative_decoding=True,
            expert_aware_batching=True,
            max_concurrent_requests=50,
            supported_quantizations=["fp16", "q8", "q4", "q2"],
        )

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(base_url=self._url, timeout=120.0)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": False,
        }
        resp = await self._client.post("/v1/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        return CompletionResponse(
            text=choice["text"],
            model=data["model"],
            engine=self.name,
            tokens_generated=data["usage"]["completion_tokens"],
            tokens_per_second=data.get("dana_meta", {}).get("tokens_per_second", 0.0),
            finish_reason=choice["finish_reason"],
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": True,
        }
        async with self._client.stream("POST", "/v1/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    raw = line[6:]
                    if raw == "[DONE]":
                        yield StreamChunk(delta="", done=True, engine=self.name)
                        return
                    import json
                    data = json.loads(raw)
                    delta = data["choices"][0].get("text", "")
                    yield StreamChunk(delta=delta, done=False, engine=self.name)

    async def health(self) -> EngineHealth:
        try:
            resp = await self._client.get("/health", timeout=5.0)
            data = resp.json()
            return EngineHealth(
                healthy=resp.status_code == 200,
                engine=self.name,
                latency_ms=data.get("latency_ms", 0),
                active_requests=data.get("active_requests", 0),
            )
        except Exception as e:
            return EngineHealth(healthy=False, engine=self.name, latency_ms=0,
                                active_requests=0, message=str(e))

    async def list_models(self) -> list[ModelInfo]:
        resp = await self._client.get("/v1/models")
        resp.raise_for_status()
        return [
            ModelInfo(
                id=m["id"],
                engine=self.name,
                quantization=m.get("quantization", "fp16"),
                context_length=m.get("context_length", 32768),
            )
            for m in resp.json()["data"]
        ]
