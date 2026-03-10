"""Inference gateway — engine-agnostic request router.

Switching engines:
    ENGINE=dana    → routes to dana-engine (custom MoE engine)
    ENGINE=sglang  → routes to sglang-worker (original engine)
    ENGINE=mock    → mock adapter for testing

A/B testing (future):
    ENGINE=dana:90,sglang:10  → 90% dana, 10% sglang
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from inference_interface import CompletionRequest, EngineRegistry

logger = logging.getLogger(__name__)

registry = EngineRegistry()
engine_name = os.environ.get("ENGINE", "dana")
engine = registry.load(engine_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting inference gateway with engine: {engine.name}")
    logger.info(f"Engine capabilities: {engine.capabilities}")
    await engine.startup()
    yield
    await engine.shutdown()
    logger.info("Inference gateway shut down")


app = FastAPI(title="Dana Inference Gateway", lifespan=lifespan)


@app.get("/health")
async def health():
    status = await engine.health()
    if not status.healthy:
        raise HTTPException(status_code=503, detail=status.message)
    return {"engine": status.engine, "healthy": True,
            "latency_ms": status.latency_ms,
            "active_requests": status.active_requests}


@app.get("/engine")
async def engine_info():
    """Which engine is currently active and what does it support."""
    return {
        "engine": engine.name,
        "capabilities": {
            "streaming": engine.capabilities.streaming,
            "batching": engine.capabilities.batching,
            "speculative_decoding": engine.capabilities.speculative_decoding,
            "expert_aware_batching": engine.capabilities.expert_aware_batching,
            "max_concurrent_requests": engine.capabilities.max_concurrent_requests,
            "supported_quantizations": engine.capabilities.supported_quantizations,
        }
    }


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if request.stream:
        async def generate():
            async for chunk in engine.stream(request):
                if chunk.done:
                    yield "data: [DONE]\n\n"
                else:
                    import json
                    yield f"data: {json.dumps({'choices': [{'text': chunk.delta}]})}\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")

    response = await engine.complete(request)
    return {
        "model": response.model,
        "engine": response.engine,
        "choices": [{"text": response.text, "finish_reason": response.finish_reason}],
        "usage": {"completion_tokens": response.tokens_generated},
        "dana_meta": {"tokens_per_second": response.tokens_per_second},
    }


@app.get("/v1/models")
async def list_models():
    models = await engine.list_models()
    return {"data": [
        {"id": m.id, "engine": m.engine,
         "quantization": m.quantization,
         "context_length": m.context_length}
        for m in models
    ]}
