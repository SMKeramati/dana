"""Dana Engine API server — OpenAI-compatible HTTP interface.

Contract (from DanaEngineAdapter in inference-gateway):
  POST /v1/completions
    request:  {model, prompt, max_tokens, temperature, top_p, stream}
    response (stream=False):
      {model, choices: [{text, finish_reason}],
       usage: {completion_tokens}, dana_meta: {tokens_per_second}}
    response (stream=True):  SSE lines
      data: {"choices": [{"text": "<delta>"}]}
      data: [DONE]

  GET /v1/models
    response: {data: [{id, quantization, context_length}]}

  GET /health
    response: {latency_ms, active_requests}
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from dana_engine.pipeline import DanaInferencePipeline, PipelineConfig

# Global pipeline instance — created on startup
_pipeline: DanaInferencePipeline | None = None
_active_requests: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    global _pipeline
    _pipeline = DanaInferencePipeline(PipelineConfig())
    yield
    _pipeline = None


app = FastAPI(title="Dana Engine", version="0.1.0", lifespan=lifespan)


def get_pipeline() -> DanaInferencePipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized — server not started via lifespan")
    return _pipeline


def _tokenize(prompt: str, vocab_size: int) -> torch.Tensor:
    """Synthetic tokenizer: convert string chars to token IDs mod vocab_size.
    In production: replace with real BPE tokenizer.
    """
    ids = [ord(c) % vocab_size for c in prompt] if prompt else [0]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def _detokenize(token_ids: list[int]) -> str:
    """Synthetic detokenizer: convert token IDs back to a string.
    In production: replace with real BPE decoder.
    """
    return " ".join(str(t) for t in token_ids)


@app.post("/v1/completions", response_model=None)
async def completions(request: Request):
    """Handle completion requests — streaming and non-streaming."""
    global _active_requests
    _active_requests += 1
    try:
        body = await request.json()
        model_id: str = body.get("model", "tiny-moe")
        prompt: str = body.get("prompt", "")
        max_tokens: int = int(body.get("max_tokens", 32))
        stream: bool = bool(body.get("stream", False))

        pipeline = get_pipeline()
        vocab_size = pipeline.config.model_config.vocab_size
        input_ids = _tokenize(prompt, vocab_size)

        if stream:
            return StreamingResponse(
                _stream_generator(pipeline, input_ids, max_tokens, model_id),
                media_type="text/event-stream",
            )
        else:
            t0 = time.perf_counter()
            result = await pipeline.generate_async(input_ids, max_new_tokens=max_tokens)
            latency_ms = (time.perf_counter() - t0) * 1000

            text = _detokenize(result.tokens)
            return {
                "model": model_id,
                "choices": [{"text": text, "finish_reason": result.finish_reason}],
                "usage": {"completion_tokens": result.tokens_generated},
                "dana_meta": {
                    "tokens_per_second": round(result.tokens_per_second, 2),
                    "latency_ms": round(latency_ms, 2),
                    "spec_decode_used": result.spec_decode_used,
                    "avg_tokens_per_step": round(result.avg_tokens_per_step, 2),
                },
            }
    finally:
        _active_requests -= 1


async def _stream_generator(
    pipeline: DanaInferencePipeline,
    input_ids: torch.Tensor,
    max_tokens: int,
    model_id: str,
):
    """SSE generator for streaming completions."""
    async for token_str in pipeline.stream_async(input_ids, max_new_tokens=max_tokens):
        data = json.dumps({"choices": [{"text": token_str + " "}]})
        yield f"data: {data}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def list_models() -> dict:
    """Return available model IDs."""
    return {
        "data": [
            {
                "id": "tiny-moe",
                "quantization": "fp16",
                "context_length": 128,
            }
        ]
    }


@app.get("/health")
async def health() -> dict:
    """Return health status."""
    t0 = time.perf_counter()
    pipeline = get_pipeline()
    status = pipeline.health()
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "healthy": status["healthy"],
        "latency_ms": round(latency_ms, 3),
        "active_requests": _active_requests,
        "spec_decode": status["spec_decode"],
    }
