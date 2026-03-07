"""OpenAI-compatible /v1/chat/completions endpoint."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator

from dana_common.models import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    UsageInfo,
)
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..middleware.auth import authenticate_request

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):  # type: ignore[no-untyped-def]
    """OpenAI-compatible chat completions endpoint.

    In production, this forwards to inference-router via RabbitMQ.
    For MVP, returns a structured response.
    """
    await authenticate_request(request)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if body.stream:
        return StreamingResponse(
            _stream_response(completion_id, created, body.model, body.messages),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    # Non-streaming response
    # In production, this dispatches to inference-router and waits for result
    response_text = _generate_placeholder_response(body.messages)

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=body.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=_count_tokens(body.messages),
            completion_tokens=len(response_text.split()),
            total_tokens=_count_tokens(body.messages) + len(response_text.split()),
        ),
    )


async def _stream_response(
    completion_id: str,
    created: int,
    model: str,
    messages: list[ChatMessage],
) -> AsyncIterator[str]:
    """Stream SSE chunks in OpenAI format."""
    response_text = _generate_placeholder_response(messages)
    words = response_text.split()

    for i, word in enumerate(words):
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": word + " "},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


def _generate_placeholder_response(messages: list[ChatMessage]) -> str:
    """Placeholder response until inference worker is connected."""
    last_msg = messages[-1].content if messages else ""
    return (
        f"[Dana AI - Model Loading] Received your message: '{last_msg[:50]}...'. "
        "The inference worker is being configured. Once connected to the GPU cluster, "
        "this endpoint will return real model responses via Qwen3-235B-MoE with "
        "speculative decoding."
    )


def _count_tokens(messages: list[ChatMessage]) -> int:
    """Approximate token count. In production, uses tiktoken."""
    return sum(len(m.content.split()) * 4 // 3 for m in messages)
