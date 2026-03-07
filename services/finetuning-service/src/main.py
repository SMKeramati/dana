"""Dana Fine-tuning Service - FastAPI application.

Daneshbonyan: Internal R&D - Custom Model Training Platform
"""
from __future__ import annotations

from typing import Any

from dana_common.models import HealthResponse
from fastapi import FastAPI

app = FastAPI(
    title="Dana Fine-tuning Service",
    description="LoRA/QLoRA fine-tuning for Dana AI models",
    version="0.1.0",
)


@app.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(service="finetuning-service")


@app.get("/api/v1/finetune/status")
async def finetune_status() -> dict[str, Any]:
    return {
        "service": "finetuning-service",
        "supported_methods": ["lora", "qlora"],
        "supported_formats": ["alpaca", "sharegpt", "jsonl", "openai-chat"],
        "max_seq_length": 4096,
        "supported_models": ["Qwen/Qwen3-235B-A22B", "Qwen/Qwen3-0.6B"],
    }
