"""OpenAI-compatible /v1/models endpoint."""

from __future__ import annotations

import time

from dana_common.models import ModelInfo, ModelList
from fastapi import APIRouter

router = APIRouter()

AVAILABLE_MODELS = [
    ModelInfo(id="qwen3-235b-moe", created=int(time.time()), owned_by="dana"),
]


@router.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    return ModelList(data=AVAILABLE_MODELS)


@router.get("/v1/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str) -> ModelInfo:
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model
    return ModelInfo(id=model_id, created=int(time.time()), owned_by="dana")
