"""Model registry service -- FastAPI application (port 8005)."""
from __future__ import annotations

from typing import Any

from dana_common.logging import get_logger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .registry.ab_testing import ABTestRouter
from .registry.model_store import ModelStore

logger = get_logger(__name__)

app = FastAPI(title="Dana Model Registry", version="0.1.0")

_store = ModelStore()
_ab_router = ABTestRouter()


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------

class RegisterRequest(BaseModel):
    model_name: str
    version: str
    endpoint_url: str
    metadata: dict[str, Any] | None = None


class PromoteRequest(BaseModel):
    model_name: str
    version: str


class ABTestCreateRequest(BaseModel):
    experiment_id: str
    model_name: str
    variants: list[dict[str, Any]]
    confidence_level: float = 0.95


class ABTestOutcomeRequest(BaseModel):
    experiment_id: str
    variant_name: str
    success: bool


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "model-registry"}


@app.post("/v1/models/register")
async def register_model(req: RegisterRequest) -> dict[str, Any]:
    mv = _store.register(req.model_name, req.version, req.endpoint_url, req.metadata)
    return {"model_name": mv.model_name, "version": mv.version, "status": mv.status.value}


@app.post("/v1/models/promote")
async def promote_model(req: PromoteRequest) -> dict[str, Any]:
    try:
        mv = _store.promote(req.model_name, req.version)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"model_name": mv.model_name, "version": mv.version, "status": mv.status.value}


@app.post("/v1/models/{model_name}/rollback")
async def rollback_model(model_name: str) -> dict[str, Any]:
    try:
        mv = _store.rollback(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"model_name": mv.model_name, "version": mv.version, "status": mv.status.value}


@app.get("/v1/models/{model_name}/active")
async def get_active_model(model_name: str) -> dict[str, Any]:
    mv = _store.get_active(model_name)
    if mv is None:
        raise HTTPException(status_code=404, detail=f"No active version for {model_name}")
    return {
        "model_name": mv.model_name,
        "version": mv.version,
        "endpoint_url": mv.endpoint_url,
        "status": mv.status.value,
    }


@app.get("/v1/models/{model_name}/versions")
async def list_versions(model_name: str) -> list[dict[str, Any]]:
    return [
        {"version": mv.version, "status": mv.status.value, "endpoint_url": mv.endpoint_url}
        for mv in _store.list_versions(model_name)
    ]


@app.get("/v1/models")
async def list_models() -> list[str]:
    return _store.list_models()


# ------------------------------------------------------------------
# A/B Testing endpoints
# ------------------------------------------------------------------

@app.post("/v1/ab-tests")
async def create_ab_test(req: ABTestCreateRequest) -> dict[str, Any]:
    test = _ab_router.create_experiment(
        experiment_id=req.experiment_id,
        model_name=req.model_name,
        variants=req.variants,
        confidence_level=req.confidence_level,
    )
    return {"experiment_id": test.experiment_id, "status": test.status.value}


@app.post("/v1/ab-tests/route/{experiment_id}")
async def route_ab_test(experiment_id: str) -> dict[str, Any]:
    test = _ab_router.get_experiment(experiment_id)
    if test is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    variant = test.route()
    return {"variant": variant.name, "model_version": variant.model_version}


@app.post("/v1/ab-tests/outcome")
async def record_outcome(req: ABTestOutcomeRequest) -> dict[str, Any]:
    test = _ab_router.get_experiment(req.experiment_id)
    if test is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    try:
        test.record_outcome(req.variant_name, req.success)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    sig = test.check_significance()
    return {
        "experiment_id": test.experiment_id,
        "status": test.status.value,
        "significance": sig.to_dict() if sig and hasattr(sig, "to_dict") else None,
    }


@app.get("/v1/ab-tests/{experiment_id}")
async def get_ab_test(experiment_id: str) -> dict[str, Any]:
    test = _ab_router.get_experiment(experiment_id)
    if test is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {
        "experiment_id": test.experiment_id,
        "model_name": test.model_name,
        "status": test.status.value,
        "variants": [
            {"name": v.name, "model_version": v.model_version, "weight": v.weight,
             "successes": v.successes, "failures": v.failures, "success_rate": round(v.success_rate, 4)}
            for v in test.variants
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8005)
