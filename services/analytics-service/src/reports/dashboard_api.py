"""API endpoints for dashboard metrics."""
from __future__ import annotations

from dana_common.logging import get_logger
from fastapi import APIRouter, Query

from ..pipelines.usage_pipeline import RollupWindow, UsagePipeline

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/dashboard", tags=["dashboard"])

# Shared pipeline instance -- injected from main at startup
_pipeline: UsagePipeline | None = None


def set_pipeline(pipeline: UsagePipeline) -> None:
    global _pipeline
    _pipeline = pipeline


def _get_pipeline() -> UsagePipeline:
    if _pipeline is None:
        raise RuntimeError("UsagePipeline not initialised")
    return _pipeline


@router.get("/usage/{tenant_id}")
async def dashboard_usage(
    tenant_id: str,
    window: str = Query("hour", description="Rollup window: minute, hour, day, month"),
    model: str | None = Query(None, description="Optional model filter"),
) -> dict:
    pipeline = _get_pipeline()
    try:
        rw = RollupWindow(window)
    except ValueError:
        return {"error": f"Invalid window: {window}"}

    buckets = pipeline.query(tenant_id, rw, model=model)
    return {
        "tenant_id": tenant_id,
        "window": window,
        "buckets": [b.to_dict() for b in buckets],
    }


@router.get("/summary/{tenant_id}")
async def dashboard_summary(tenant_id: str) -> dict:
    pipeline = _get_pipeline()
    daily = pipeline.query(tenant_id, RollupWindow.DAY)
    total_tokens = sum(b.total_tokens for b in daily)
    total_requests = sum(b.total_requests for b in daily)
    avg_latency = (
        sum(b.total_latency_ms for b in daily) / total_requests if total_requests else 0.0
    )
    return {
        "tenant_id": tenant_id,
        "total_tokens": total_tokens,
        "total_requests": total_requests,
        "avg_latency_ms": round(avg_latency, 2),
        "days_tracked": len(daily),
    }
