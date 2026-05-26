"""Analytics service -- FastAPI application (port 8004)."""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from dana_common.logging import get_logger
from fastapi import FastAPI

from .pipelines.anomaly_detector import AnomalyDetector
from .pipelines.cost_analyzer import CostAnalyzer
from .pipelines.usage_pipeline import UsagePipeline
from .reports.dashboard_api import router as dashboard_router
from .reports.dashboard_api import set_pipeline

logger = get_logger(__name__)

# Shared instances
_pipeline = UsagePipeline()
_anomaly_detector = AnomalyDetector()
_cost_analyzer = CostAnalyzer()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    set_pipeline(_pipeline)
    logger.info("analytics_service.started")
    yield
    logger.info("analytics_service.stopped")


app = FastAPI(title="Dana Analytics Service", version="0.1.0", lifespan=lifespan)
app.include_router(dashboard_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "analytics"}


@app.get("/v1/stats")
async def stats() -> dict[str, Any]:
    return {
        "events_ingested": _pipeline.events_ingested,
        "anomaly_detector_window_size": len(_anomaly_detector._window),
        "ewma_mean": round(_anomaly_detector.current_ewma_mean, 4),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)
