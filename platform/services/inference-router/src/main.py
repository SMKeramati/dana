"""Dana Inference Router Service - Routes inference requests to GPU workers.

Daneshbonyan: Internal Design & Development - Central routing service that
accepts inference requests, prioritises them by user tier, dispatches to
GPU workers via RabbitMQ, and streams results back to clients via SSE.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from dana_common.logging import setup_logging
from dana_common.models import HealthResponse
from fastapi import FastAPI

from .metrics.latency_tracker import LatencyTracker
from .router.load_balancer import LoadBalancer
from .router.priority_queue import PriorityScheduler
from .router.queue_manager import QueueManager
from .streaming.sse_bridge import SSEBridge

logger = setup_logging("inference-router")

# Service-wide singletons
load_balancer = LoadBalancer()
queue_manager = QueueManager()
priority_scheduler = PriorityScheduler()
sse_bridge = SSEBridge()
latency_tracker = LatencyTracker()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: connect to RabbitMQ and begin consuming results."""
    await queue_manager.connect()
    await queue_manager.start_result_consumer()
    logger.info("Inference router started")
    yield
    await queue_manager.close()
    logger.info("Inference router shut down")


app = FastAPI(
    title="Dana Inference Router",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        service="inference-router",
    )


@app.get("/health/detailed")
async def health_detailed() -> dict[str, Any]:
    """Extended health check with worker and queue stats."""
    return {
        "status": "ok",
        "service": "inference-router",
        "version": "0.1.0",
        "workers": {
            "total": load_balancer.worker_count,
            "healthy": load_balancer.healthy_count,
        },
        "scheduler": priority_scheduler.stats,
        "streams": {
            "active": sse_bridge.active_streams,
        },
        "latency": latency_tracker.all_metrics(),
    }
