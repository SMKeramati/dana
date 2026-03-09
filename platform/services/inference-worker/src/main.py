"""Dana inference-worker FastAPI application.

Exposes health-check and GPU status endpoints on port 8006.
"""

from __future__ import annotations

import time

from fastapi import FastAPI

from src.health.gpu_monitor import GPUMonitor

app = FastAPI(
    title="Dana Inference Worker",
    version="0.1.0",
    description="Core AI inference service with speculative decoding, MoE offloading, and KV cache management.",
)

_gpu_monitor = GPUMonitor(num_gpus=1, simulate=True)
_start_time = time.monotonic()


@app.get("/health")
async def health() -> dict[str, object]:
    """Liveness / readiness probe."""
    gpu_summary = _gpu_monitor.collect()
    return {
        "status": "healthy" if gpu_summary.all_healthy else "degraded",
        "uptime_s": round(time.monotonic() - _start_time, 2),
        "gpu_healthy": gpu_summary.all_healthy,
    }


@app.get("/gpu/status")
async def gpu_status() -> dict[str, object]:
    """Detailed GPU utilization and health metrics."""
    summary = _gpu_monitor.collect()
    per_gpu = []
    for m in summary.per_gpu:
        per_gpu.append({
            "gpu_id": m.gpu_id,
            "utilization_pct": round(m.utilization_pct, 1),
            "memory_used_mb": round(m.memory_used_bytes / (1024 * 1024), 1),
            "memory_total_mb": round(m.memory_total_bytes / (1024 * 1024), 1),
            "memory_utilization_pct": round(m.memory_utilization_pct, 1),
            "temperature_c": round(m.temperature_c, 1),
            "power_draw_w": round(m.power_draw_w, 1),
            "healthy": m.is_healthy,
        })
    return {
        "num_gpus": summary.num_gpus,
        "all_healthy": summary.all_healthy,
        "avg_utilization_pct": round(summary.avg_utilization_pct, 1),
        "avg_temperature_c": round(summary.avg_temperature_c, 1),
        "gpus": per_gpu,
    }
