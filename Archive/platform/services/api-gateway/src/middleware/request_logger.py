"""Request logging middleware with correlation ID propagation."""

from __future__ import annotations

import time
import uuid

from dana_common.logging import set_correlation_id, setup_logging
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = setup_logging("api-gateway")


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Extract or generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID", uuid.uuid4().hex[:16])
        set_correlation_id(correlation_id)

        start = time.monotonic()
        logger.info(
            "Request started: %s %s",
            request.method,
            request.url.path,
        )

        response = await call_next(request)

        duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Request completed: %s %s status=%d duration=%.1fms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Response-Time"] = f"{duration_ms:.1f}ms"
        return response
