"""Custom health probing with failure thresholds.

Periodically probes model endpoints and tracks consecutive failures to decide
when a model version should be marked unhealthy and removed from serving.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx
from dana_common.logging import get_logger

logger = get_logger(__name__)


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ProbeResult:
    endpoint_url: str
    status: HealthStatus
    latency_ms: float
    consecutive_failures: int
    last_check: float  # monotonic timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_url": self.endpoint_url,
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 2),
            "consecutive_failures": self.consecutive_failures,
        }


@dataclass
class _EndpointState:
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_latency_ms: float = 0.0
    last_check: float = 0.0
    status: HealthStatus = HealthStatus.UNKNOWN


class HealthChecker:
    """Probes model endpoints and applies failure-threshold logic.

    Parameters
    ----------
    failure_threshold:
        Number of consecutive failures before marking an endpoint UNHEALTHY.
    degraded_threshold:
        Number of consecutive failures before marking DEGRADED.
    recovery_threshold:
        Number of consecutive successes to transition from UNHEALTHY back to
        HEALTHY.
    timeout_seconds:
        HTTP probe timeout.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        degraded_threshold: int = 1,
        recovery_threshold: int = 2,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._degraded_threshold = degraded_threshold
        self._recovery_threshold = recovery_threshold
        self._timeout = timeout_seconds
        self._states: dict[str, _EndpointState] = {}

    async def probe(self, endpoint_url: str) -> ProbeResult:
        """Send an HTTP GET to *endpoint_url*/health and update state."""
        state = self._states.setdefault(endpoint_url, _EndpointState())

        url = endpoint_url.rstrip("/") + "/health"
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url)
            latency_ms = (time.monotonic() - start) * 1000
            if resp.status_code == 200:
                state.consecutive_failures = 0
                state.consecutive_successes += 1
            else:
                state.consecutive_failures += 1
                state.consecutive_successes = 0
        except Exception:
            latency_ms = (time.monotonic() - start) * 1000
            state.consecutive_failures += 1
            state.consecutive_successes = 0

        state.last_latency_ms = latency_ms
        state.last_check = time.monotonic()

        # Determine status
        if state.consecutive_failures >= self._failure_threshold:
            state.status = HealthStatus.UNHEALTHY
        elif state.consecutive_failures >= self._degraded_threshold:
            state.status = HealthStatus.DEGRADED
        elif state.consecutive_successes >= self._recovery_threshold:
            state.status = HealthStatus.HEALTHY
        # else keep current status

        result = ProbeResult(
            endpoint_url=endpoint_url,
            status=state.status,
            latency_ms=latency_ms,
            consecutive_failures=state.consecutive_failures,
            last_check=state.last_check,
        )

        if state.status in (HealthStatus.UNHEALTHY, HealthStatus.DEGRADED):
            logger.warning("health_probe_failed", endpoint=endpoint_url, status=state.status.value,
                           failures=state.consecutive_failures)
        return result

    def record_success(self, endpoint_url: str) -> HealthStatus:
        """Manually record a successful probe (for testing / synthetic probes)."""
        state = self._states.setdefault(endpoint_url, _EndpointState())
        state.consecutive_failures = 0
        state.consecutive_successes += 1
        state.last_check = time.monotonic()
        if state.consecutive_successes >= self._recovery_threshold:
            state.status = HealthStatus.HEALTHY
        return state.status

    def record_failure(self, endpoint_url: str) -> HealthStatus:
        """Manually record a failed probe."""
        state = self._states.setdefault(endpoint_url, _EndpointState())
        state.consecutive_failures += 1
        state.consecutive_successes = 0
        state.last_check = time.monotonic()
        if state.consecutive_failures >= self._failure_threshold:
            state.status = HealthStatus.UNHEALTHY
        elif state.consecutive_failures >= self._degraded_threshold:
            state.status = HealthStatus.DEGRADED
        return state.status

    def get_status(self, endpoint_url: str) -> HealthStatus:
        state = self._states.get(endpoint_url)
        return state.status if state else HealthStatus.UNKNOWN

    def get_all_statuses(self) -> dict[str, HealthStatus]:
        return {url: s.status for url, s in self._states.items()}
