"""Tests for inference router load balancer and priority queue logic."""

from __future__ import annotations

import time

import pytest
from dana_common.models import (
    ChatCompletionRequest,
    ChatMessage,
    InferenceJob,
    UserTier,
)
from router.load_balancer import GPUWorker, LoadBalancer
from router.priority_queue import (
    STARVATION_THRESHOLD_S,
    PendingJob,
    PriorityLevel,
    PriorityScheduler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_worker(
    worker_id: str = "w1",
    weight: int = 1,
    healthy: bool = True,
    active_requests: int = 0,
    max_concurrent: int = 16,
) -> GPUWorker:
    return GPUWorker(
        worker_id=worker_id,
        host="127.0.0.1",
        port=50051,
        weight=weight,
        healthy=healthy,
        active_requests=active_requests,
        max_concurrent=max_concurrent,
        last_heartbeat=time.time(),
    )


def _make_job(
    job_id: str = "job-1",
    user_tier: UserTier = UserTier.FREE,
) -> InferenceJob:
    return InferenceJob(
        job_id=job_id,
        api_key_id=1,
        user_id=1,
        user_tier=user_tier,
        request=ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hello")],
        ),
    )


# ---------------------------------------------------------------------------
# LoadBalancer tests
# ---------------------------------------------------------------------------

class TestLoadBalancer:
    def test_register_and_select(self) -> None:
        lb = LoadBalancer()
        w = _make_worker("w1")
        lb.register(w)

        assert lb.worker_count == 1
        selected = lb.select_worker()
        assert selected is not None
        assert selected.worker_id == "w1"

    def test_select_returns_none_when_empty(self) -> None:
        lb = LoadBalancer()
        assert lb.select_worker() is None

    def test_unhealthy_worker_skipped(self) -> None:
        lb = LoadBalancer()
        lb.register(_make_worker("w1", healthy=False))
        lb.register(_make_worker("w2", healthy=True))

        selected = lb.select_worker()
        assert selected is not None
        assert selected.worker_id == "w2"

    def test_deregister(self) -> None:
        lb = LoadBalancer()
        lb.register(_make_worker("w1"))
        lb.deregister("w1")
        assert lb.worker_count == 0

    def test_report_failure_marks_unhealthy(self) -> None:
        lb = LoadBalancer()
        lb.register(_make_worker("w1"))
        for _ in range(LoadBalancer.MAX_CONSECUTIVE_FAILURES):
            lb.report_failure("w1")
        assert lb.healthy_count == 0

    def test_heartbeat_restores_health(self) -> None:
        lb = LoadBalancer()
        w = _make_worker("w1", healthy=False)
        w.consecutive_failures = 5
        lb.register(w)
        lb.heartbeat("w1")
        assert lb.healthy_count == 1

    def test_prefers_less_loaded_worker(self) -> None:
        lb = LoadBalancer()
        lb.register(_make_worker("w1", active_requests=15, max_concurrent=16))
        lb.register(_make_worker("w2", active_requests=1, max_concurrent=16))

        selected = lb.select_worker()
        assert selected is not None
        # w2 should have higher effective weight (less loaded)
        assert selected.worker_id == "w2"

    def test_load_score_calculation(self) -> None:
        w = _make_worker(active_requests=8, max_concurrent=16)
        w.gpu_memory_total_mb = 24000
        w.gpu_memory_used_mb = 12000
        # 0.6 * (8/16) + 0.4 * (12000/24000) = 0.6*0.5 + 0.4*0.5 = 0.5
        assert abs(w.load_score - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# PriorityScheduler tests
# ---------------------------------------------------------------------------

class TestPriorityScheduler:
    def test_enterprise_higher_priority_than_free(self) -> None:
        scheduler = PriorityScheduler()
        free_job = _make_job("free-1", UserTier.FREE)
        ent_job = _make_job("ent-1", UserTier.ENTERPRISE)

        free_priority = scheduler.compute_priority(free_job)
        ent_priority = scheduler.compute_priority(ent_job)

        assert ent_priority > free_priority

    def test_pro_higher_priority_than_free(self) -> None:
        scheduler = PriorityScheduler()
        assert scheduler.compute_priority(_make_job("p", UserTier.PRO)) > \
               scheduler.compute_priority(_make_job("f", UserTier.FREE))

    def test_enqueue_dequeue_order(self) -> None:
        scheduler = PriorityScheduler()
        scheduler.enqueue(_make_job("free-1", UserTier.FREE))
        scheduler.enqueue(_make_job("ent-1", UserTier.ENTERPRISE))

        # Enterprise job should come out first
        first = scheduler.dequeue()
        assert first is not None
        assert first.job.job_id == "ent-1"

        second = scheduler.dequeue()
        assert second is not None
        assert second.job.job_id == "free-1"

    def test_dequeue_empty_returns_none(self) -> None:
        scheduler = PriorityScheduler()
        assert scheduler.dequeue() is None

    def test_pending_count(self) -> None:
        scheduler = PriorityScheduler()
        scheduler.enqueue(_make_job("j1"))
        scheduler.enqueue(_make_job("j2"))
        assert scheduler.pending_count == 2
        scheduler.dequeue()
        assert scheduler.pending_count == 1

    def test_max_capacity_raises(self) -> None:
        scheduler = PriorityScheduler(max_pending=2)
        scheduler.enqueue(_make_job("j1"))
        scheduler.enqueue(_make_job("j2"))
        with pytest.raises(RuntimeError, match="capacity"):
            scheduler.enqueue(_make_job("j3"))

    def test_starvation_boost(self) -> None:
        """Free-tier jobs get priority boost after waiting."""
        pending = PendingJob(
            job=_make_job("old-free", UserTier.FREE),
            enqueued_at=time.time() - STARVATION_THRESHOLD_S * 2.5,
            base_priority=PriorityLevel.FREE,
        )
        # Should receive a +2 age boost (capped at MAX_AGE_BOOST=2)
        assert pending.effective_priority == PriorityLevel.FREE + 2

    def test_stats(self) -> None:
        scheduler = PriorityScheduler()
        scheduler.enqueue(_make_job("j1", UserTier.PRO))
        scheduler.dequeue()
        stats = scheduler.stats
        assert stats["total_enqueued"] == 1
        assert stats["total_dispatched"] == 1
        assert stats["pending"] == 0
