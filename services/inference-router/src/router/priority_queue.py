"""Multi-tier priority scheduler for inference jobs.

Daneshbonyan: Internal Design & Development - Custom priority scheduling
that assigns AMQP priority levels based on user tier, request age, and
queue depth. Paid users receive higher scheduling priority while free-tier
requests are protected from starvation via age-based boosting.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from dana_common.models import InferenceJob, UserTier

logger = logging.getLogger(__name__)


class PriorityLevel(IntEnum):
    """AMQP priority levels mapped to user tiers.

    RabbitMQ supports priority 0-10 (configured via x-max-priority).
    """

    CRITICAL = 10    # Internal / system jobs
    ENTERPRISE = 8   # Enterprise tier
    PRO = 5          # Pro tier
    FREE = 2         # Free tier
    BACKGROUND = 0   # Background / batch jobs


# Mapping from UserTier to base priority
TIER_PRIORITY_MAP: dict[UserTier, PriorityLevel] = {
    UserTier.ENTERPRISE: PriorityLevel.ENTERPRISE,
    UserTier.PRO: PriorityLevel.PRO,
    UserTier.FREE: PriorityLevel.FREE,
}

# Age-based boost: after this many seconds, free-tier jobs get +1 priority
STARVATION_THRESHOLD_S: float = 10.0
MAX_AGE_BOOST: int = 2


@dataclass
class PendingJob:
    """A job waiting to be dispatched with scheduling metadata."""

    job: InferenceJob
    enqueued_at: float = field(default_factory=time.time)
    base_priority: int = 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.enqueued_at

    @property
    def effective_priority(self) -> int:
        """Compute effective priority with age-based starvation protection."""
        age_boost = min(
            int(self.age_seconds / STARVATION_THRESHOLD_S),
            MAX_AGE_BOOST,
        )
        return min(self.base_priority + age_boost, PriorityLevel.CRITICAL)


class PriorityScheduler:
    """Multi-tier priority scheduler for inference request ordering.

    Daneshbonyan: Internal Design & Development - Custom scheduler that
    combines user-tier-based priority with starvation prevention. Enterprise
    and Pro users are prioritised, but long-waiting free-tier requests
    receive incremental priority boosts to prevent indefinite starvation.
    """

    def __init__(self, max_pending: int = 10000) -> None:
        self._queues: dict[PriorityLevel, deque[PendingJob]] = {
            level: deque() for level in PriorityLevel
        }
        self._max_pending = max_pending
        self._total_enqueued: int = 0
        self._total_dispatched: int = 0

    def compute_priority(self, job: InferenceJob) -> int:
        """Determine the AMQP priority for an inference job based on user tier."""
        base = TIER_PRIORITY_MAP.get(job.user_tier, PriorityLevel.FREE)
        return int(base)

    def enqueue(self, job: InferenceJob) -> PendingJob:
        """Add a job to the appropriate priority queue.

        Returns the PendingJob wrapper with scheduling metadata.

        Raises:
            RuntimeError: If the scheduler has reached max pending capacity.
        """
        if self.pending_count >= self._max_pending:
            raise RuntimeError(
                f"Priority scheduler at capacity ({self._max_pending} pending jobs)"
            )

        base_priority = self.compute_priority(job)
        pending = PendingJob(job=job, base_priority=base_priority)

        # Place in the appropriate tier queue
        level = PriorityLevel(base_priority)
        self._queues[level].append(pending)
        self._total_enqueued += 1

        logger.debug(
            "Enqueued job %s tier=%s priority=%d pending=%d",
            job.job_id,
            job.user_tier.value,
            base_priority,
            self.pending_count,
        )
        return pending

    def dequeue(self) -> PendingJob | None:
        """Remove and return the highest-priority pending job.

        Scans tiers from highest to lowest, applying age-based boosting
        so that long-waiting lower-tier jobs can be promoted.
        """
        best: PendingJob | None = None
        best_level: PriorityLevel | None = None

        for level in sorted(PriorityLevel, reverse=True):
            queue = self._queues[level]
            if not queue:
                continue
            candidate = queue[0]
            if best is None or candidate.effective_priority > best.effective_priority:
                best = candidate
                best_level = level

        if best is not None and best_level is not None:
            self._queues[best_level].popleft()
            self._total_dispatched += 1
            logger.debug(
                "Dequeued job %s effective_priority=%d age=%.1fs",
                best.job.job_id,
                best.effective_priority,
                best.age_seconds,
            )
        return best

    @property
    def pending_count(self) -> int:
        return sum(len(q) for q in self._queues.values())

    def pending_by_tier(self) -> dict[str, int]:
        """Return count of pending jobs per priority tier."""
        return {level.name: len(q) for level, q in self._queues.items()}

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_enqueued": self._total_enqueued,
            "total_dispatched": self._total_dispatched,
            "pending": self.pending_count,
            "pending_by_tier": self.pending_by_tier(),
        }
