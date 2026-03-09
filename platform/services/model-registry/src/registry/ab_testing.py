"""Custom A/B testing router with traffic splitting and statistical significance.

Daneshbonyan: Internal R&D -- Implements a weighted traffic splitter that
routes incoming requests across model variants according to configured
weights.  Accumulates per-variant success/failure counts and uses a two-
proportion Z-test to determine when the experiment has reached statistical
significance, enabling data-driven model promotion decisions.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from dana_common.logging import get_logger

logger = get_logger(__name__)


class ExperimentStatus(StrEnum):
    RUNNING = "running"
    SIGNIFICANT = "significant"
    STOPPED = "stopped"


@dataclass
class Variant:
    """A model variant participating in an A/B test."""

    name: str
    model_version: str
    weight: float  # traffic fraction, e.g. 0.5
    successes: int = 0
    failures: int = 0

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.successes / self.total


@dataclass
class SignificanceResult:
    """Result of a two-proportion Z-test."""

    z_statistic: float
    p_value: float
    significant: bool
    confidence_level: float
    control_rate: float
    treatment_rate: float


@dataclass
class ABTest:
    """An A/B test experiment between two or more model variants."""

    experiment_id: str
    model_name: str
    variants: list[Variant]
    status: ExperimentStatus = ExperimentStatus.RUNNING
    confidence_level: float = 0.95
    created_at: float = field(default_factory=time.time)

    def route(self) -> Variant:
        """Select a variant based on configured traffic weights.

        Uses weighted random selection.
        """
        if not self.variants:
            raise ValueError("No variants configured")
        total_weight = sum(v.weight for v in self.variants)
        r = random.random() * total_weight
        cumulative = 0.0
        for variant in self.variants:
            cumulative += variant.weight
            if r <= cumulative:
                return variant
        return self.variants[-1]

    def record_outcome(self, variant_name: str, success: bool) -> None:
        """Record a success or failure for the named variant."""
        for v in self.variants:
            if v.name == variant_name:
                if success:
                    v.successes += 1
                else:
                    v.failures += 1
                return
        raise ValueError(f"Unknown variant: {variant_name}")

    def check_significance(self) -> SignificanceResult | None:
        """Run a two-proportion Z-test between the first two variants.

        Returns None if there are fewer than 2 variants or insufficient data.

        The Z-statistic for two proportions p1, p2 with sample sizes n1, n2 is:

            p_pool = (x1 + x2) / (n1 + n2)
            z = (p1 - p2) / sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

        We compute the two-tailed p-value from the standard normal CDF.
        """
        if len(self.variants) < 2:
            return None
        control = self.variants[0]
        treatment = self.variants[1]

        n1 = control.total
        n2 = treatment.total

        if n1 < 10 or n2 < 10:
            return None

        p1 = control.success_rate
        p2 = treatment.success_rate

        x1 = control.successes
        x2 = treatment.successes

        p_pool = (x1 + x2) / (n1 + n2)

        if p_pool == 0.0 or p_pool == 1.0:
            return SignificanceResult(
                z_statistic=0.0,
                p_value=1.0,
                significant=False,
                confidence_level=self.confidence_level,
                control_rate=p1,
                treatment_rate=p2,
            )

        se = math.sqrt(p_pool * (1 - p_pool) * (1.0 / n1 + 1.0 / n2))
        z = (p1 - p2) / se if se > 0 else 0.0

        # Two-tailed p-value using the standard normal survival function
        p_value = 2.0 * _normal_sf(abs(z))

        alpha = 1.0 - self.confidence_level
        significant = p_value < alpha

        result = SignificanceResult(
            z_statistic=z,
            p_value=p_value,
            significant=significant,
            confidence_level=self.confidence_level,
            control_rate=p1,
            treatment_rate=p2,
        )

        if significant and self.status == ExperimentStatus.RUNNING:
            self.status = ExperimentStatus.SIGNIFICANT
            logger.info(
                "ab_test_significant",
                experiment_id=self.experiment_id,
                z=round(z, 4),
                p_value=round(p_value, 6),
            )
        return result


# ---------------------------------------------------------------------------
# Standard normal CDF / survival function (avoids scipy dependency at runtime)
# ---------------------------------------------------------------------------

def _normal_cdf(x: float) -> float:
    """Approximate the standard normal CDF using the complementary error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_sf(x: float) -> float:
    """Standard normal survival function: 1 - CDF(x)."""
    return 1.0 - _normal_cdf(x)


# ---------------------------------------------------------------------------
# Router convenience
# ---------------------------------------------------------------------------

class ABTestRouter:
    """Manages multiple concurrent A/B test experiments."""

    def __init__(self) -> None:
        self._experiments: dict[str, ABTest] = {}

    def create_experiment(
        self,
        experiment_id: str,
        model_name: str,
        variants: list[dict[str, Any]],
        confidence_level: float = 0.95,
    ) -> ABTest:
        test = ABTest(
            experiment_id=experiment_id,
            model_name=model_name,
            variants=[
                Variant(name=v["name"], model_version=v["model_version"], weight=v.get("weight", 0.5))
                for v in variants
            ],
            confidence_level=confidence_level,
        )
        self._experiments[experiment_id] = test
        logger.info("ab_test_created", experiment_id=experiment_id, model=model_name)
        return test

    def get_experiment(self, experiment_id: str) -> ABTest | None:
        return self._experiments.get(experiment_id)

    def stop_experiment(self, experiment_id: str) -> ABTest | None:
        test = self._experiments.get(experiment_id)
        if test:
            test.status = ExperimentStatus.STOPPED
        return test

    def list_experiments(self) -> list[ABTest]:
        return list(self._experiments.values())
