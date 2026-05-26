"""CUSTOM anomaly detection using Z-score + EWMA (Exponentially Weighted Moving Average).

Daneshbonyan: Internal R&D, Score Booster -- Real-time anomaly detection
engine that combines classical Z-score thresholding with EWMA-smoothed
baselines to identify unusual usage patterns (latency spikes, token surges,
cost anomalies) with low false-positive rates.  The dual-signal approach
(static Z-score AND adaptive EWMA) provides robustness against both sudden
outliers and gradual distribution shifts.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from dana_common.logging import get_logger

logger = get_logger(__name__)


class AnomalyLevel(StrEnum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class AnomalyResult:
    """Result of anomaly evaluation for a single data point."""

    value: float
    z_score: float
    ewma_deviation: float
    level: AnomalyLevel
    threshold_z: float
    threshold_ewma: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "z_score": round(self.z_score, 4),
            "ewma_deviation": round(self.ewma_deviation, 4),
            "level": self.level.value,
        }


@dataclass
class _EWMAState:
    """Internal EWMA accumulator state."""

    mean: float = 0.0
    variance: float = 0.0
    initialised: bool = False


class AnomalyDetector:
    """Dual-signal anomaly detector combining Z-score and EWMA.

    **Z-score detection** uses a sliding window of the last *window_size*
    observations to compute a running mean and standard deviation.  Any new
    value whose absolute Z-score exceeds *z_threshold* is flagged.

    **EWMA detection** maintains an exponentially weighted moving average
    (controlled by *alpha*) and an exponentially weighted moving variance.
    A point is flagged when its deviation from the EWMA mean exceeds
    *ewma_threshold* standard deviations of the EWMA variance.

    A point is classified as:
    - CRITICAL when *both* signals fire,
    - WARNING  when *either* signal fires,
    - NORMAL   otherwise.
    """

    def __init__(
        self,
        window_size: int = 100,
        z_threshold: float = 3.0,
        ewma_alpha: float = 0.3,
        ewma_threshold: float = 2.5,
    ) -> None:
        self._window_size = window_size
        self._z_threshold = z_threshold
        self._alpha = ewma_alpha
        self._ewma_threshold = ewma_threshold

        # Sliding window for Z-score
        self._window: list[float] = []

        # EWMA state
        self._ewma = _EWMAState()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(self, value: float) -> AnomalyResult:
        """Feed a new observation and return the anomaly assessment."""
        z_score = self._update_z_score(value)
        ewma_dev = self._update_ewma(value)

        z_fired = abs(z_score) > self._z_threshold
        ewma_fired = abs(ewma_dev) > self._ewma_threshold

        if z_fired and ewma_fired:
            level = AnomalyLevel.CRITICAL
        elif z_fired or ewma_fired:
            level = AnomalyLevel.WARNING
        else:
            level = AnomalyLevel.NORMAL

        result = AnomalyResult(
            value=value,
            z_score=z_score,
            ewma_deviation=ewma_dev,
            level=level,
            threshold_z=self._z_threshold,
            threshold_ewma=self._ewma_threshold,
        )

        if level is not AnomalyLevel.NORMAL:
            logger.warning(
                "anomaly_detected",
                value=value,
                z_score=round(z_score, 4),
                ewma_dev=round(ewma_dev, 4),
                level=level.value,
            )
        return result

    def observe_batch(self, values: list[float] | np.ndarray) -> list[AnomalyResult]:
        """Feed multiple observations and return per-element results."""
        return [self.observe(float(v)) for v in values]

    @property
    def current_ewma_mean(self) -> float:
        return self._ewma.mean

    @property
    def current_ewma_std(self) -> float:
        return max(np.sqrt(self._ewma.variance), 1e-9)

    @property
    def window_mean(self) -> float:
        if not self._window:
            return 0.0
        return float(np.mean(self._window))

    @property
    def window_std(self) -> float:
        if len(self._window) < 2:
            return 0.0
        return float(np.std(self._window, ddof=1))

    # ------------------------------------------------------------------
    # Z-score internals
    # ------------------------------------------------------------------

    def _update_z_score(self, value: float) -> float:
        """Append *value* to the sliding window and compute its Z-score.

        If the window contains fewer than 2 elements the Z-score is 0.
        """
        self._window.append(value)
        if len(self._window) > self._window_size:
            self._window.pop(0)

        if len(self._window) < 2:
            return 0.0

        arr = np.array(self._window)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        if std < 1e-9:
            return 0.0
        return (value - mean) / std

    # ------------------------------------------------------------------
    # EWMA internals
    # ------------------------------------------------------------------

    def _update_ewma(self, value: float) -> float:
        """Update the EWMA mean and variance, and return the deviation in
        units of EWMA standard deviation.

        The update equations are:
            ewma_mean  <- alpha * value + (1 - alpha) * ewma_mean
            ewma_var   <- alpha * (value - ewma_mean)^2 + (1 - alpha) * ewma_var

        The deviation is (value - ewma_mean) / sqrt(ewma_var).
        """
        alpha = self._alpha
        state = self._ewma

        if not state.initialised:
            state.mean = value
            state.variance = 0.0
            state.initialised = True
            return 0.0

        diff = value - state.mean
        state.mean = alpha * value + (1.0 - alpha) * state.mean
        state.variance = alpha * (diff ** 2) + (1.0 - alpha) * state.variance

        std = max(np.sqrt(state.variance), 1e-9)
        return diff / std
