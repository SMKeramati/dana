"""Custom output quality tracking.

Daneshbonyan: Internal Design & Development

Tracks response quality metrics including response length distribution,
token diversity, repetition detection, and coherence scoring.  Provides
a sliding-window view of quality over time so degradation can be detected
and alerted on.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualitySample:
    """Quality metrics for a single generated response."""

    request_id: str
    timestamp: float
    num_tokens: int
    unique_tokens: int
    repetition_score: float  # 0 = no repetition, 1 = fully repetitive
    avg_token_entropy: float  # higher = more diverse token usage
    coherence_score: float  # 0..1 heuristic coherence
    latency_s: float = 0.0


@dataclass
class QualitySnapshot:
    """Aggregated quality over a window of samples."""

    window_size: int
    avg_response_length: float
    std_response_length: float
    avg_repetition: float
    avg_coherence: float
    avg_token_entropy: float
    avg_latency_s: float
    degradation_detected: bool = False


class QualityMonitor:
    """Monitors output quality over a sliding window of responses.

    Daneshbonyan: Internal Design & Development

    Parameters
    ----------
    window_size : int
        Number of recent samples to keep for aggregate statistics.
    repetition_threshold : float
        If average repetition score exceeds this, flag degradation.
    coherence_threshold : float
        If average coherence drops below this, flag degradation.
    """

    def __init__(
        self,
        window_size: int = 200,
        repetition_threshold: float = 0.5,
        coherence_threshold: float = 0.3,
    ) -> None:
        self._window_size = window_size
        self._rep_threshold = repetition_threshold
        self._coh_threshold = coherence_threshold
        self._samples: deque[QualitySample] = deque(maxlen=window_size)
        self._total_samples: int = 0

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_repetition_score(token_ids: list[int] | np.ndarray) -> float:
        """Measure repetition via n-gram overlap.

        Computes the fraction of bigrams that are repeated at least once.
        """
        if len(token_ids) < 3:
            return 0.0

        bigrams: list[tuple[int, int]] = []
        for i in range(len(token_ids) - 1):
            bigrams.append((int(token_ids[i]), int(token_ids[i + 1])))

        if not bigrams:
            return 0.0

        unique = len(set(bigrams))
        total = len(bigrams)
        # Repetition ratio: 1 - (unique / total).  0 = all unique, 1 = all same
        return 1.0 - (unique / total)

    @staticmethod
    def compute_token_entropy(token_ids: list[int] | np.ndarray) -> float:
        """Shannon entropy of the token distribution in a response."""
        if len(token_ids) == 0:
            return 0.0

        freq: dict[int, int] = {}
        for t in token_ids:
            t = int(t)
            freq[t] = freq.get(t, 0) + 1

        n = len(token_ids)
        entropy = 0.0
        for count in freq.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def compute_coherence_score(token_ids: list[int] | np.ndarray) -> float:
        """Heuristic coherence score based on local token variation.

        Measures how often consecutive tokens differ -- monotonic or
        stuck sequences score low.  This is a lightweight proxy; real
        coherence would use embeddings.
        """
        if len(token_ids) < 2:
            return 1.0

        transitions = 0
        for i in range(len(token_ids) - 1):
            if int(token_ids[i]) != int(token_ids[i + 1]):
                transitions += 1

        transition_rate = transitions / (len(token_ids) - 1)

        # Penalise very low AND very high transition rates
        # (all-same is incoherent; fully random may also be incoherent)
        # Sweet spot around 0.5-0.9
        if transition_rate < 0.1:
            return transition_rate * 5.0  # scale up low values
        if transition_rate > 0.95:
            return max(0.0, 1.0 - (transition_rate - 0.95) * 10.0)
        return min(1.0, transition_rate * 1.1)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        request_id: str,
        token_ids: list[int] | np.ndarray,
        latency_s: float = 0.0,
    ) -> QualitySample:
        """Record quality metrics for a generated response."""
        arr = np.asarray(token_ids, dtype=np.int32)

        sample = QualitySample(
            request_id=request_id,
            timestamp=time.monotonic(),
            num_tokens=len(arr),
            unique_tokens=len(set(arr.tolist())),
            repetition_score=self.compute_repetition_score(arr),
            avg_token_entropy=self.compute_token_entropy(arr),
            coherence_score=self.compute_coherence_score(arr),
            latency_s=latency_s,
        )

        self._samples.append(sample)
        self._total_samples += 1
        return sample

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def snapshot(self) -> QualitySnapshot:
        """Compute aggregate quality over the current window."""
        if not self._samples:
            return QualitySnapshot(
                window_size=0,
                avg_response_length=0.0,
                std_response_length=0.0,
                avg_repetition=0.0,
                avg_coherence=1.0,
                avg_token_entropy=0.0,
                avg_latency_s=0.0,
            )

        lengths = np.array([s.num_tokens for s in self._samples], dtype=np.float64)
        reps = np.array([s.repetition_score for s in self._samples])
        cohs = np.array([s.coherence_score for s in self._samples])
        ents = np.array([s.avg_token_entropy for s in self._samples])
        lats = np.array([s.latency_s for s in self._samples])

        avg_rep = float(np.mean(reps))
        avg_coh = float(np.mean(cohs))

        degraded = avg_rep > self._rep_threshold or avg_coh < self._coh_threshold

        if degraded:
            logger.warning(
                "Quality degradation detected: avg_repetition=%.3f, avg_coherence=%.3f",
                avg_rep,
                avg_coh,
            )

        return QualitySnapshot(
            window_size=len(self._samples),
            avg_response_length=float(np.mean(lengths)),
            std_response_length=float(np.std(lengths)),
            avg_repetition=avg_rep,
            avg_coherence=avg_coh,
            avg_token_entropy=float(np.mean(ents)),
            avg_latency_s=float(np.mean(lats)),
            degradation_detected=degraded,
        )

    @property
    def total_samples(self) -> int:
        return self._total_samples

    def clear(self) -> None:
        self._samples.clear()
        self._total_samples = 0
