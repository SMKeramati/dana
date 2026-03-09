"""ExpertClassifier — hot/warm/cold classification of experts.

Classifies experts into three tiers based on activation frequency:
  hot:  activated frequently — pin in VRAM (FP16/Q8)
  warm: activated sometimes — keep in RAM (Q4)
  cold: activated rarely/never — store on SSD (Q2)

Thresholds are auto-tunable via CacheAnalytics.suggest_thresholds().
"""

from __future__ import annotations

from typing import Literal

from expert_cache.frequency_cache import FrequencyExpertCache


ExpertTier = Literal["hot", "warm", "cold"]


class ExpertClassifier:
    """Classify experts by activation frequency.

    Usage:
        clf = ExpertClassifier(hot_threshold=10, warm_threshold=1)
        tier = clf.classify(expert_id=3, cache=freq_cache)
        # → "hot", "warm", or "cold"
    """

    def __init__(
        self,
        hot_threshold: float = 10.0,
        warm_threshold: float = 1.0,
    ) -> None:
        """
        Args:
            hot_threshold: Min frequency count to be classified "hot"
            warm_threshold: Min frequency count to be classified "warm"
        """
        self.hot_threshold = hot_threshold
        self.warm_threshold = warm_threshold

    def classify(self, expert_id: int, cache: FrequencyExpertCache) -> ExpertTier:
        """Classify a single expert by its sliding-window frequency."""
        freq = cache.frequency(expert_id)
        if freq >= self.hot_threshold:
            return "hot"
        elif freq >= self.warm_threshold:
            return "warm"
        else:
            return "cold"

    def classify_all(self, cache: FrequencyExpertCache) -> dict[int, ExpertTier]:
        """Classify all tracked experts.

        Returns:
            {expert_id: "hot" | "warm" | "cold"}
        """
        result: dict[int, ExpertTier] = {}
        for expert_id, freq in cache.all_frequencies().items():
            if freq >= self.hot_threshold:
                result[expert_id] = "hot"
            elif freq >= self.warm_threshold:
                result[expert_id] = "warm"
            else:
                result[expert_id] = "cold"
        return result

    def update_thresholds(
        self,
        hot_threshold: float,
        warm_threshold: float,
    ) -> None:
        """Update classification thresholds (called by analytics auto-tuning)."""
        assert warm_threshold <= hot_threshold, \
            "warm_threshold must be <= hot_threshold"
        self.hot_threshold = hot_threshold
        self.warm_threshold = warm_threshold
