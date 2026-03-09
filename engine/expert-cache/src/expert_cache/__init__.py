"""Expert Cache — intelligent expert caching for MoE inference.

Provides LRU, frequency-aware, and predictive caching strategies with
VRAM budget enforcement and hit/miss analytics.
"""

from expert_cache.lru_cache import LRUExpertCache
from expert_cache.frequency_cache import FrequencyExpertCache
from expert_cache.predictive_cache import PredictiveExpertCache
from expert_cache.classifier import ExpertClassifier
from expert_cache.budget_manager import VRAMBudgetManager
from expert_cache.analytics import CacheAnalytics

__all__ = [
    "LRUExpertCache",
    "FrequencyExpertCache",
    "PredictiveExpertCache",
    "ExpertClassifier",
    "VRAMBudgetManager",
    "CacheAnalytics",
]

__version__ = "0.1.0"
