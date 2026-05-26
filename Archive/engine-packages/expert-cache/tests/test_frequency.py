"""Tests for FrequencyExpertCache and PredictiveExpertCache."""

import pytest
import torch
from expert_cache.frequency_cache import FrequencyExpertCache
from expert_cache.predictive_cache import PredictiveExpertCache


def t():
    return torch.randn(4, 4)


# ---------------------------------------------------------------------------
# FrequencyExpertCache
# ---------------------------------------------------------------------------

@pytest.fixture
def freq_cache():
    return FrequencyExpertCache(capacity=3, window_size=20)


def test_freq_get_hit_increments_frequency(freq_cache):
    freq_cache.put(0, t())
    assert freq_cache.frequency(0) == 0
    freq_cache.get(0)
    assert freq_cache.frequency(0) == 1
    freq_cache.get(0)
    assert freq_cache.frequency(0) == 2


def test_freq_miss_returns_none(freq_cache):
    assert freq_cache.get(99) is None


def test_freq_evicts_lowest_frequency(freq_cache):
    freq_cache.put(0, t())
    freq_cache.put(1, t())
    freq_cache.put(2, t())

    # Make expert 1 hot
    for _ in range(5):
        freq_cache.get(1)
    # Make expert 0 warm
    freq_cache.get(0)
    # expert 2 has freq=0

    # Insert 3 → should evict expert 2 (freq=0)
    evicted = freq_cache.put(3, t())
    assert evicted == 2


def test_freq_window_decays_counts():
    cache = FrequencyExpertCache(capacity=4, window_size=3)
    cache.put(0, t())
    # Access 0 three times — fills window
    cache.get(0)
    cache.get(0)
    cache.get(0)
    assert cache.frequency(0) == 3

    # Now access other experts to push 0's accesses out of window
    cache.put(1, t())
    cache.get(1)  # window shifts
    cache.put(2, t())
    cache.get(2)
    cache.put(3, t())
    cache.get(3)
    # expert 0 frequency should have decayed
    assert cache.frequency(0) < 3


def test_freq_retains_hot_expert_under_pressure():
    cache = FrequencyExpertCache(capacity=2, window_size=100)
    hot = t()
    cache.put(0, hot)
    cache.put(1, t())
    for _ in range(10):
        cache.get(0)  # make 0 hot

    # Fill cache — expert 1 (freq=0) should be evicted, not 0
    cache.put(2, t())
    assert cache.contains(0)
    assert not cache.contains(1)


# ---------------------------------------------------------------------------
# PredictiveExpertCache
# ---------------------------------------------------------------------------

@pytest.fixture
def pred_cache():
    return PredictiveExpertCache(capacity=4, window_size=20)


def test_predictive_hint_protects_from_eviction(pred_cache):
    pred_cache.put(0, t())
    pred_cache.put(1, t())
    pred_cache.put(2, t())
    pred_cache.put(3, t())

    # Hint that expert 0 will be needed soon
    pred_cache.hint([0])

    # Try to insert 4 → should evict non-hinted (1, 2, or 3), not 0
    pred_cache.put(4, t())
    assert pred_cache.contains(0)


def test_predictive_get_clears_hint(pred_cache):
    pred_cache.put(0, t())
    pred_cache.hint([0])
    assert pred_cache.is_hinted(0)
    pred_cache.get(0)
    assert not pred_cache.is_hinted(0)


def test_predictive_pending_hints_excludes_cached(pred_cache):
    pred_cache.put(0, t())
    pred_cache.hint([0, 5, 6])
    pending = pred_cache.pending_hints()
    # 0 is already cached, so only 5 and 6 should be pending
    assert 0 not in pending
    assert 5 in pending
    assert 6 in pending


def test_predictive_put_updates_existing(pred_cache):
    t1 = torch.ones(2)
    t2 = torch.zeros(2)
    pred_cache.put(0, t1)
    pred_cache.put(0, t2)
    loaded = pred_cache.get(0)
    assert loaded is not None
    assert torch.allclose(loaded, t2)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def test_classifier_hot_warm_cold():
    from expert_cache.classifier import ExpertClassifier
    cache = FrequencyExpertCache(capacity=10, window_size=100)
    cache.put(0, t())
    cache.put(1, t())
    cache.put(2, t())

    for _ in range(15):
        cache.get(0)   # hot
    for _ in range(3):
        cache.get(1)   # warm
    # expert 2 freq = 0 → cold

    clf = ExpertClassifier(hot_threshold=10.0, warm_threshold=1.0)
    assert clf.classify(0, cache) == "hot"
    assert clf.classify(1, cache) == "warm"
    assert clf.classify(2, cache) == "cold"


def test_classifier_classify_all():
    from expert_cache.classifier import ExpertClassifier
    cache = FrequencyExpertCache(capacity=5, window_size=100)
    cache.put(0, t())
    cache.put(1, t())
    for _ in range(12):
        cache.get(0)
    for _ in range(2):
        cache.get(1)

    clf = ExpertClassifier(hot_threshold=10.0, warm_threshold=1.0)
    result = clf.classify_all(cache)
    assert result[0] == "hot"
    assert result[1] == "warm"


# ---------------------------------------------------------------------------
# BudgetManager
# ---------------------------------------------------------------------------

def test_budget_can_fit():
    from expert_cache.budget_manager import VRAMBudgetManager
    budget = VRAMBudgetManager(budget_bytes=1000)
    small = torch.zeros(10)  # 40 bytes (float32)
    assert budget.can_fit(small)
    large = torch.zeros(1000)  # 4000 bytes
    assert not budget.can_fit(large)


def test_budget_register_unregister():
    from expert_cache.budget_manager import VRAMBudgetManager
    budget = VRAMBudgetManager(budget_bytes=10000)
    t1 = torch.zeros(100)
    budget.register(t1)
    assert budget.used_bytes() == t1.element_size() * t1.numel()
    budget.unregister(t1)
    assert budget.used_bytes() == 0


def test_budget_enforce_evicts():
    from expert_cache.budget_manager import VRAMBudgetManager
    from expert_cache.lru_cache import LRUExpertCache as LRU
    budget = VRAMBudgetManager(budget_bytes=0)   # zero budget
    cache = LRU(capacity=5)
    cache.put(0, t())
    cache.put(1, t())

    # Force budget to report overage
    budget._used = 1000
    budget.budget = 1
    evicted = budget.enforce(cache)
    assert evicted > 0


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def test_analytics_hit_rate():
    from expert_cache.analytics import CacheAnalytics
    a = CacheAnalytics()
    a.record_hit(0)
    a.record_hit(1)
    a.record_miss(2)
    assert abs(a.hit_rate() - 2/3) < 1e-6


def test_analytics_suggest_thresholds():
    from expert_cache.analytics import CacheAnalytics
    a = CacheAnalytics()
    for i in range(10):
        for _ in range(i * 2):
            a.record_hit(i)
    thresholds = a.suggest_thresholds()
    assert "hot_threshold" in thresholds
    assert "warm_threshold" in thresholds
    assert thresholds["hot_threshold"] >= thresholds["warm_threshold"]


def test_analytics_summary():
    from expert_cache.analytics import CacheAnalytics
    a = CacheAnalytics()
    a.record_hit(0)
    a.record_miss(1)
    a.record_eviction(2)
    s = a.summary()
    assert s["total_hits"] == 1
    assert s["total_misses"] == 1
    assert s["total_evictions"] == 1
    assert s["unique_experts_seen"] == 2
