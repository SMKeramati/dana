"""Tests for LRUExpertCache."""

import pytest
import torch
from expert_cache.lru_cache import LRUExpertCache


@pytest.fixture
def cache():
    return LRUExpertCache(capacity=3)


def t():
    return torch.randn(4, 4)


def test_put_and_get(cache):
    tensor = t()
    cache.put(0, tensor)
    result = cache.get(0)
    assert result is not None
    assert torch.allclose(tensor, result)


def test_miss_returns_none(cache):
    assert cache.get(99) is None


def test_capacity_evicts_lru(cache):
    cache.put(0, t())
    cache.put(1, t())
    cache.put(2, t())
    assert cache.is_full()

    # Access 0 and 1 to make them recently used; 2 becomes LRU
    cache.get(1)
    cache.get(0)
    # Now insert 3 → should evict 2 (LRU)
    evicted = cache.put(3, t())
    assert evicted == 2
    assert cache.get(2) is None
    assert cache.get(3) is not None


def test_update_existing_key(cache):
    t1 = torch.ones(2)
    t2 = torch.zeros(2)
    cache.put(0, t1)
    cache.put(0, t2)
    loaded = cache.get(0)
    assert loaded is not None
    assert torch.allclose(loaded, t2)


def test_explicit_evict(cache):
    cache.put(0, t())
    cache.put(1, t())
    evicted_id = cache.evict()
    assert evicted_id == 0  # LRU (first inserted)
    assert len(cache) == 1


def test_evict_empty_returns_none(cache):
    assert cache.evict() is None


def test_contains(cache):
    cache.put(5, t())
    assert cache.contains(5)
    assert not cache.contains(99)


def test_len(cache):
    assert len(cache) == 0
    cache.put(0, t())
    assert len(cache) == 1
    cache.put(1, t())
    assert len(cache) == 2


def test_cached_ids(cache):
    cache.put(0, t())
    cache.put(1, t())
    assert set(cache.cached_ids()) == {0, 1}
