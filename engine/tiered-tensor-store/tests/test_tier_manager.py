"""Tests for TieredTensorStore — core tier management."""

from __future__ import annotations

import tempfile

import pytest
import torch

from tiered_tensor_store.tier_manager import TieredTensorStore, Tier


@pytest.fixture
def store(tmp_path):
    return TieredTensorStore(
        base_dir=str(tmp_path / "store"),
        hot_budget_bytes=10 * 1024**2,   # 10MB
        ram_budget_bytes=100 * 1024**2,  # 100MB
        auto_promote=False,              # don't auto-promote in unit tests
    )


@pytest.fixture
def small_tensor():
    return torch.randn(16, 16)  # ~1KB


# ---------------------------------------------------------------------------
# Store / load round-trips
# ---------------------------------------------------------------------------

def test_store_and_load_ssd(store, small_tensor):
    store.store("t1", small_tensor, tier="ssd")
    loaded = store.load("t1")
    assert torch.allclose(small_tensor, loaded.cpu(), atol=1e-5)


def test_store_and_load_ram(store, small_tensor):
    store.store("t1", small_tensor, tier="ram")
    assert store.tier_of("t1") == "ram"
    loaded = store.load("t1")
    assert loaded.shape == small_tensor.shape


def test_store_and_load_hot(store, small_tensor):
    store.store("t1", small_tensor, tier="hot")
    assert store.tier_of("t1") == "hot"
    loaded = store.load("t1")
    assert torch.allclose(small_tensor, loaded.cpu(), atol=1e-5)


def test_load_nonexistent_raises(store):
    with pytest.raises(KeyError, match="not found"):
        store.load("nonexistent_key")


# ---------------------------------------------------------------------------
# Tier of / access count
# ---------------------------------------------------------------------------

def test_tier_of_reports_correct_tier(store, small_tensor):
    store.store("a", small_tensor, tier="ram")
    store.store("b", small_tensor, tier="ssd")
    assert store.tier_of("a") == "ram"
    assert store.tier_of("b") == "ssd"


def test_access_count_increments(store, small_tensor):
    store.store("t1", small_tensor, tier="ssd")
    assert store.access_count("t1") == 0
    store.load("t1")
    store.load("t1")
    assert store.access_count("t1") == 2


def test_access_count_zero_for_unknown(store):
    assert store.access_count("never_stored") == 0


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------

def test_evict_to_ssd(store, small_tensor):
    store.store("t1", small_tensor, tier="hot")
    assert store.tier_of("t1") == "hot"
    store.evict("t1", to_tier="ssd")
    assert store.tier_of("t1") == "ssd"
    # Can still load after eviction
    loaded = store.load("t1")
    assert loaded.shape == small_tensor.shape


def test_evict_from_ram_to_ssd(store, small_tensor):
    store.store("t1", small_tensor, tier="ram")
    store.evict("t1", to_tier="ssd")
    assert store.tier_of("t1") == "ssd"


def test_evict_same_tier_noop(store, small_tensor):
    store.store("t1", small_tensor, tier="ssd")
    store.evict("t1", to_tier="ssd")  # should not raise
    assert store.tier_of("t1") == "ssd"


# ---------------------------------------------------------------------------
# Auto-promote
# ---------------------------------------------------------------------------

def test_auto_promote_on_load(tmp_path, small_tensor):
    store = TieredTensorStore(
        base_dir=str(tmp_path / "store"),
        hot_budget_bytes=10 * 1024**2,
        auto_promote=True,
    )
    store.store("t1", small_tensor, tier="ssd")
    assert store.tier_of("t1") == "ssd"
    loaded = store.load("t1")
    assert store.tier_of("t1") == "hot"
    assert torch.allclose(small_tensor, loaded.cpu(), atol=1e-5)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_all_keys(store, small_tensor):
    store.store("a", small_tensor, tier="hot")
    store.store("b", small_tensor, tier="ram")
    assert set(store.all_keys()) == {"a", "b"}


def test_stats(store, small_tensor):
    store.store("a", small_tensor, tier="hot")
    store.store("b", small_tensor, tier="ssd")
    s = store.stats()
    assert s["total_keys"] == 2
    assert s["per_tier"]["hot"] == 1
    assert s["per_tier"]["ssd"] == 1


# ---------------------------------------------------------------------------
# Overwrite existing key
# ---------------------------------------------------------------------------

def test_overwrite_key(store):
    t1 = torch.ones(4, 4)
    t2 = torch.zeros(4, 4)
    store.store("x", t1, tier="ssd")
    store.store("x", t2, tier="ssd")  # overwrite
    loaded = store.load("x")
    assert torch.allclose(loaded.cpu(), t2, atol=1e-5)
