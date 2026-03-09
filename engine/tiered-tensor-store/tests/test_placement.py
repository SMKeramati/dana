"""Tests for PlacementOptimizer and BackgroundPromoter."""

from __future__ import annotations

import asyncio

import pytest
import torch

from tiered_tensor_store.placement_optimizer import PlacementOptimizer
from tiered_tensor_store.promoter import BackgroundPromoter
from tiered_tensor_store.tier_manager import TieredTensorStore


# ---------------------------------------------------------------------------
# PlacementOptimizer
# ---------------------------------------------------------------------------

ONE_MB = 1024 * 1024


@pytest.fixture
def optimizer():
    return PlacementOptimizer(
        hot_budget_bytes=3 * ONE_MB,
        ram_budget_bytes=10 * ONE_MB,
    )


def test_most_accessed_goes_hot(optimizer):
    counts = {"a": 100, "b": 50, "c": 10}
    sizes = {"a": ONE_MB, "b": ONE_MB, "c": ONE_MB}
    result = optimizer.optimize(counts, sizes)
    assert result["a"] == "hot"
    assert result["b"] == "hot"
    assert result["c"] == "hot"  # all 3 fit in 3MB hot budget


def test_overflow_to_ram(optimizer):
    counts = {"a": 100, "b": 50, "c": 10, "d": 5}
    sizes = {k: ONE_MB for k in counts}
    result = optimizer.optimize(counts, sizes)
    assert result["a"] == "hot"
    assert result["b"] == "hot"
    assert result["c"] == "hot"
    assert result["d"] == "ram"  # hot budget full, goes to ram


def test_overflow_to_ssd():
    optimizer = PlacementOptimizer(
        hot_budget_bytes=ONE_MB,
        ram_budget_bytes=ONE_MB,
    )
    counts = {"a": 100, "b": 50, "c": 10}
    sizes = {k: ONE_MB for k in counts}
    result = optimizer.optimize(counts, sizes)
    assert result["a"] == "hot"
    assert result["b"] == "ram"
    assert result["c"] == "ssd"


def test_empty_input(optimizer):
    result = optimizer.optimize({}, {})
    assert result == {}


def test_zero_access_goes_ssd():
    optimizer = PlacementOptimizer(hot_budget_bytes=0, ram_budget_bytes=0)
    result = optimizer.optimize({"a": 0}, {"a": ONE_MB})
    assert result["a"] == "ssd"


def test_delta_computes_moves(optimizer):
    current = {"a": "hot", "b": "ram", "c": "ssd"}
    recommended = {"a": "ram", "b": "hot", "c": "ssd"}  # swap a and b, c unchanged
    moves = optimizer.delta(current, recommended)
    move_dict = {k: (f, t) for k, f, t in moves}
    assert move_dict["a"] == ("hot", "ram")
    assert move_dict["b"] == ("ram", "hot")
    assert "c" not in move_dict  # no change


def test_optimize_store(tmp_path):
    store = TieredTensorStore(base_dir=str(tmp_path), auto_promote=False)
    t = torch.randn(8, 8)
    store.store("e0", t, tier="ssd")
    store.store("e1", t, tier="ssd")

    # Simulate e0 being accessed 10 times
    for _ in range(10):
        store._entries["e0"].access_count += 1

    optimizer = PlacementOptimizer(
        hot_budget_bytes=t.element_size() * t.numel() + 1,
        ram_budget_bytes=t.element_size() * t.numel() * 10,
    )
    result = optimizer.optimize_store(store)
    assert result["e0"] == "hot"
    assert result["e1"] == "ram"


# ---------------------------------------------------------------------------
# BackgroundPromoter
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_promoter_runs_once(tmp_path):
    store = TieredTensorStore(base_dir=str(tmp_path), auto_promote=False)
    t = torch.randn(4, 4)
    store.store("e0", t, tier="ssd")

    # Simulate high access count
    store._entries["e0"].access_count = 100

    promoter = BackgroundPromoter(
        store,
        interval_seconds=999,  # won't auto-trigger
        hot_budget_bytes=t.element_size() * t.numel() + 1,
        ram_budget_bytes=t.element_size() * t.numel() * 10,
    )

    moves = await promoter.run_once()
    assert any(key == "e0" for key, _, _ in moves)


@pytest.mark.asyncio
async def test_promoter_start_stop(tmp_path):
    store = TieredTensorStore(base_dir=str(tmp_path))
    promoter = BackgroundPromoter(store, interval_seconds=0.05)
    await promoter.start()
    await asyncio.sleep(0.15)  # let it run ~3 cycles
    await promoter.stop()
    # Should not raise; moves_executed may be 0 (empty store)
    assert promoter.moves_executed >= 0
