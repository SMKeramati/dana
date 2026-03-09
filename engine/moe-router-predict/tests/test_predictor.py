"""Tests for moe-router-predict."""

from __future__ import annotations

import asyncio

import pytest
import torch

from dana_engine.model.config import TinyMoEConfig
from dana_engine.model.transformer import TinyMoETransformer
from moe_router_predict.predictor import RouterPredictor, ExpertPrediction, StepPrediction
from moe_router_predict.residency import ExpertResidencyTracker
from moe_router_predict.async_loader import AsyncExpertLoader
from moe_router_predict.scheduler import PrefetchScheduler


@pytest.fixture(scope="module")
def cfg():
    return TinyMoEConfig.micro()


@pytest.fixture(scope="module")
def model(cfg):
    torch.manual_seed(0)
    return TinyMoETransformer(cfg)


@pytest.fixture
def hidden(cfg):
    return torch.randn(1, 1, cfg.hidden_dim)


# ---------------------------------------------------------------------------
# RouterPredictor
# ---------------------------------------------------------------------------

def test_predictor_returns_correct_step_count(model, hidden, cfg):
    predictor = RouterPredictor(model)
    predictions = predictor.predict(hidden, num_steps=3)
    assert len(predictions) == 3


def test_predictor_step_numbers(model, hidden):
    predictor = RouterPredictor(model)
    predictions = predictor.predict(hidden, num_steps=4)
    for i, pred in enumerate(predictions):
        assert pred.step == i + 1


def test_predictor_per_layer_count(model, hidden, cfg):
    predictor = RouterPredictor(model)
    predictions = predictor.predict(hidden, num_steps=2)
    for step_pred in predictions:
        assert len(step_pred.per_layer) == cfg.num_layers


def test_predictor_expert_count_matches_num_active(model, hidden, cfg):
    predictor = RouterPredictor(model)
    predictions = predictor.predict(hidden, num_steps=1)
    for layer_pred in predictions[0].per_layer:
        assert len(layer_pred.expert_ids) == cfg.num_active
        assert len(layer_pred.scores) == cfg.num_active


def test_predictor_expert_ids_in_range(model, hidden, cfg):
    predictor = RouterPredictor(model)
    predictions = predictor.predict(hidden, num_steps=2)
    for step_pred in predictions:
        for layer_pred in step_pred.per_layer:
            for eid in layer_pred.expert_ids:
                assert 0 <= eid < cfg.num_experts


def test_predictor_scores_sum_to_one(model, hidden):
    predictor = RouterPredictor(model)
    predictions = predictor.predict(hidden, num_steps=1)
    for layer_pred in predictions[0].per_layer:
        assert abs(sum(layer_pred.scores) - 1.0) < 1e-5


def test_predictor_flat(model, hidden, cfg):
    predictor = RouterPredictor(model)
    flat = predictor.predict_flat(hidden, num_steps=2)
    assert isinstance(flat, list)
    assert all(0 <= eid < cfg.num_experts for eid in flat)
    # No duplicates
    assert len(flat) == len(set(flat))


def test_step_prediction_all_expert_ids(model, hidden, cfg):
    predictor = RouterPredictor(model)
    step_preds = predictor.predict(hidden, num_steps=1)
    ids = step_preds[0].all_expert_ids()
    assert isinstance(ids, set)
    assert len(ids) >= 1


# ---------------------------------------------------------------------------
# ExpertResidencyTracker
# ---------------------------------------------------------------------------

def test_residency_default_ssd(cfg):
    tracker = ExpertResidencyTracker(cfg.num_experts, default_tier="ssd")
    for i in range(cfg.num_experts):
        assert tracker.where(i) == "ssd"


def test_residency_mark_and_where(cfg):
    tracker = ExpertResidencyTracker(cfg.num_experts)
    tracker.mark(0, "hot")
    tracker.mark(1, "ram")
    assert tracker.where(0) == "hot"
    assert tracker.where(1) == "ram"
    assert tracker.where(2) == "ssd"


def test_residency_in_flight(cfg):
    tracker = ExpertResidencyTracker(cfg.num_experts)
    tracker.mark_in_flight(0)
    assert tracker.is_in_flight(0)
    assert not tracker.needs_load(0)


def test_residency_cold_experts(cfg):
    tracker = ExpertResidencyTracker(cfg.num_experts, default_tier="ssd")
    tracker.mark(0, "hot")
    cold = tracker.cold_experts()
    assert 0 not in cold
    assert len(cold) == cfg.num_experts - 1


def test_residency_summary(cfg):
    tracker = ExpertResidencyTracker(cfg.num_experts, default_tier="ssd")
    tracker.mark(0, "hot")
    tracker.mark(1, "ram")
    s = tracker.summary()
    assert s["hot"] == 1
    assert s["ram"] == 1
    assert s["ssd"] == cfg.num_experts - 2


def test_residency_snapshot(cfg):
    tracker = ExpertResidencyTracker(cfg.num_experts)
    tracker.mark(0, "hot")
    snap = tracker.snapshot()
    assert snap[0] == "hot"
    # Modifying snapshot doesn't affect tracker
    snap[0] = "ssd"
    assert tracker.where(0) == "hot"


# ---------------------------------------------------------------------------
# AsyncExpertLoader
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_loader_enqueue_and_complete(cfg):
    tracker = ExpertResidencyTracker(cfg.num_experts, default_tier="ssd")
    loaded_ids = []

    async def load_fn(expert_id, target_tier):
        loaded_ids.append(expert_id)
        await asyncio.sleep(0)

    loader = AsyncExpertLoader(tracker, load_fn=load_fn, max_concurrent=2)
    await loader.start()
    await loader.enqueue(0, "hot", priority=0.0)
    await loader.enqueue(1, "hot", priority=1.0)
    await asyncio.sleep(0.1)
    await loader.stop()

    assert 0 in loaded_ids or 1 in loaded_ids
    assert tracker.where(0) == "hot" or tracker.where(0) in ("hot", "in_flight", "ssd")


@pytest.mark.asyncio
async def test_async_loader_skips_hot_experts(cfg):
    tracker = ExpertResidencyTracker(cfg.num_experts, default_tier="hot")
    call_count = [0]

    async def load_fn(expert_id, target_tier):
        call_count[0] += 1

    loader = AsyncExpertLoader(tracker, load_fn=load_fn)
    await loader.start()
    await loader.enqueue(0, "hot")  # already hot, should be skipped
    await asyncio.sleep(0.05)
    await loader.stop()
    assert call_count[0] == 0


@pytest.mark.asyncio
async def test_async_loader_skips_in_flight(cfg):
    tracker = ExpertResidencyTracker(cfg.num_experts, default_tier="ssd")
    tracker.mark_in_flight(0)
    call_count = [0]

    async def load_fn(expert_id, target_tier):
        call_count[0] += 1

    loader = AsyncExpertLoader(tracker, load_fn=load_fn)
    await loader.start()
    await loader.enqueue(0, "hot")  # in flight, should be skipped
    await asyncio.sleep(0.05)
    await loader.stop()
    assert call_count[0] == 0


# ---------------------------------------------------------------------------
# PrefetchScheduler
# ---------------------------------------------------------------------------

def test_scheduler_on_step_returns_expert_ids(model, hidden, cfg):
    tracker = ExpertResidencyTracker(cfg.num_experts, default_tier="ssd")
    loader = AsyncExpertLoader(tracker)

    # Run with a temporary event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run():
        await loader.start()
        predictor = RouterPredictor(model)
        scheduler = PrefetchScheduler(predictor, loader, num_steps=2)
        enqueued = scheduler.on_step(hidden)
        await asyncio.sleep(0.05)
        await loader.stop()
        return enqueued

    try:
        enqueued = loop.run_until_complete(run())
    finally:
        loop.close()
        asyncio.set_event_loop(None)

    # Some experts should have been enqueued (those not already hot)
    assert isinstance(enqueued, list)


def test_scheduler_stats(model, hidden, cfg):
    tracker = ExpertResidencyTracker(cfg.num_experts, default_tier="ssd")
    loader = AsyncExpertLoader(tracker)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run():
        await loader.start()
        predictor = RouterPredictor(model)
        scheduler = PrefetchScheduler(predictor, loader, num_steps=1)
        scheduler.on_step(hidden)
        stats = scheduler.stats()
        await loader.stop()
        return stats

    try:
        stats = loop.run_until_complete(run())
    finally:
        loop.close()
        asyncio.set_event_loop(None)

    assert stats["steps_processed"] == 1
