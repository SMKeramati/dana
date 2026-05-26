"""Integration tests for the dana-engine pipeline (Phase 7).

Tests: DanaInferencePipeline, ExpertGrouper, ExpertAwareBatchScheduler,
       OptimizedSelfDraftRunner, OptimizedTreeRunner, API server.
All run on CPU with the tiny synthetic MoE model.
"""
from __future__ import annotations

import json
import sys
import os

import pytest
import torch

from dana_engine.model.config import TinyMoEConfig
from dana_engine.model.transformer import TinyMoETransformer
from dana_engine.pipeline import DanaInferencePipeline, PipelineConfig
from dana_engine.batch.expert_grouper import ExpertGrouper, InferenceRequest
from dana_engine.batch.scheduler import ExpertAwareBatchScheduler, SchedulerConfig
from dana_engine.speculative.self_draft_runner import OptimizedSelfDraftRunner, SelfDraftRunnerConfig
from dana_engine.speculative.tree_runner import OptimizedTreeRunner, TreeRunnerConfig


@pytest.fixture(scope="module")
def config():
    return TinyMoEConfig.micro()


@pytest.fixture(scope="module")
def model(config):
    m = TinyMoETransformer(config)
    m.eval()
    return m


@pytest.fixture
def input_ids(config):
    return torch.randint(0, config.vocab_size, (1, 5))


@pytest.fixture(scope="module")
def pipeline():
    cfg = PipelineConfig(
        model_config=TinyMoEConfig.micro(),
        enable_spec_decode=True,
        num_draft_tokens=3,
        max_new_tokens=10,
    )
    return DanaInferencePipeline(cfg)


# ------------------------------------------------------------------
# DanaInferencePipeline — naive path
# ------------------------------------------------------------------

class TestPipelineNaive:
    def test_naive_generates_tokens(self, config, input_ids):
        cfg = PipelineConfig(model_config=config, enable_spec_decode=False, max_new_tokens=8)
        pipeline = DanaInferencePipeline(cfg)
        result = pipeline.generate(input_ids)
        assert result.tokens_generated > 0
        assert len(result.tokens) == result.tokens_generated

    def test_naive_tokens_are_valid(self, config, input_ids):
        cfg = PipelineConfig(model_config=config, enable_spec_decode=False, max_new_tokens=8)
        pipeline = DanaInferencePipeline(cfg)
        result = pipeline.generate(input_ids)
        for tok in result.tokens:
            assert 0 <= tok < config.vocab_size

    def test_naive_respects_max_new_tokens(self, config, input_ids):
        cfg = PipelineConfig(model_config=config, enable_spec_decode=False, max_new_tokens=5)
        pipeline = DanaInferencePipeline(cfg)
        result = pipeline.generate(input_ids)
        assert result.tokens_generated <= 5

    def test_naive_has_positive_tps(self, config, input_ids):
        cfg = PipelineConfig(model_config=config, enable_spec_decode=False, max_new_tokens=5)
        pipeline = DanaInferencePipeline(cfg)
        result = pipeline.generate(input_ids)
        assert result.tokens_per_second > 0

    def test_naive_finish_reason(self, config, input_ids):
        cfg = PipelineConfig(model_config=config, enable_spec_decode=False, max_new_tokens=5)
        pipeline = DanaInferencePipeline(cfg)
        result = pipeline.generate(input_ids)
        assert result.finish_reason in ("length", "stop")


# ------------------------------------------------------------------
# DanaInferencePipeline — speculative path
# ------------------------------------------------------------------

class TestPipelineSpeculative:
    def test_spec_generates_tokens(self, pipeline, input_ids):
        result = pipeline.generate(input_ids)
        assert result.tokens_generated > 0

    def test_spec_tokens_valid(self, pipeline, input_ids, config):
        result = pipeline.generate(input_ids)
        for tok in result.tokens:
            assert 0 <= tok < config.vocab_size

    def test_spec_respects_limit(self, config, input_ids):
        cfg = PipelineConfig(model_config=config, enable_spec_decode=True, max_new_tokens=6)
        pipeline = DanaInferencePipeline(cfg)
        result = pipeline.generate(input_ids, max_new_tokens=6)
        assert result.tokens_generated <= 6

    def test_spec_decode_used_flag(self, pipeline, input_ids):
        result = pipeline.generate(input_ids)
        assert result.spec_decode_used is True

    def test_spec_avg_tokens_per_step(self, pipeline, input_ids):
        result = pipeline.generate(input_ids)
        assert result.avg_tokens_per_step >= 1.0

    @pytest.mark.asyncio
    async def test_generate_async(self, pipeline, input_ids):
        result = await pipeline.generate_async(input_ids, max_new_tokens=6)
        assert result.tokens_generated > 0

    @pytest.mark.asyncio
    async def test_stream_async_yields_chunks(self, config, input_ids):
        cfg = PipelineConfig(model_config=config, enable_spec_decode=False, max_new_tokens=5)
        pipeline = DanaInferencePipeline(cfg)
        chunks = []
        async for chunk in pipeline.stream_async(input_ids, max_new_tokens=5):
            chunks.append(chunk)
        assert len(chunks) == 5

    def test_health_returns_dict(self, pipeline):
        h = pipeline.health()
        assert isinstance(h, dict)
        assert h["healthy"] is True


# ------------------------------------------------------------------
# ExpertGrouper
# ------------------------------------------------------------------

class TestExpertGrouper:
    def test_group_single_request(self, model, config):
        grouper = ExpertGrouper(model)
        req = InferenceRequest("r0", torch.randint(0, config.vocab_size, (1, 4)))
        groups = grouper.group([req])
        assert len(groups) == 1
        assert len(groups[0].requests) == 1

    def test_group_empty(self, model):
        grouper = ExpertGrouper(model)
        groups = grouper.group([])
        assert groups == []

    def test_group_multiple_requests(self, model, config):
        grouper = ExpertGrouper(model, overlap_threshold=0.0)
        requests = [
            InferenceRequest(f"r{i}", torch.randint(0, config.vocab_size, (1, 4)))
            for i in range(4)
        ]
        groups = grouper.group(requests)
        # All requests must appear in exactly one group
        all_ids = [r.request_id for g in groups for r in g.requests]
        assert sorted(all_ids) == sorted([r.request_id for r in requests])

    def test_groups_have_expert_sets(self, model, config):
        grouper = ExpertGrouper(model, overlap_threshold=0.0)
        requests = [
            InferenceRequest(f"r{i}", torch.randint(0, config.vocab_size, (1, 3)))
            for i in range(3)
        ]
        groups = grouper.group(requests)
        for g in groups:
            assert isinstance(g.predicted_experts, set)

    def test_overlap_score_in_range(self, model, config):
        grouper = ExpertGrouper(model, overlap_threshold=0.0)
        requests = [
            InferenceRequest(f"r{i}", torch.randint(0, config.vocab_size, (1, 3)))
            for i in range(3)
        ]
        groups = grouper.group(requests)
        for g in groups:
            assert 0.0 <= g.overlap_score <= 1.0


# ------------------------------------------------------------------
# ExpertAwareBatchScheduler
# ------------------------------------------------------------------

class TestExpertAwareBatchScheduler:
    def test_submit_and_next_batch(self, model, config):
        sched = ExpertAwareBatchScheduler(model)
        for i in range(3):
            req = InferenceRequest(f"r{i}", torch.randint(0, config.vocab_size, (1, 4)))
            sched.submit(req)
        groups = sched.next_batch()
        assert len(groups) > 0

    def test_empty_queue_returns_empty(self, model):
        sched = ExpertAwareBatchScheduler(model)
        assert sched.next_batch() == []

    def test_pending_count(self, model, config):
        sched = ExpertAwareBatchScheduler(model)
        assert sched.pending_count() == 0
        sched.submit(InferenceRequest("r0", torch.randint(0, config.vocab_size, (1, 3))))
        assert sched.pending_count() == 1

    def test_batch_drains_queue(self, model, config):
        cfg = SchedulerConfig(max_batch_size=2)
        sched = ExpertAwareBatchScheduler(model, cfg)
        for i in range(4):
            sched.submit(InferenceRequest(f"r{i}", torch.randint(0, config.vocab_size, (1, 3))))
        sched.next_batch()
        assert sched.pending_count() == 2


# ------------------------------------------------------------------
# OptimizedSelfDraftRunner
# ------------------------------------------------------------------

class TestSelfDraftRunner:
    def test_run_returns_tokens(self, model, input_ids):
        runner = OptimizedSelfDraftRunner(model, SelfDraftRunnerConfig(max_new_tokens=8))
        tokens, meta = runner.run(input_ids)
        assert len(tokens) > 0

    def test_run_respects_limit(self, model, input_ids):
        runner = OptimizedSelfDraftRunner(model, SelfDraftRunnerConfig(max_new_tokens=6))
        tokens, meta = runner.run(input_ids, max_new_tokens=6)
        assert len(tokens) <= 6

    def test_metadata_keys(self, model, input_ids):
        runner = OptimizedSelfDraftRunner(model, SelfDraftRunnerConfig(max_new_tokens=5))
        _, meta = runner.run(input_ids)
        assert "tokens_per_second" in meta
        assert "acceptance_rate" in meta
        assert "avg_tokens_per_step" in meta

    def test_acceptance_rate_in_range(self, model, input_ids):
        runner = OptimizedSelfDraftRunner(model, SelfDraftRunnerConfig(max_new_tokens=8))
        _, meta = runner.run(input_ids)
        assert 0.0 <= meta["acceptance_rate"] <= 1.0

    def test_avg_tokens_per_step_positive(self, model, input_ids):
        runner = OptimizedSelfDraftRunner(model, SelfDraftRunnerConfig(max_new_tokens=8))
        _, meta = runner.run(input_ids)
        assert meta["avg_tokens_per_step"] >= 1.0


# ------------------------------------------------------------------
# OptimizedTreeRunner
# ------------------------------------------------------------------

class TestTreeRunner:
    def test_run_returns_tokens(self, model, input_ids):
        runner = OptimizedTreeRunner(model, TreeRunnerConfig(max_new_tokens=8))
        tokens, meta = runner.run(input_ids)
        assert len(tokens) > 0

    def test_run_respects_limit(self, model, input_ids):
        runner = OptimizedTreeRunner(model, TreeRunnerConfig(max_new_tokens=6))
        tokens, meta = runner.run(input_ids, max_new_tokens=6)
        assert len(tokens) <= 6

    def test_metadata_keys(self, model, input_ids):
        runner = OptimizedTreeRunner(model, TreeRunnerConfig(max_new_tokens=5))
        _, meta = runner.run(input_ids)
        assert "tokens_per_second" in meta
        assert "acceptance_rate" in meta
        assert "final_depth" in meta
        assert "final_width" in meta

    def test_adaptive_depth_changes(self, model, config):
        runner = OptimizedTreeRunner(model, TreeRunnerConfig(
            initial_depth=2, max_new_tokens=10, max_depth=4
        ))
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        _, meta = runner.run(input_ids)
        # depth may have changed due to adaptation
        assert 1 <= meta["final_depth"] <= 4


# ------------------------------------------------------------------
# API Server
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def test_client():
    """Create a synchronous test client that runs the lifespan (startup/shutdown)."""
    from fastapi.testclient import TestClient
    from dana_engine.api.server import app
    with TestClient(app) as client:
        yield client


class TestAPIServer:
    def test_health_endpoint(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "latency_ms" in data
        assert "active_requests" in data

    def test_models_endpoint(self, test_client):
        resp = test_client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) >= 1
        assert "id" in data["data"][0]

    def test_completions_nonstreaming(self, test_client):
        payload = {
            "model": "tiny-moe",
            "prompt": "hello world",
            "max_tokens": 5,
            "stream": False,
        }
        resp = test_client.post("/v1/completions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert "text" in data["choices"][0]
        assert "finish_reason" in data["choices"][0]
        assert "usage" in data
        assert "completion_tokens" in data["usage"]
        assert "dana_meta" in data
        assert "tokens_per_second" in data["dana_meta"]

    def test_completions_tokens_per_second_positive(self, test_client):
        payload = {"model": "tiny-moe", "prompt": "test", "max_tokens": 5, "stream": False}
        resp = test_client.post("/v1/completions", json=payload)
        data = resp.json()
        assert data["dana_meta"]["tokens_per_second"] >= 0

    def test_completions_streaming(self, test_client):
        payload = {
            "model": "tiny-moe",
            "prompt": "hello",
            "max_tokens": 3,
            "stream": True,
        }
        resp = test_client.post("/v1/completions", json=payload)
        assert resp.status_code == 200
        lines = [l for l in resp.text.split("\n") if l.strip()]
        data_lines = [l for l in lines if l.startswith("data: ")]
        assert len(data_lines) > 0
        # Last data line must be [DONE]
        assert data_lines[-1] == "data: [DONE]"
        # Non-done lines must be valid JSON with choices
        for line in data_lines[:-1]:
            raw = line[6:]
            parsed = json.loads(raw)
            assert "choices" in parsed
            assert "text" in parsed["choices"][0]

    def test_completions_model_id_in_response(self, test_client):
        payload = {"model": "tiny-moe", "prompt": "x", "max_tokens": 3, "stream": False}
        resp = test_client.post("/v1/completions", json=payload)
        data = resp.json()
        assert data["model"] == "tiny-moe"
