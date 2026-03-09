"""End-to-end integration benchmark for the Dana engine.

Runs all pillars together on the synthetic tiny MoE and measures
actual wall-clock TPS at each stage. Results are printed as a table
and stored in module-level RESULTS for post-test inspection.

Run with:
    pytest tests/test_integration_benchmark.py -v -s
"""
from __future__ import annotations

import time
import statistics
from dataclasses import dataclass
from typing import List

import pytest
import torch

from dana_engine.model.config import TinyMoEConfig
from dana_engine.model.transformer import TinyMoETransformer
from dana_engine.naive_inference import greedy_generate
from dana_engine.pipeline import DanaInferencePipeline, PipelineConfig
from dana_engine.batch.expert_grouper import ExpertGrouper, InferenceRequest
from dana_engine.batch.scheduler import ExpertAwareBatchScheduler
from dana_engine.speculative.self_draft_runner import OptimizedSelfDraftRunner, SelfDraftRunnerConfig
from dana_engine.speculative.tree_runner import OptimizedTreeRunner, TreeRunnerConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VOCAB = 256
SEQ_LEN = 8
MAX_NEW = 32
N_RUNS = 5  # repetitions per benchmark (median reported)


@pytest.fixture(scope="module")
def cfg():
    return TinyMoEConfig.micro()


@pytest.fixture(scope="module")
def model(cfg):
    m = TinyMoETransformer(cfg)
    m.eval()
    return m


def make_input(cfg, length=SEQ_LEN):
    return torch.randint(0, cfg.vocab_size, (1, length))


def median_tps(fn, runs=N_RUNS) -> tuple[float, float]:
    """Return (median_tps, median_tokens_per_step) over `runs` calls."""
    tps_list, tps_list2 = [], []
    for _ in range(runs):
        result = fn()
        tps_list.append(result[0])
        tps_list2.append(result[1])
    return statistics.median(tps_list), statistics.median(tps_list2)


# ---------------------------------------------------------------------------
# Stage 1 — Naïve greedy baseline
# ---------------------------------------------------------------------------

class TestBenchmarkNaive:
    """Baseline: standard greedy decode, 1 token per forward pass."""

    def test_naive_tps(self, model, cfg):
        def run():
            ids = make_input(cfg)
            t0 = time.perf_counter()
            result = greedy_generate(model, ids, max_new_tokens=MAX_NEW)
            elapsed = time.perf_counter() - t0
            new_tokens = result.tokens[0, ids.shape[1]:].tolist()
            return len(new_tokens) / elapsed, 1.0

        tps, tps_per_step = median_tps(run)
        print(f"\n[NAIVE]  TPS={tps:.1f}  tokens/step={tps_per_step:.2f}")
        RESULTS["naive_tps"] = tps
        assert tps > 0


# ---------------------------------------------------------------------------
# Stage 2 — Pipeline: naive path (through DanaInferencePipeline wrapper)
# ---------------------------------------------------------------------------

class TestBenchmarkPipelineNaive:
    """Pipeline naive path: adds pipeline overhead, still 1 token/step."""

    def test_pipeline_naive_tps(self, cfg):
        pipeline = DanaInferencePipeline(PipelineConfig(
            model_config=cfg,
            enable_spec_decode=False,
            enable_prefetch=False,
            max_new_tokens=MAX_NEW,
        ))

        def run():
            ids = make_input(cfg)
            result = pipeline.generate(ids, max_new_tokens=MAX_NEW)
            return result.tokens_per_second, 1.0

        tps, _ = median_tps(run)
        print(f"\n[PIPELINE-NAIVE]  TPS={tps:.1f}")
        RESULTS["pipeline_naive_tps"] = tps
        assert tps > 0


# ---------------------------------------------------------------------------
# Stage 3 — Speculative decode: self-draft (top-1 draft / top-2 verify)
# ---------------------------------------------------------------------------

class TestBenchmarkSpecDecode:
    """Self-draft speculative decoding: multiple tokens per verify pass."""

    def test_self_draft_tps(self, model, cfg):
        runner = OptimizedSelfDraftRunner(
            model,
            SelfDraftRunnerConfig(
                num_draft_tokens=4,
                max_new_tokens=MAX_NEW,
            ),
        )

        def run():
            ids = make_input(cfg)
            tokens, meta = runner.run(ids, max_new_tokens=MAX_NEW)
            return meta["tokens_per_second"], meta["avg_tokens_per_step"]

        tps, tps_per_step = median_tps(run)
        print(f"\n[SELF-DRAFT]  TPS={tps:.1f}  tokens/step={tps_per_step:.2f}  "
              f"accept_rate={_last_accept_rate(model, cfg):.2f}")
        RESULTS["self_draft_tps"] = tps
        RESULTS["self_draft_tokens_per_step"] = tps_per_step
        assert tps > 0
        assert tps_per_step >= 1.0

    def test_self_draft_acceptance_rate(self, model, cfg):
        runner = OptimizedSelfDraftRunner(
            model,
            SelfDraftRunnerConfig(num_draft_tokens=4, max_new_tokens=MAX_NEW),
        )
        rates = []
        for _ in range(N_RUNS):
            ids = make_input(cfg)
            _, meta = runner.run(ids, max_new_tokens=MAX_NEW)
            rates.append(meta["acceptance_rate"])
        median_rate = statistics.median(rates)
        RESULTS["self_draft_acceptance_rate"] = median_rate
        print(f"\n[SELF-DRAFT] median acceptance rate: {median_rate:.2%}")
        assert 0.0 <= median_rate <= 1.0


def _last_accept_rate(model, cfg):
    runner = OptimizedSelfDraftRunner(
        model, SelfDraftRunnerConfig(num_draft_tokens=4, max_new_tokens=MAX_NEW)
    )
    ids = make_input(cfg)
    _, meta = runner.run(ids, max_new_tokens=MAX_NEW)
    return meta["acceptance_rate"]


# ---------------------------------------------------------------------------
# Stage 4 — Tree speculative decoding
# ---------------------------------------------------------------------------

class TestBenchmarkTreeDecode:
    """Tree speculative decode: branching drafts, wider acceptance."""

    def test_tree_decode_tps(self, model, cfg):
        runner = OptimizedTreeRunner(
            model,
            TreeRunnerConfig(
                initial_depth=2,
                initial_width=2,
                max_new_tokens=MAX_NEW,
            ),
        )

        def run():
            ids = make_input(cfg)
            tokens, meta = runner.run(ids, max_new_tokens=MAX_NEW)
            return meta["tokens_per_second"], meta.get("acceptance_rate", 0.0)

        tps_list, acc_list = [], []
        for _ in range(N_RUNS):
            t, a = run()
            tps_list.append(t)
            acc_list.append(a)

        tps = statistics.median(tps_list)
        acc = statistics.median(acc_list)
        print(f"\n[TREE-DECODE]  TPS={tps:.1f}  accept_rate={acc:.2f}")
        RESULTS["tree_decode_tps"] = tps
        RESULTS["tree_decode_acceptance_rate"] = acc
        assert tps > 0

    def test_tree_adapts_depth(self, model, cfg):
        runner = OptimizedTreeRunner(
            model,
            TreeRunnerConfig(initial_depth=2, max_new_tokens=MAX_NEW, max_depth=4),
        )
        ids = make_input(cfg)
        _, meta = runner.run(ids, max_new_tokens=MAX_NEW)
        RESULTS["tree_final_depth"] = meta["final_depth"]
        RESULTS["tree_final_width"] = meta["final_width"]
        print(f"\n[TREE-DECODE] final depth={meta['final_depth']}  width={meta['final_width']}")
        assert 1 <= meta["final_depth"] <= 4


# ---------------------------------------------------------------------------
# Stage 5 — Expert-aware batching
# ---------------------------------------------------------------------------

class TestBenchmarkBatching:
    """Expert-aware batch scheduler: groups requests by expert overlap."""

    def test_grouper_throughput(self, model, cfg):
        grouper = ExpertGrouper(model, overlap_threshold=0.0)
        batch_size = 8
        requests = [
            InferenceRequest(f"r{i}", make_input(cfg))
            for i in range(batch_size)
        ]

        t0 = time.perf_counter()
        for _ in range(20):
            groups = grouper.group(requests)
        elapsed = time.perf_counter() - t0

        calls_per_sec = 20 / elapsed
        total_reqs = sum(len(g.requests) for g in groups)
        RESULTS["grouper_calls_per_sec"] = calls_per_sec
        RESULTS["grouper_groups_for_8_reqs"] = len(groups)
        print(f"\n[BATCHING]  groups={len(groups)} for {batch_size} reqs  "
              f"grouper={calls_per_sec:.1f} calls/s")
        assert total_reqs == batch_size

    def test_scheduler_drain(self, model, cfg):
        sched = ExpertAwareBatchScheduler(model)
        for i in range(6):
            sched.submit(InferenceRequest(f"r{i}", make_input(cfg)))
        batches = 0
        while sched.pending_count() > 0:
            sched.next_batch()
            batches += 1
        RESULTS["scheduler_batches_for_6_reqs"] = batches
        print(f"\n[SCHEDULER]  drained 6 requests in {batches} batch(es)")
        assert batches >= 1


# ---------------------------------------------------------------------------
# Stage 6 — Full pipeline: spec decode + prefetch enabled
# ---------------------------------------------------------------------------

class TestBenchmarkFullPipeline:
    """All pillars enabled together: spec decode + prefetch in one pass."""

    def test_full_pipeline_tps(self, cfg):
        pipeline = DanaInferencePipeline(PipelineConfig(
            model_config=cfg,
            enable_spec_decode=True,
            num_draft_tokens=4,
            enable_prefetch=True,
            max_new_tokens=MAX_NEW,
        ))

        def run():
            ids = make_input(cfg)
            result = pipeline.generate(ids, max_new_tokens=MAX_NEW)
            return result.tokens_per_second, result.avg_tokens_per_step

        tps, tps_per_step = median_tps(run)
        print(f"\n[FULL-PIPELINE]  TPS={tps:.1f}  tokens/step={tps_per_step:.2f}")
        RESULTS["full_pipeline_tps"] = tps
        RESULTS["full_pipeline_tokens_per_step"] = tps_per_step
        assert tps > 0
        assert tps_per_step >= 1.0

    def test_full_pipeline_spec_used(self, cfg):
        pipeline = DanaInferencePipeline(PipelineConfig(
            model_config=cfg,
            enable_spec_decode=True,
            num_draft_tokens=4,
            max_new_tokens=MAX_NEW,
        ))
        ids = make_input(cfg)
        result = pipeline.generate(ids, max_new_tokens=MAX_NEW)
        assert result.spec_decode_used is True

    def test_naive_vs_spec_tps_ratio(self):
        naive = RESULTS.get("naive_tps", 1.0)
        spec = RESULTS.get("full_pipeline_tps", 1.0)
        ratio = spec / naive if naive > 0 else 0.0
        RESULTS["spec_vs_naive_ratio"] = ratio
        print(f"\n[SPEEDUP]  spec/naive = {ratio:.2f}x")
        # On CPU the overhead of spec decode may negate gains (draft+verify > 1 pass);
        # we assert the pipeline at least doesn't catastrophically regress (>0.3x).
        assert ratio > 0.3


# ---------------------------------------------------------------------------
# Final summary printout
# ---------------------------------------------------------------------------

RESULTS: dict = {}


def pytest_sessionfinish(session, exitstatus):
    """Print benchmark table after all tests complete."""
    if not RESULTS:
        return
    print("\n" + "=" * 60)
    print("DANA ENGINE — INTEGRATION BENCHMARK RESULTS (tiny CPU model)")
    print("=" * 60)
    rows = [
        ("Naïve greedy (raw)",        "naive_tps",                    "TPS"),
        ("Naïve pipeline wrapper",     "pipeline_naive_tps",           "TPS"),
        ("Self-draft spec decode",     "self_draft_tps",               "TPS"),
        ("Self-draft tokens/step",     "self_draft_tokens_per_step",   "tok/step"),
        ("Self-draft acceptance",      "self_draft_acceptance_rate",   "rate"),
        ("Tree spec decode",           "tree_decode_tps",              "TPS"),
        ("Tree acceptance",            "tree_decode_acceptance_rate",  "rate"),
        ("Tree final depth",           "tree_final_depth",             "depth"),
        ("Full pipeline (spec+prefetch)", "full_pipeline_tps",         "TPS"),
        ("Full pipeline tok/step",     "full_pipeline_tokens_per_step","tok/step"),
        ("Spec vs naïve speedup",      "spec_vs_naive_ratio",          "x"),
        ("Grouper (8-req batches/s)",  "grouper_calls_per_sec",        "calls/s"),
    ]
    for label, key, unit in rows:
        val = RESULTS.get(key)
        if val is not None:
            print(f"  {label:<40} {val:>8.2f}  {unit}")
    print("=" * 60)
