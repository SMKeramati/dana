# Dana: Custom MoE Inference Engine

## The Pivot Decision

**Previous direction:** Dana as an API platform wrapping SGLang/KTransformers to serve Qwen3-235B.

**New direction:** Dana as a **custom MoE inference engine** built from scratch. The engine IS the product. The API platform is just a thin layer on top.

**Why the pivot:**
- Wrapping SGLang/KTransformers means our "core AI" is someone else's code. Zero defensibility.
- The real unsolved problem is: **MoE models are I/O-bound on consumer/prosumer hardware.** Only ~8/128 experts are active per token, but swapping cold experts across PCIe is the bottleneck nobody has cracked well.
- Building the engine from scratch means every line of code is internal R&D.
- The engine alone is the moat. Everything else (API, dashboard, billing) is commodity.

---

## The Core Thesis

> **MoE models activate only 5-15% of parameters per token, yet current inference engines treat them like dense models.** Dana exploits the sparse activation pattern with six synergistic innovations that turn I/O-bound inference into compute-bound inference on consumer hardware.

**Target:** Run Qwen3-235B-MoE (or DeepSeek-V3-671B) at **30+ tokens/sec** on a $20K workstation (2x RTX 4090 + 512GB DDR5 RAM) instead of $200K+ in H100s.

---

## Product Unbundling Strategy

The engine is built as **7 separate products** — 6 standalone libraries + 1 private integrator. Each library is independently useful, independently publishable, and has its own repo, tests, and PyPI package. The 7th product composes them all into the full engine and adds the proprietary serving layer.

### Why Unbundle?

1. **Stars multiply.** 6 repos with 500 stars each > 1 repo with 1,000 stars. Each repo has its own audience.
2. **Hiring signal.** "We maintain 4 popular MoE infrastructure libraries" is a stronger pitch than "we have one engine."
3. **Dependency lock-in.** If researchers/companies build on `expert-cache` and `tiered-tensor-store`, they're already in your ecosystem. Migration cost to Dana (the full engine) = zero.
4. **Competitive intelligence.** GitHub issues and PRs reveal what the market needs.
5. **Free QA.** Community finds bugs, adds features, battle-tests the code. You integrate the best improvements.
6. **Plausible separation.** No one looks at `expert-cache` and thinks "this is 1/6 of a competitive inference engine." It looks like a useful caching library.

### The 7 Products at a Glance

| # | Product | Release Strategy | License | Standalone Value |
|---|---|---|---|---|
| 1 | `moe-router-predict` | **Full OSS** | MIT | MoE routing research, expert utilization analysis |
| 2 | `expert-cache` | **Full OSS** | MIT | Anyone running MoE on limited VRAM |
| 3 | `tiered-tensor-store` | **Full OSS** | MIT | ANY large model inference, not just MoE |
| 4 | `moe-quant` | **Full OSS** | MIT | Per-expert quantization (huge community) |
| 5 | `spec-decode-tree` | **Limited** (samples + paper, no optimized core) | Apache 2.0 | Get citations, establish thought leadership |
| 6 | `moe-self-draft` | **Limited** (concept + benchmarks, core impl private) | Apache 2.0 | Publish the idea first, keep optimized impl |
| 7 | `dana-engine` | **Private** (never released) | Proprietary | The integration + batching moat |

---

## Product 1: `moe-router-predict` — Expert Activation Predictor

**Release: Full OSS (MIT)**

Given hidden states from a MoE transformer, predict which experts the next N tokens will require — enabling prefetching, caching, and batching optimizations upstream.

### Problem

When the router selects expert #47, the engine stalls waiting for it to load from RAM to VRAM across PCIe (~32GB/s). If we could predict the selection *before* it happens, we could start loading early.

### Solution

Use a lightweight lookahead (draft model or router-only forward pass) to predict which experts the next N tokens will need, and begin DMA transfers *before* compute needs them.

**Key metric:** PCIe idle time reduced from ~60% to <10%.

### Subcomponents

| Module | Description | Est. LoC |
|---|---|---|
| `src/moe_router_predict/predictor.py` | Router-only forward pass on draft hidden states to predict expert IDs N tokens ahead | ~300 |
| `src/moe_router_predict/async_loader.py` | Double-buffered async DMA transfer manager (compute on buffer A while loading buffer B) | ~500 |
| `src/moe_router_predict/residency.py` | Expert location tracking: VRAM, in-flight, RAM, SSD | ~200 |
| `src/moe_router_predict/scheduler.py` | Prefetch queue with priority ordering and stale cancellation | ~300 |
| `tests/` | Correctness tests (same output with/without prefetch), I/O overlap measurement | ~300 |
| `benchmarks/` | Before/after comparison, wall-clock speedup | ~200 |
| **Total** | | **~1,800** |

### Standalone Value

- Researchers studying MoE routing patterns and expert utilization
- Anyone building MoE inference tools, profilers, or visualizers
- MoE training frameworks wanting routing-aware scheduling

### Folder Structure

```
moe-router-predict/
├── pyproject.toml
├── README.md
├── LICENSE                          # MIT
├── src/
│   └── moe_router_predict/
│       ├── __init__.py
│       ├── predictor.py             # Router lookahead predictor
│       ├── async_loader.py          # Async DMA transfer manager
│       ├── residency.py             # Expert residency tracker
│       └── scheduler.py             # Prefetch queue + priority scheduler
├── tests/
│   ├── test_predictor.py
│   ├── test_async_loader.py
│   └── test_scheduler.py
└── benchmarks/
    └── bench_prefetch.py
```

---

## Product 2: `expert-cache` — Intelligent Expert Caching

**Release: Full OSS (MIT)**

Pluggable caching library that goes beyond LRU — frequency-aware, predictive, and budget-constrained. Works with any MoE framework.

### Problem

LRU cache evicts experts that happen to be old but are frequently used. Expert popularity follows a power-law distribution — some experts are "hot" across most inputs.

### Solution

Frequency-aware + predictive caching. Track per-expert activation frequency across a sliding window. Retain high-frequency experts permanently in VRAM. Predict upcoming expert needs based on token context.

**Key metric:** Cache hit rate from ~60% (LRU) to 90%+ (predictive).

### Subcomponents

| Module | Description | Est. LoC |
|---|---|---|
| `src/expert_cache/lru_cache.py` | Baseline LRU expert cache | ~150 |
| `src/expert_cache/frequency_cache.py` | Frequency-aware cache (LFU + sliding-window recency) | ~200 |
| `src/expert_cache/predictive_cache.py` | Uses router lookahead hints to pre-cache before needed | ~350 |
| `src/expert_cache/classifier.py` | Hot/warm/cold classification based on activation frequency | ~150 |
| `src/expert_cache/budget_manager.py` | VRAM budget enforcement + intelligent eviction | ~200 |
| `src/expert_cache/analytics.py` | Hit/miss tracking, self-tuning tier thresholds | ~150 |
| `tests/` | Cache correctness, eviction policy, budget compliance tests | ~250 |
| `benchmarks/` | LRU vs frequency vs predictive on 1000-token generation | ~200 |
| **Total** | | **~1,650** |

### Standalone Value

- Anyone running MoE inference on consumer GPUs (limited VRAM)
- MoE serving frameworks wanting a drop-in cache upgrade
- Researchers measuring expert utilization patterns

### Folder Structure

```
expert-cache/
├── pyproject.toml
├── README.md
├── LICENSE                          # MIT
├── src/
│   └── expert_cache/
│       ├── __init__.py
│       ├── lru_cache.py
│       ├── frequency_cache.py
│       ├── predictive_cache.py
│       ├── classifier.py
│       ├── budget_manager.py
│       └── analytics.py
├── tests/
│   ├── test_lru.py
│   ├── test_frequency.py
│   ├── test_predictive.py
│   └── test_budget.py
└── benchmarks/
    └── bench_cache.py
```

---

## Product 3: `tiered-tensor-store` — Three-Tier Tensor Storage

**Release: Full OSS (MIT)**

General-purpose library for managing large tensor collections across VRAM / RAM / SSD. Not MoE-specific — useful for **any** large model inference on constrained hardware.

### Problem

Not all 128 experts fit in VRAM (limited). Not all fit in RAM (possible but expensive). SSD is cheap but slower. No library handles this three-tier hierarchy well.

### Solution

Three-tier storage hierarchy with intelligent placement:

```
┌─────────────────────────┐
│   VRAM (24-80GB)        │  ← Hot tensors: pinned, FP16/Q8
│   Latency: 0 (resident) │
├─────────────────────────┤
│   System RAM (512GB)    │  ← Warm tensors: Q4, prefetchable
│   Latency: ~1ms (PCIe)  │
├─────────────────────────┤
│   NVMe SSD (2-4TB)     │  ← Cold tensors: Q2, background load
│   Latency: ~5ms (NVMe)  │
└─────────────────────────┘
```

### Subcomponents

| Module | Description | Est. LoC |
|---|---|---|
| `src/tiered_tensor_store/tier_manager.py` | Unified interface for VRAM/RAM/SSD tensor storage | ~300 |
| `src/tiered_tensor_store/placement_optimizer.py` | Frequency-based tier assignment (greedy/ILP solver) | ~400 |
| `src/tiered_tensor_store/promoter.py` | Background promotion/demotion between tiers on access pattern change | ~300 |
| `src/tiered_tensor_store/ssd_direct.py` | Direct I/O for NVMe, bypass OS page cache | ~250 |
| `src/tiered_tensor_store/mmap_pool.py` | Memory-mapped tensor pool for RAM tier | ~200 |
| `tests/` | Tier placement, promotion/demotion, I/O correctness tests | ~300 |
| `benchmarks/` | Throughput at each tier, placement optimizer quality | ~200 |
| **Total** | | **~1,950** |

### Standalone Value

- Any project serving large models on consumer/prosumer hardware
- GGML/llama.cpp-style projects wanting tiered storage
- Dataset loading pipelines with heterogeneous storage
- Broadest audience of all 6 libraries

### Folder Structure

```
tiered-tensor-store/
├── pyproject.toml
├── README.md
├── LICENSE                          # MIT
├── src/
│   └── tiered_tensor_store/
│       ├── __init__.py
│       ├── tier_manager.py
│       ├── placement_optimizer.py
│       ├── promoter.py
│       ├── ssd_direct.py
│       └── mmap_pool.py
├── tests/
│   ├── test_tier_manager.py
│   ├── test_placement.py
│   └── test_ssd.py
└── benchmarks/
    └── bench_storage.py
```

---

## Product 4: `moe-quant` — Per-Expert Sensitivity-Aware Quantization

**Release: Full OSS (MIT)**

Not all experts are equal. Hot experts deserve high precision. Cold experts can be aggressively compressed. This library profiles each expert's sensitivity and assigns optimal bit-widths.

### Problem

Experts in RAM are large (FP16/FP32). Transferring them across PCIe is the bottleneck. Standard uniform quantization (all Q4) wastes quality on hot experts and wastes bandwidth on cold ones.

### Solution

Tier-aware quantization:
- **Hot experts (pinned in VRAM):** FP16 or Q8 — maximum quality, already resident, no transfer cost.
- **Warm experts (in RAM):** Q4 — good balance of quality and transfer speed.
- **Cold experts (on SSD):** Q2 or Q3 — aggressive compression. Rarely activated; quality loss amortized.

**Key metric:** 2-4x faster expert loading with <1% quality degradation on benchmarks.

### Subcomponents

| Module | Description | Est. LoC |
|---|---|---|
| `src/moe_quant/quantize.py` | Per-expert quantization (Q2/Q3/Q4/Q8) with calibration data | ~400 |
| `src/moe_quant/sensitivity.py` | Measure per-expert output divergence at each bit-width | ~300 |
| `src/moe_quant/tier_assigner.py` | Assign bit-widths to minimize total quality loss under bandwidth budget | ~200 |
| `src/moe_quant/dequantize.py` | Fast vectorized dequantization, overlappable with transfer | ~300 |
| `src/moe_quant/dynamic.py` | Runtime re-quantization on tier promotion (cold→warm = Q2→Q4) | ~200 |
| `tests/` | Round-trip accuracy, sensitivity profiling, edge cases | ~250 |
| `benchmarks/` | Load speed + output divergence at each bit-width | ~200 |
| **Total** | | **~1,850** |

### Standalone Value

- Anyone quantizing MoE models (Mixtral, DeepSeek, Qwen)
- Quantization researchers exploring non-uniform strategies
- Model compression pipelines wanting expert-level granularity
- The quantization community is huge — "per-expert" angle is novel

### Folder Structure

```
moe-quant/
├── pyproject.toml
├── README.md
├── LICENSE                          # MIT
├── src/
│   └── moe_quant/
│       ├── __init__.py
│       ├── quantize.py
│       ├── sensitivity.py
│       ├── tier_assigner.py
│       ├── dequantize.py
│       └── dynamic.py
├── tests/
│   ├── test_quantize.py
│   ├── test_sensitivity.py
│   └── test_dequant.py
└── benchmarks/
    └── bench_quant.py
```

---

## Product 5: `spec-decode-tree` — Tree-Based Speculative Decoding

**Release: Limited (Apache 2.0) — reference impl + paper + benchmarks, no optimized core**

This is a major speed differentiator. Enough is published to get citations and establish thought leadership, but the production-optimized implementation stays in `dana-engine`.

### What's Published

- Algorithm description and theoretical analysis
- Unoptimized reference implementation (for understanding and citation)
- Benchmark results and comparison tables
- API specification and usage examples

### What's Private (lives in `dana-engine`)

- Optimized CUDA tree verification kernels
- Production attention mask construction
- Integration with prefetch pipeline
- Tuned adaptive branching heuristics

### How It Works

Standard speculation drafts a linear chain. If token 2 is wrong, tokens 3-8 are wasted:
```
Linear:  t1 → t2 → t3 → t4 → t5    (if t2 wrong, only t1 accepted)

Tree:          t1
              / \
           t2a  t2b
           /\    |
         t3a t3b t3c              (verify all in one pass, accept longest valid path)
```

**Result:** ~7 tokens/step (tree) vs ~4 tokens/step (linear) at same acceptance rate.

### Subcomponents (Reference Implementation)

| Module | Description | Est. LoC |
|---|---|---|
| `src/spec_decode_tree/tree_spec.py` | Tree candidate generation + flattened verification | ~400 |
| `src/spec_decode_tree/adaptive.py` | Dynamic draft length from rolling acceptance rate | ~200 |
| `src/spec_decode_tree/verify.py` | Batched verification with attention mask | ~250 |
| `src/spec_decode_tree/acceptance.py` | Acceptance rate tracker + per-layer analytics | ~150 |
| `tests/` | Output equivalence, acceptance tracking tests | ~250 |
| `benchmarks/` | Linear vs tree comparison, adaptive vs fixed | ~200 |
| **Total (reference)** | | **~1,450** |

*Note: The production implementation in `dana-engine` with optimized kernels is ~2.5x this.*

### Folder Structure

```
spec-decode-tree/
├── pyproject.toml
├── README.md
├── LICENSE                          # Apache 2.0
├── src/
│   └── spec_decode_tree/
│       ├── __init__.py
│       ├── tree_spec.py             # Reference (unoptimized)
│       ├── adaptive.py
│       ├── verify.py
│       └── acceptance.py
├── tests/
│   ├── test_tree.py
│   ├── test_adaptive.py
│   └── test_verify.py
└── benchmarks/
    └── bench_tree.py
```

---

## Product 6: `moe-self-draft` — Zero-Memory Speculative Decoding for MoE

**Release: Limited (Apache 2.0) — concept + benchmarks published, optimized impl private**

This idea is novel enough for a paper. Publishing establishes invention priority. But the fast, integrated implementation stays in `dana-engine`.

### What's Published

- Algorithm description and theoretical analysis
- Minimal reference implementation (proof of concept)
- Benchmark comparisons vs standard speculative decoding

### What's Private (lives in `dana-engine`)

- Optimized router logit extraction pipeline
- Prefetch-during-draft integration (the synergy with Product 1)
- Multi-sequence parallel drafting (the API serving enhancement)
- Production CUDA implementation

### The Key Insight

Standard speculative decoding needs a separate small draft model (~70% agreement, +1.5B memory). MoE-Self-Draft uses the **same model with top-1 expert** instead of top-2:
- Same attention, same embeddings, same router → ~85% agreement (vs ~70%)
- Zero extra memory — draft mode is a config flag
- **Bonus:** Draft pass router logits reveal exactly which experts verification needs → free prefetch window

### Subcomponents (Reference Implementation)

| Module | Description | Est. LoC |
|---|---|---|
| `src/moe_self_draft/self_draft.py` | Top-1 expert forward pass (draft mode) | ~300 |
| `src/moe_self_draft/logit_extractor.py` | Capture router probability distributions during draft | ~150 |
| `src/moe_self_draft/verify.py` | Standard draft-verify with pre-loaded experts | ~250 |
| `tests/` | Acceptance rate, output equivalence tests | ~200 |
| `benchmarks/` | Self-draft vs separate-draft comparison | ~200 |
| **Total (reference)** | | **~1,100** |

*Note: The production version with prefetch integration + multi-sequence is ~3x this.*

### Folder Structure

```
moe-self-draft/
├── pyproject.toml
├── README.md
├── LICENSE                          # Apache 2.0
├── src/
│   └── moe_self_draft/
│       ├── __init__.py
│       ├── self_draft.py            # Reference (unoptimized)
│       ├── logit_extractor.py
│       └── verify.py
├── tests/
│   ├── test_self_draft.py
│   └── test_verify.py
└── benchmarks/
    └── bench_self_draft.py
```

---

## Product 7: `dana-engine` — The Private Integrator

**Release: NEVER. Proprietary. All rights reserved.**

This is the orchestration layer that composes all 6 libraries into a single optimized inference pipeline. The individual libraries are useful alone, but the **integration** — how they compose, how data flows between them, how they multiply each other's speedups — that's the real product.

### What Lives Here (and nowhere else)

#### 7a. Expert-Aware Request Batching (Pillar 3) — The API Moat

Group concurrent requests by predicted expert overlap. Load expert once, process all requests that need it. Useless for single-user, critical for API serving.

**Key metric:** Total expert loads reduced by 5-10x under concurrent load.

| Module | Description | Est. LoC |
|---|---|---|
| `src/dana_engine/batch/expert_predictor.py` | Predict expert needs per queued request | ~250 |
| `src/dana_engine/batch/overlap_scorer.py` | Pairwise expert overlap scoring + greedy set-cover grouping | ~400 |
| `src/dana_engine/batch/scheduler.py` | Execute grouped batches, manage expert lifecycle | ~350 |
| `src/dana_engine/batch/adaptive_window.py` | Latency-aware batch window sizing (max 50ms, skip if <3 requests) | ~200 |
| **Subtotal** | | **~1,200** |

#### 7b. Multi-Sequence Parallel Drafting — Batching + Speculation Synergy

Draft ALL concurrent sequences in parallel using MoE-Self-Draft (top-1 = cheap). All drafts share expert loads. Verify all in one batched pass.

| Module | Description | Est. LoC |
|---|---|---|
| `src/dana_engine/speculative/multi_seq.py` | Parallel multi-sequence drafting for batched API mode | ~300 |
| `src/dana_engine/speculative/prefetch_bridge.py` | Draft router logits → prefetch commands (synergy glue) | ~250 |
| **Subtotal** | | **~550** |

#### 7c. Pipeline Orchestration — The Composition Layer

How prefetch feeds cache, cache feeds quant, quant feeds spec decode, spec decode feeds batch. This is the "magic" that makes 6 independent libraries into a 50x speedup.

| Module | Description | Est. LoC |
|---|---|---|
| `src/dana_engine/pipeline.py` | Unified engine composing all 6 products | ~500 |
| `src/dana_engine/config.py` | Engine configuration with per-pillar toggles | ~100 |
| `src/dana_engine/api/server.py` | OpenAI-compatible API server | ~300 |
| `src/dana_engine/cli.py` | CLI: `dana run --model qwen3-235b` | ~200 |
| **Subtotal** | | **~1,100** |

#### 7d. Tiny MoE Model — For Testing All Products

The shared toy model used by all products for benchmarking and integration testing. Lives in `dana-engine` because it's the integrator.

```
Tiny MoE Spec:
- 4 transformer layers, 8 experts per layer (32 total)
- Hidden dim: 512, Expert FFN: 512 → 2048 → 512
- Top-2 routing (same as DeepSeek/Qwen)
- Each expert: ~4MB FP32, ~1MB Q4, ~0.5MB Q2
- Total: ~128MB FP32 — fits in 16GB RAM
```

| Module | Description | Est. LoC |
|---|---|---|
| `src/dana_engine/model/config.py` | Model configuration dataclass | ~50 |
| `src/dana_engine/model/moe_layer.py` | Expert FFN + top-k router with load balancing | ~300 |
| `src/dana_engine/model/attention.py` | Multi-head attention | ~200 |
| `src/dana_engine/model/transformer.py` | Full transformer stack (4 layers) | ~250 |
| `src/dana_engine/naive_inference.py` | Baseline autoregressive loop (the "before" measurement) | ~200 |
| **Subtotal** | | **~1,000** |

#### 7e. Production Kernels (Future — C++/CUDA)

The OSS repos are Python/NumPy proof-of-concepts. This has optimized implementations:
- Optimized CUDA tree verification kernels
- Production attention mask construction
- Fast dequantization fused with PCIe transfer
- Router-only forward pass on GPU

#### 7f. Tests + Benchmarks

| Module | Description | Est. LoC |
|---|---|---|
| `tests/test_batch.py` | Expert-aware batching correctness | ~200 |
| `tests/test_multi_seq.py` | Multi-sequence drafting tests | ~200 |
| `tests/test_pipeline.py` | Pipeline orchestration tests | ~300 |
| `tests/test_integration.py` | End-to-end: full pipeline output matches naive baseline | ~300 |
| `benchmarks/bench_batch.py` | FIFO vs expert-aware with 50 simulated requests | ~200 |
| `benchmarks/bench_full.py` | Full pipeline ablation study (toggle each pillar) | ~400 |
| `benchmarks/report.py` | Generate markdown tables with results | ~200 |
| `colab/` | Google Colab validation with real MoE model (Phase 7) | ~800 |
| **Subtotal** | | **~2,600** |

### `dana-engine` Total: ~6,450 LoC (private code)

### Dependencies (imports the 6 libraries)

```toml
[project]
name = "dana-engine"
dependencies = [
    "moe-router-predict",
    "expert-cache",
    "tiered-tensor-store",
    "moe-quant",
    "spec-decode-tree",
    "moe-self-draft",
]
```

### Folder Structure

```
dana-engine/
├── pyproject.toml
├── README.md                        # Private — "do not distribute"
├── src/
│   └── dana_engine/
│       ├── __init__.py
│       ├── pipeline.py              # Unified engine (composes all 6)
│       ├── config.py                # Per-pillar toggles
│       ├── naive_inference.py       # Baseline (the "before")
│       ├── cli.py                   # dana run --model ...
│       ├── model/                   # Tiny MoE for testing
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── moe_layer.py
│       │   ├── attention.py
│       │   └── transformer.py
│       ├── batch/                   # PRIVATE: Expert-aware batching
│       │   ├── __init__.py
│       │   ├── expert_predictor.py
│       │   ├── overlap_scorer.py
│       │   ├── scheduler.py
│       │   └── adaptive_window.py
│       ├── speculative/             # PRIVATE: Multi-seq + prefetch bridge
│       │   ├── __init__.py
│       │   ├── multi_seq.py
│       │   └── prefetch_bridge.py
│       └── api/                     # OpenAI-compatible server
│           ├── __init__.py
│           └── server.py
├── tests/
│   ├── test_batch.py
│   ├── test_multi_seq.py
│   ├── test_pipeline.py
│   └── test_integration.py
├── benchmarks/
│   ├── bench_batch.py
│   ├── bench_full.py
│   └── report.py
├── colab/                           # GPU validation (Phase 7)
│   ├── setup.py
│   ├── real_model_bench.py
│   ├── quality_eval.py
│   └── demo.py
└── results/
    └── .gitkeep
```

---

## Combined Speedup Model

Each pillar multiplies with the others (they are largely orthogonal):

| Scenario | Library Used | Expert Load Time | Compute Efficiency | Effective TPS |
|---|---|---|---|---|
| **Naive baseline** | (none) | 100% I/O wait | 1 token/step | ~0.5 TPS |
| + Prefetching | `moe-router-predict` | ~20% I/O wait (5x) | 1 token/step | ~2.5 TPS |
| + Predictive Cache | `expert-cache` | ~8% I/O wait (2.5x) | 1 token/step | ~6 TPS |
| + Hybrid Quantization | `moe-quant` | ~3% I/O wait (2.5x) | 1 token/step | ~15 TPS |
| + Speculative Decoding | `moe-self-draft` + `spec-decode-tree` | ~3% I/O wait | ~4 tokens/step (4x) | ~60 TPS |
| + Expert-Aware Batching (10 users) | `dana-engine` (private) | shared loads (5x) | ~4 tokens/step | ~30 TPS/user |

**Single-user target:** 40-60 TPS on 2x RTX 4090 + 512GB RAM
**API target:** 20-30 TPS/user at 10 concurrent users

---

## Full Monorepo Structure

All 7 products live under `engine/` in the Dana monorepo. Each has its own `pyproject.toml`, README, tests, and benchmarks. They can be published as independent repos when ready.

```
dana/
├── DANA_ENGINE_PLAN.md              # This document
├── engine/
│   ├── moe-router-predict/          # Product 1 — Full OSS (MIT)
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── LICENSE
│   │   ├── src/moe_router_predict/
│   │   ├── tests/
│   │   └── benchmarks/
│   │
│   ├── expert-cache/                # Product 2 — Full OSS (MIT)
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── LICENSE
│   │   ├── src/expert_cache/
│   │   ├── tests/
│   │   └── benchmarks/
│   │
│   ├── tiered-tensor-store/         # Product 3 — Full OSS (MIT)
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── LICENSE
│   │   ├── src/tiered_tensor_store/
│   │   ├── tests/
│   │   └── benchmarks/
│   │
│   ├── moe-quant/                   # Product 4 — Full OSS (MIT)
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── LICENSE
│   │   ├── src/moe_quant/
│   │   ├── tests/
│   │   └── benchmarks/
│   │
│   ├── spec-decode-tree/            # Product 5 — Limited (Apache 2.0)
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── LICENSE
│   │   ├── src/spec_decode_tree/
│   │   ├── tests/
│   │   └── benchmarks/
│   │
│   ├── moe-self-draft/              # Product 6 — Limited (Apache 2.0)
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── LICENSE
│   │   ├── src/moe_self_draft/
│   │   ├── tests/
│   │   └── benchmarks/
│   │
│   └── dana-engine/                 # Product 7 — PRIVATE (Proprietary)
│       ├── pyproject.toml
│       ├── README.md
│       ├── src/dana_engine/
│       │   ├── pipeline.py          # Composes all 6 libraries
│       │   ├── config.py
│       │   ├── cli.py
│       │   ├── naive_inference.py
│       │   ├── model/               # Tiny MoE for testing
│       │   ├── batch/               # PRIVATE: Expert-aware batching
│       │   ├── speculative/         # PRIVATE: Multi-seq + prefetch bridge
│       │   └── api/                 # OpenAI-compatible server
│       ├── tests/
│       ├── benchmarks/
│       ├── colab/                   # GPU validation
│       └── results/
```

---

## Implementation Phases (Revised for Unbundled Architecture)

Each phase now produces a **shippable, independently testable product**.

### Phase 0: Tiny MoE + Baseline → `dana-engine` foundation (~2,150 LoC)

Build the toy model and naive inference inside `dana-engine`. This provides the "before" measurement and the test harness all other products use.

| Task | Product | Est. LoC |
|---|---|---|
| `model/moe_layer.py` — Expert FFN + top-k router | `dana-engine` | ~300 |
| `model/transformer.py` — Attention + MoE stack | `dana-engine` | ~250 |
| `model/config.py` — Model configuration | `dana-engine` | ~50 |
| `naive_inference.py` — Baseline autoregressive loop | `dana-engine` | ~200 |
| Storage simulation (memory/tmpfs/disk) | `dana-engine` | ~300 |
| Expert lifecycle manager | `dana-engine` | ~250 |
| Benchmarks + tests | `dana-engine` | ~600 |
| Init/config/utils | `dana-engine` | ~200 |
| **Subtotal** | | **~2,150** |

**Deliverable:** Baseline TPS. Proof that >60% of time is I/O.

---

### Phase 1: Expert Prefetching → `moe-router-predict` (~1,800 LoC)

The first standalone product. Highest-impact single optimization.

**Deliverable:** `pip install moe-router-predict` works. "Prefetching gives X.Xx speedup."

---

### Phase 2: Predictive Caching → `expert-cache` (~1,650 LoC)

Second standalone product. Depends on hints from `moe-router-predict` for predictive mode.

**Deliverable:** `pip install expert-cache` works. Cache hit rate: predictive 92% vs LRU 60%.

---

### Phase 3: Tiered Storage → `tiered-tensor-store` (~1,950 LoC)

Third standalone product. Most general — works beyond MoE.

**Deliverable:** `pip install tiered-tensor-store` works. Three-tier throughput benchmarks.

---

### Phase 4: Hybrid Quantization → `moe-quant` (~1,850 LoC)

Fourth standalone product. Integrates with `tiered-tensor-store` for tier-aware bit-widths.

**Deliverable:** `pip install moe-quant` works. "Q2 loads 4x faster with <2% divergence."

---

### Phase 5: Speculative Decoding → `spec-decode-tree` + `moe-self-draft` (~2,550 LoC)

Two limited-release products built together. Reference implementations only.

**Deliverable:** Published benchmarks. "MoE-Self-Draft: 85% acceptance. Tree: ~7 tokens/step."

---

### Phase 6: Integration → `dana-engine` private features (~4,300 LoC)

Build the private-only components: expert-aware batching, multi-sequence drafting, pipeline orchestration, API server.

**Deliverable:** Full ablation table. The "money slide."

---

### Phase 7: Colab Validation (~800 LoC)

Run on real MoE model on GPU. Lives in `dana-engine/colab/`.

**Deliverable:** Real text generation at 30+ TPS. <1% perplexity increase.

---

## LoC Summary (Unbundled)

| # | Product | Est. LoC | Release | Testable locally? |
|---|---|---|---|---|
| 1 | `moe-router-predict` | ~1,800 | Full OSS (MIT) | Yes (CPU) |
| 2 | `expert-cache` | ~1,650 | Full OSS (MIT) | Yes (CPU) |
| 3 | `tiered-tensor-store` | ~1,950 | Full OSS (MIT) | Yes (CPU) |
| 4 | `moe-quant` | ~1,850 | Full OSS (MIT) | Yes (CPU) |
| 5 | `spec-decode-tree` | ~1,450 | Limited (Apache 2.0) | Yes (CPU) |
| 6 | `moe-self-draft` | ~1,100 | Limited (Apache 2.0) | Yes (CPU) |
| 7 | `dana-engine` | ~6,450 | Private (Proprietary) | Yes (CPU) / Colab (GPU) |
| **Total** | | **~16,250** | | **95% testable locally** |

*LoC is higher than the monolithic version (~12,850) because each product has its own tests, benchmarks, and packaging overhead. The actual algorithm code is the same.*

---

## Speculative Decoding: Deep Dive on "Smarter & More"

Standard speculative decoding guesses ~4-5 tokens at ~70% acceptance = ~3 free tokens/step.

Our target: **8-12 effective tokens/step** through five enhancements:

### Enhancement 1: MoE-Self-Draft (`moe-self-draft`)
- **Standard:** Separate 1.5B draft model. Different architecture, different weights, low agreement with target.
- **Ours:** Use the same model with top-1 expert instead of top-2. Same attention, same embeddings, same router. Agreement rate jumps from ~70% to ~85%+ because the draft IS a subset of the target model.
- **Bonus:** Zero extra memory. The draft model is just a configuration flag on the main model.

### Enhancement 2: Tree Speculation (`spec-decode-tree`)
- **Standard:** Generate linear chain: t1 → t2 → t3 → t4. If t2 is wrong, t3 and t4 are wasted.
- **Ours:** Generate a tree with branching factor 2-3. Verify all leaves in one pass. Accept longest valid path.
- **Math:** Linear chain with 70% acceptance, 8 candidates = ~5.6 accepted. Tree with 70% acceptance, 8 leaf paths = ~7.2 accepted (best path is longer because you have multiple chances).

### Enhancement 3: Adaptive Draft Length (`spec-decode-tree`)
- **Standard:** Fixed N=5 always.
- **Ours:** Track rolling acceptance rate per-layer. If last 10 verifications averaged 90% acceptance → increase N to 12-16. If acceptance dropped to 50% → reduce N to 3.
- **Why it matters:** Code completion = high predictability = speculate more. Creative writing = low predictability = speculate less. Saves wasted compute.

### Enhancement 4: Prefetch-During-Draft Synergy (`dana-engine` PRIVATE)
- **Standard:** Draft finishes → load experts → verify. Sequential.
- **Ours:** While draft runs (top-1 expert, fast), we read the router logits at each layer. These logits tell us EXACTLY which top-2 experts the verification pass will need. We begin PCIe transfers immediately. By the time draft finishes, verification experts are already in VRAM.
- **Result:** Verification pass has near-zero I/O wait. The draft phase doubles as a prefetch scheduling phase.

### Enhancement 5: Parallel Multi-Sequence Drafting (`dana-engine` PRIVATE)
- **Standard:** Each user gets independent speculative decoding.
- **Ours:** In API mode with 10 concurrent users, draft ALL 10 sequences in parallel using MoE-Self-Draft (top-1 expert = cheap). All 10 drafts share expert loads (expert #5 loaded once, used for all 10 drafts). Then verify all 10 sequences in one batched pass.
- **Result:** Speculation overhead is amortized across users. Expert-aware batching makes speculation nearly free.

### Projected Speedup Stack

| Config | Tokens accepted/step | Verification cost | Effective TPS multiplier |
|---|---|---|---|
| No speculation | 1 | 1 full forward pass | 1x |
| Standard (separate draft, N=5) | ~3.5 | 1 full pass + 1 draft pass | ~2.5x |
| MoE-Self-Draft (N=5) | ~4.2 | 1 full pass + 0.5x draft (top-1 = half compute) | ~3.5x |
| + Tree speculation (N=8, branch=2) | ~6.5 | 1 full pass + 0.7x draft | ~5x |
| + Adaptive N (avg N=10) | ~8.0 | 1 full pass + 0.8x draft | ~6x |
| + Prefetch-during-draft (zero I/O wait on verify) | ~8.0 | 0.3x full pass (I/O eliminated) + 0.8x draft | ~8x |

Combined with the other pillars, this is how we hit 40-60 TPS on consumer hardware.

---

## Open-Source Release Strategy

### Phase 1: Build Everything in Monorepo (now)
All 7 products live under `engine/` in this repo. Develop, test, iterate.

### Phase 2: Extract OSS repos (when ready for public)
1. Create separate GitHub repos: `dana-ai/moe-router-predict`, `dana-ai/expert-cache`, etc.
2. Copy code from monorepo → separate repos (git filter-branch or fresh init)
3. Publish to PyPI: `pip install moe-router-predict`, `pip install expert-cache`, etc.
4. `dana-engine` stays in this private repo, installs the OSS libs as dependencies

### Phase 3: Community building
- Blog posts for each library launch
- Papers for `spec-decode-tree` and `moe-self-draft`
- HuggingFace integration examples
- Benchmark reproducibility scripts

### What's Never Released
1. **`dana-engine/batch/`** — Expert-aware request batching (the API-serving moat)
2. **`dana-engine/speculative/multi_seq.py`** — Multi-sequence parallel drafting
3. **`dana-engine/speculative/prefetch_bridge.py`** — Prefetch-during-draft synergy glue
4. **`dana-engine/pipeline.py`** — The full orchestration (how all 6 compose)
5. **Production C++/CUDA kernels** — OSS repos have Python/NumPy only
6. **Tuned parameters** — Optimal batch windows, cache thresholds, draft lengths

---

## Development Approach: Start Small, Prove Everything

### Principle: Every Innovation is Testable on CPU

We do NOT need GPUs or large models to prove the algorithms work. We build a **tiny MoE model** in `dana-engine` that has the same architectural properties as real models:

```
Tiny MoE Spec:
- 4 transformer layers
- 8 experts per layer (32 total)
- Hidden dim: 512
- Expert FFN: 512 → 2048 → 512
- Top-2 routing (same as DeepSeek/Qwen)
- Each expert: ~4MB FP32, ~1MB Q4, ~0.5MB Q2
- Total: ~128MB FP32
- Fits easily in 16GB RAM
```

Storage tiers simulated with real I/O:
- "VRAM" = numpy arrays in memory (instant access)
- "RAM" = numpy arrays loaded from tmpfs
- "SSD" = numpy arrays loaded from disk with calibrated sleep to match real NVMe:PCIe:HBM ratios

**The speedup ratios are real regardless of model size.** If prefetching gives 5x on tiny model, it gives ~5x on real model. The algorithms are the same — only the absolute bytes transferred change.

### What Only Needs Real Hardware (Google Colab)
1. **Quality validation:** Does Q2 quantization preserve output quality on real text generation?
2. **CUDA kernel profiling:** Are our custom kernels actually faster than PyTorch defaults?
3. **End-to-end demo:** Run the engine on a real model, produce real text, measure real TPS.

Everything else — algorithmic proof, speedup ratios, cache hit rates, batching efficiency, spec decode acceptance rates — is testable right here.

---

## Success Metrics (What We'll Prove in Each Phase)

| Phase | Product | Metric | Target | How Measured |
|---|---|---|---|---|
| 0 | `dana-engine` | Baseline TPS | ~0.5 TPS (intentionally slow) | Wall clock / tokens |
| 0 | `dana-engine` | I/O bottleneck proof | >60% time in expert loading | Time profiling |
| 1 | `moe-router-predict` | Prefetch speedup | 4-6x over baseline | Wall clock comparison |
| 2 | `expert-cache` | Cache hit rate | >90% (predictive) vs ~60% (LRU) | Hit/miss counter |
| 3 | `tiered-tensor-store` | Tier throughput | VRAM > RAM > SSD with correct ratios | Throughput measurement |
| 4 | `moe-quant` | Load speed at Q2 | 4x faster than FP32 | Transfer time |
| 4 | `moe-quant` | Quality at Q2 | <2% output divergence | Cosine similarity |
| 5 | `spec-decode-tree` | Tokens accepted/step | >6 (tree) vs ~3.5 (standard) | Acceptance counter |
| 5 | `moe-self-draft` | Draft agreement | >85% (self-draft) vs ~70% (separate) | Acceptance rate |
| 6 | `dana-engine` | Expert loads (10 users) | <100 (batched) vs ~500 (FIFO) | Load counter |
| 6 | `dana-engine` | Full pipeline TPS | >50x over baseline | Wall clock, end-to-end |
| 7 | `dana-engine` | Real model TPS | 30+ TPS on Colab T4/A100 | Real generation |
| 7 | `dana-engine` | Real model quality | <1% perplexity increase | Evaluation script |

---

## Technology Stack

### Proof-of-Concept (All 7 products)
- **Python 3.11+**
- **NumPy** — tensor operations, quantization math
- **asyncio** — async expert loading, prefetch pipeline
- **threading** — parallel I/O simulation
- **struct/mmap** — binary expert storage, memory-mapped files
- **time/cProfile** — benchmarking and profiling
- Zero external ML frameworks needed. Pure Python + NumPy.

### Production (`dana-engine` only, Phase 7+)
- **C++17** — core engine
- **CUDA 12** — GPU kernels (dequantization, attention, router)
- **Python bindings (pybind11)** — API layer
- **PyTorch** — model loading compatibility only

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Prefetch predictions are inaccurate | Low hit rate, wasted bandwidth | Router-only forward pass uses exact same router weights. Accuracy should be >80%. Fall back to on-demand loading for misses. |
| Q2 quantization degrades quality too much | Unusable outputs from cold experts | Per-expert sensitivity profiling. Sensitive experts stay at Q4. Only robust experts get Q2. |
| Tree speculation overhead exceeds benefit | Slower than linear speculation | Adaptive: fall back to linear when branching doesn't help (e.g., deterministic outputs). |
| Expert-aware batching adds too much latency | Users wait too long for batch formation | Adaptive batch window: max 50ms wait. If <3 requests in window, skip grouping and process immediately. |
| Tiny model results don't transfer to real model | Proof doesn't hold at scale | I/O ratios and algorithmic properties are scale-invariant. Phase 7 (Colab) validates on real model. |
| Someone publishes MoE-Self-Draft before us | Lose narrative ownership | Publish concept + benchmarks early (limited release). Establish priority. |
| OSS libraries get forked and used against us | Competitor builds their own engine | MIT/Apache for libraries is intentional — the integration layer is the moat, not the building blocks. |
