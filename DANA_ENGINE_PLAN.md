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

> **MoE models activate only 5-15% of parameters per token, yet current inference engines treat them like dense models.** Dana exploits the sparse activation pattern with five synergistic innovations that turn I/O-bound inference into compute-bound inference on consumer hardware.

**Target:** Run Qwen3-235B-MoE (or DeepSeek-V3-671B) at **30+ tokens/sec** on a $20K workstation (2x RTX 4090 + 512GB DDR5 RAM) instead of $200K+ in H100s.

---

## The Six Pillars

### Pillar 1: Asynchronous Expert Prefetching
**Problem:** When the router selects expert #47, the engine stalls waiting for it to load from RAM to VRAM across PCIe (~32GB/s).

**Solution:** Use a lightweight lookahead (draft model or router-only forward pass) to predict which experts the next N tokens will need, and begin DMA transfers *before* compute needs them.

**Key metric:** PCIe idle time reduced from ~60% to <10%.

| Subcomponent | Description | Est. LoC |
|---|---|---|
| Router lookahead predictor | Runs router MLP on draft hidden states to predict expert IDs N tokens ahead | ~300 |
| Async DMA transfer manager | Manages PCIe transfers with double-buffering (compute on buffer A while loading into buffer B) | ~500 |
| Expert residency tracker | Tracks which experts are currently in VRAM, in-flight, or cold in RAM/SSD | ~200 |
| Prefetch queue + priority scheduler | Orders prefetch requests by predicted need-time, cancels stale ones | ~300 |
| **Subtotal** | | **~1,300** |

---

### Pillar 2: Predictive Expert Caching
**Problem:** LRU cache evicts experts that happen to be old but are frequently used. Expert popularity follows a power-law distribution — some experts are "hot" across most inputs.

**Solution:** Frequency-aware + predictive caching. Track per-expert activation frequency across a sliding window. Retain high-frequency experts permanently in VRAM. Predict upcoming expert needs based on token context.

**Key metric:** Cache hit rate from ~60% (LRU) to 90%+ (predictive).

| Subcomponent | Description | Est. LoC |
|---|---|---|
| Expert frequency profiler | Sliding-window activation counter per expert, per layer | ~200 |
| Hot/warm/cold classifier | Classifies experts into tiers based on activation frequency. Hot = pinned in VRAM, warm = RAM, cold = SSD | ~150 |
| Predictive eviction policy | Considers both recency AND predicted future need (from router lookahead) before evicting | ~300 |
| VRAM budget manager | Enforces VRAM budget, triggers eviction when approaching limit | ~200 |
| Cache analytics + self-tuning | Tracks hit/miss rates, auto-adjusts tier thresholds | ~200 |
| **Subtotal** | | **~1,050** |

---

### Pillar 3: Expert-Aware Request Batching
**Problem:** With N concurrent users, naive batching processes requests in FIFO order. Each request activates different experts, causing constant swapping. Serving 10 users = loading 10x different expert sets.

**Solution:** Group requests that share predicted expert activations. Load expert once, process all requests that need it. This is the **closed-source moat** — the single-user OSS version doesn't need this, but it's critical for API serving.

**Key metric:** Total expert loads reduced by 5-10x under concurrent load.

| Subcomponent | Description | Est. LoC |
|---|---|---|
| Request expert predictor | For each queued request, predict which experts it will need (via draft or router-only pass) | ~250 |
| Overlap scorer + grouper | Score pairwise expert overlap between queued requests, form optimal groups using greedy set-cover | ~400 |
| Batch execution scheduler | Execute grouped batches: load shared experts once, run all grouped requests, then move to next group | ~350 |
| Adaptive batch window | Dynamically adjust how long to wait for more requests vs. latency SLA | ~200 |
| **Subtotal** | | **~1,200** |

---

### Pillar 4: Hybrid Quantization
**Problem:** Experts in RAM are large (FP16/FP32). Transferring them across PCIe is the bottleneck. Standard uniform quantization (all Q4) wastes quality on hot experts and wastes bandwidth on cold ones.

**Solution:** Tier-aware quantization:
- **Hot experts (pinned in VRAM):** FP16 or Q8 — maximum quality, already resident, no transfer cost.
- **Warm experts (in RAM):** Q4 — good balance of quality and transfer speed.
- **Cold experts (on SSD):** Q2 or Q3 — aggressive compression. These are rarely activated; quality loss is amortized across their low usage.

**Key metric:** 2-4x faster expert loading with <1% quality degradation on benchmarks.

| Subcomponent | Description | Est. LoC |
|---|---|---|
| Per-expert quantizer | Quantize individual experts to target bit-width (Q2/Q3/Q4/Q8) with calibration data | ~400 |
| Quality-aware tier assigner | Profile each expert's sensitivity to quantization (measure output divergence), assign bit-width to minimize total quality loss under a bandwidth budget | ~300 |
| Fast dequantizer | Optimized dequantization kernels that run during/after PCIe transfer (overlap dequant with transfer) | ~500 |
| Dynamic re-quantization | If an expert gets promoted from cold→warm, re-quantize to higher precision | ~200 |
| **Subtotal** | | **~1,400** |

---

### Pillar 5: Speculative Decoding (MoE-Optimized)

This is not standard speculative decoding. We push it further with MoE-specific innovations.

**Standard spec. decode:** Small draft model generates N candidate tokens, large model verifies all N in one forward pass. Accepted tokens = free speedup (skip N-1 autoregressive steps).

**Our MoE-specific enhancements:**

#### 5a. MoE-Self-Draft (No Separate Draft Model)
Instead of a separate 1.5B draft model, use the **main model with only 1 expert per layer** (instead of top-2). This means:
- Zero additional memory for draft model
- Draft shares the same router — its expert predictions directly inform prefetching for verification
- Quality of draft is higher than a tiny separate model (same attention weights, same embeddings)

#### 5b. Router-Predictive Prefetch During Drafting
While the draft pass runs (which is fast — 1 expert per layer), we already know the router scores for ALL experts at each layer. Before verification even starts, we:
1. Read the router logits from the draft pass
2. Identify top-2 experts per layer for each drafted token
3. Begin prefetching ALL needed experts for the full verification pass
4. By the time draft completes, verification experts are already in VRAM

This turns the draft phase into a **free prefetch window**.

#### 5c. Tree-Based Speculation
Instead of drafting one linear sequence of N tokens, draft a **tree** of candidates:
```
         token_1
        /       \
   token_2a   token_2b
   /     \       |
 t_3a   t_3b   t_3c
```
Verify all branches in a single batched forward pass. Accept the longest valid branch. This increases acceptance rate dramatically for branching-heavy outputs (code, reasoning).

#### 5d. Adaptive Draft Length
Track rolling acceptance rate. If the model is "easy" (high acceptance), speculate more tokens (12-16). If "hard" (low acceptance), speculate fewer (3-4). This prevents wasting compute on rejected drafts.

#### 5e. Parallel Multi-Sequence Drafting
For API batching scenarios: draft multiple sequences in parallel using MoE-Self-Draft (cheap — 1 expert per layer). All drafts share the same expert loads. Then verify all sequences in one batched verification pass. This combines batching + speculation synergistically.

| Subcomponent | Description | Est. LoC |
|---|---|---|
| MoE-Self-Draft engine | Forward pass using top-1 expert per layer (draft mode) | ~400 |
| Router logit extractor | Capture full router probability distribution during draft pass | ~150 |
| Prefetch-during-draft pipeline | Convert draft router logits into prefetch commands, execute async | ~300 |
| Verification pass with pre-loaded experts | Standard draft-verify but experts are already resident from prefetch | ~350 |
| Tree speculation engine | Generate candidate tree, flatten for batched verification, trace longest accepted path | ~600 |
| Adaptive draft length controller | Online learning of acceptance rate, dynamic N adjustment | ~200 |
| Multi-sequence parallel drafter | Batch multiple user drafts together, share expert loads | ~300 |
| Acceptance rate tracker + analytics | Per-layer, per-expert acceptance statistics for tuning | ~150 |
| **Subtotal** | | **~2,450** |

---

### Pillar 6: Tiered Storage Engine
**Problem:** Not all 128 experts fit in VRAM (limited). Not all fit in RAM (possible but expensive). SSD is cheap but slower.

**Solution:** Three-tier storage hierarchy with intelligent placement:

```
┌─────────────────────────┐
│   VRAM (24-80GB)        │  ← Hot experts: pinned, FP16/Q8
│   Latency: 0 (resident) │
├─────────────────────────┤
│   System RAM (512GB)    │  ← Warm experts: Q4, prefetchable
│   Latency: ~1ms (PCIe)  │
├─────────────────────────┤
│   NVMe SSD (2-4TB)     │  ← Cold experts: Q2, background load
│   Latency: ~5ms (NVMe)  │
└─────────────────────────┘
```

| Subcomponent | Description | Est. LoC |
|---|---|---|
| Storage tier abstraction | Unified interface for VRAM/RAM/SSD expert storage | ~300 |
| Expert placement optimizer | Assign experts to tiers based on frequency profile + capacity constraints (ILP or greedy) | ~400 |
| Background promotion/demotion | Move experts between tiers based on changing access patterns | ~300 |
| SSD direct I/O manager | Bypass OS page cache for predictable latency from NVMe | ~250 |
| Memory-mapped expert pool | mmap for RAM-tier experts with custom fault handling | ~200 |
| **Subtotal** | | **~1,450** |

---

## Total Estimated LoC (Engine Core)

| Pillar | LoC |
|---|---|
| 1. Expert Prefetching | ~1,300 |
| 2. Predictive Caching | ~1,050 |
| 3. Expert-Aware Batching | ~1,200 |
| 4. Hybrid Quantization | ~1,400 |
| 5. Speculative Decoding (MoE-enhanced) | ~2,450 |
| 6. Tiered Storage Engine | ~1,450 |
| **Shared infrastructure** (model loader, tokenizer, attention pass, config, logging, tests) | **~2,000** |
| **Test suite + benchmarks** | **~1,500** |
| **Total Engine** | **~12,350** |

The proof-of-concept (Python, testable in Claude Code Web) will be ~60% of this. The production C++/CUDA rewrite will be ~2x the LoC.

---

## Combined Speedup Model

Each pillar multiplies with the others (they are largely orthogonal):

| Scenario | Expert Load Time | Compute Efficiency | Effective TPS |
|---|---|---|---|
| **Naive baseline** (no optimizations) | 100% I/O wait | 1 token/step | ~0.5 TPS |
| + Prefetching | ~20% I/O wait (5x) | 1 token/step | ~2.5 TPS |
| + Predictive Cache | ~8% I/O wait (2.5x more) | 1 token/step | ~6 TPS |
| + Hybrid Quantization | ~3% I/O wait (2.5x more) | 1 token/step | ~15 TPS |
| + Speculative Decoding | ~3% I/O wait | ~4 tokens/step (4x) | ~60 TPS |
| + Expert-Aware Batching (10 users) | shared loads (5x efficiency) | ~4 tokens/step | ~30 TPS/user |

**Single-user target:** 40-60 TPS on 2x RTX 4090 + 512GB RAM
**API target:** 20-30 TPS/user at 10 concurrent users

These are achievable because MoE models activate so few parameters per token — once you eliminate the I/O bottleneck, the actual compute per token is tiny (22B active params vs 235B total).

---

## Development Approach: Start Small, Prove Everything

### Principle: Every Innovation is Testable on CPU in Claude Code Web

We do NOT need GPUs or large models to prove the algorithms work. We build a **tiny MoE model** that has the same architectural properties as real models:

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

## Implementation Phases

### Phase 0: Tiny MoE + Baseline (~2,500 LoC)
Build the toy model and naive inference. This is our "before" measurement.

| Task | Description | Est. LoC |
|---|---|---|
| `model/moe_layer.py` | Expert FFN + top-k router with load balancing | ~300 |
| `model/transformer.py` | Attention + MoE layer stack (4 layers) | ~250 |
| `model/config.py` | Model configuration dataclass | ~50 |
| `engine/naive_inference.py` | Simple autoregressive loop, all experts in memory | ~200 |
| `engine/storage.py` | Tiered storage abstraction (memory/tmpfs/disk) with realistic latency | ~300 |
| `engine/expert_manager.py` | Load/unload experts between tiers | ~250 |
| `benchmarks/bench_baseline.py` | Measure naive TPS, expert load counts, I/O wait time | ~200 |
| `tests/` | Unit tests for model forward pass, router, storage | ~400 |
| Init/config/utils | Setup, logging, helpers | ~200 |
| **Subtotal** | | **~2,150** |

**Deliverable:** Baseline TPS number. Proof that I/O is the bottleneck (show that >60% of time is spent loading experts).

---

### Phase 1: Expert Prefetching (~1,500 LoC)
The first and highest-impact optimization.

| Task | Description | Est. LoC |
|---|---|---|
| `engine/prefetch/predictor.py` | Router-only forward pass to predict next-N experts | ~300 |
| `engine/prefetch/async_loader.py` | Async expert loading with double-buffering | ~400 |
| `engine/prefetch/scheduler.py` | Prefetch queue with priority and cancellation | ~300 |
| `benchmarks/bench_prefetch.py` | Before/after comparison, I/O overlap measurement | ~200 |
| `tests/test_prefetch.py` | Correctness tests (same output with/without prefetch) | ~300 |
| **Subtotal** | | **~1,500** |

**Deliverable:** "Prefetching gives X.Xx speedup" with real wall-clock measurements.

---

### Phase 2: Predictive Caching (~1,200 LoC)

| Task | Description | Est. LoC |
|---|---|---|
| `engine/cache/lru_cache.py` | Baseline LRU expert cache | ~150 |
| `engine/cache/frequency_cache.py` | Frequency-aware cache (LFU + recency) | ~200 |
| `engine/cache/predictive_cache.py` | Uses router lookahead to predict + pre-cache | ~350 |
| `engine/cache/budget_manager.py` | VRAM budget enforcement + eviction | ~200 |
| `benchmarks/bench_cache.py` | Compare LRU vs frequency vs predictive on 1000-token generation | ~200 |
| `tests/test_cache.py` | Cache correctness + eviction policy tests | ~200 |
| **Subtotal** | | **~1,300** |

**Deliverable:** Cache hit rates for each strategy. Proof that predictive > frequency > LRU.

---

### Phase 3: Hybrid Quantization (~1,500 LoC)

| Task | Description | Est. LoC |
|---|---|---|
| `engine/quant/quantize.py` | FP32 → Q8/Q4/Q2 per-expert quantization with calibration | ~400 |
| `engine/quant/dequantize.py` | Fast dequantization (vectorized numpy, later CUDA) | ~300 |
| `engine/quant/sensitivity.py` | Measure per-expert quality degradation at each bit-width | ~250 |
| `engine/quant/tier_assigner.py` | Assign bit-widths to minimize quality loss under bandwidth budget | ~200 |
| `benchmarks/bench_quant.py` | Load speed + output divergence at each bit-width | ~200 |
| `tests/test_quant.py` | Round-trip accuracy, edge cases | ~200 |
| **Subtotal** | | **~1,550** |

**Deliverable:** "Q2 loads 4x faster with X% output divergence vs FP32."

---

### Phase 4: Speculative Decoding (~2,500 LoC)

| Task | Description | Est. LoC |
|---|---|---|
| `engine/speculative/self_draft.py` | MoE-Self-Draft: top-1 expert per layer forward pass | ~350 |
| `engine/speculative/verify.py` | Batched verification pass with pre-loaded experts | ~300 |
| `engine/speculative/tree_spec.py` | Tree-based candidate generation + flattened verification | ~500 |
| `engine/speculative/adaptive.py` | Dynamic draft length based on rolling acceptance rate | ~200 |
| `engine/speculative/prefetch_bridge.py` | Extract router logits from draft → feed into prefetch pipeline | ~250 |
| `engine/speculative/multi_seq.py` | Parallel multi-sequence drafting for batched API mode | ~300 |
| `engine/speculative/acceptance.py` | Acceptance rate tracker + per-layer analytics | ~150 |
| `benchmarks/bench_speculative.py` | Acceptance rate, effective tokens/step, TPS improvement | ~250 |
| `tests/test_speculative.py` | Output equivalence (spec decode must match autoregressive) | ~300 |
| **Subtotal** | | **~2,600** |

**Deliverable:** "MoE-Self-Draft achieves X.X tokens/step average. Tree speculation: X.X tokens/step. Combined with prefetch: Y TPS."

---

### Phase 5: Expert-Aware Batching (~1,300 LoC)

| Task | Description | Est. LoC |
|---|---|---|
| `engine/batch/expert_predictor.py` | Predict expert needs per request (reuse draft pass) | ~200 |
| `engine/batch/overlap_scorer.py` | Pairwise expert overlap scoring + greedy grouping | ~350 |
| `engine/batch/scheduler.py` | Execute grouped batches, manage expert lifecycle | ~350 |
| `engine/batch/adaptive_window.py` | Latency-aware batch window sizing | ~150 |
| `benchmarks/bench_batch.py` | Compare FIFO vs expert-aware with 50 simulated requests | ~200 |
| `tests/test_batch.py` | Grouping correctness, latency SLA compliance | ~200 |
| **Subtotal** | | **~1,450** |

**Deliverable:** "Expert-aware batching reduces total expert loads by X.Xx vs FIFO at N concurrent requests."

---

### Phase 6: Full Pipeline Integration (~1,500 LoC)

| Task | Description | Est. LoC |
|---|---|---|
| `engine/pipeline.py` | Unified engine combining all 6 pillars | ~500 |
| `engine/config.py` | Engine configuration with per-pillar toggles | ~100 |
| `benchmarks/bench_full.py` | Full pipeline vs naive baseline, ablation study (toggle each pillar) | ~400 |
| `benchmarks/report.py` | Generate markdown report with tables + results | ~200 |
| `tests/test_integration.py` | End-to-end correctness: full pipeline output matches naive | ~300 |
| **Subtotal** | | **~1,500** |

**Deliverable:** Complete ablation table showing individual and combined contribution of each pillar. The "money slide" for investors.

---

### Phase 7: Colab Validation (Real Model) (~800 LoC)

This is the only phase requiring GPU. Run on Google Colab with a real (small) MoE model.

| Task | Description | Est. LoC |
|---|---|---|
| `colab/setup.py` | Install deps, download small MoE model (Switch Transformer base or Mixtral-like) | ~100 |
| `colab/real_model_bench.py` | Run full pipeline on real model, measure real TPS | ~300 |
| `colab/quality_eval.py` | Compare output quality: naive vs quantized vs speculative | ~200 |
| `colab/demo.py` | Interactive demo: type a prompt, see inference with live stats | ~200 |
| **Subtotal** | | **~800** |

**Deliverable:** Real text generation at measured TPS. Quality scores (perplexity, task accuracy) confirming no degradation.

---

## Project Structure

```
dana/
├── DANA_ENGINE_PLAN.md          # This document
├── model/                        # Tiny MoE model definition
│   ├── __init__.py
│   ├── config.py                 # Model configuration
│   ├── moe_layer.py             # Expert FFN + top-k router
│   ├── attention.py             # Multi-head attention
│   └── transformer.py           # Full transformer stack
├── engine/                       # The six pillars
│   ├── __init__.py
│   ├── pipeline.py              # Unified engine (Phase 6)
│   ├── config.py                # Engine configuration
│   ├── naive_inference.py       # Baseline (Phase 0)
│   ├── storage.py               # Tiered storage abstraction
│   ├── expert_manager.py        # Expert lifecycle management
│   ├── prefetch/                # Pillar 1
│   │   ├── predictor.py
│   │   ├── async_loader.py
│   │   └── scheduler.py
│   ├── cache/                   # Pillar 2
│   │   ├── lru_cache.py
│   │   ├── frequency_cache.py
│   │   ├── predictive_cache.py
│   │   └── budget_manager.py
│   ├── batch/                   # Pillar 3
│   │   ├── expert_predictor.py
│   │   ├── overlap_scorer.py
│   │   ├── scheduler.py
│   │   └── adaptive_window.py
│   ├── quant/                   # Pillar 4
│   │   ├── quantize.py
│   │   ├── dequantize.py
│   │   ├── sensitivity.py
│   │   └── tier_assigner.py
│   ├── speculative/             # Pillar 5
│   │   ├── self_draft.py
│   │   ├── verify.py
│   │   ├── tree_spec.py
│   │   ├── adaptive.py
│   │   ├── prefetch_bridge.py
│   │   ├── multi_seq.py
│   │   └── acceptance.py
│   └── tiered_storage/          # Pillar 6
│       ├── tier_manager.py
│       ├── placement_optimizer.py
│       └── ssd_direct.py
├── benchmarks/                   # Proof benchmarks
│   ├── bench_baseline.py
│   ├── bench_prefetch.py
│   ├── bench_cache.py
│   ├── bench_quant.py
│   ├── bench_speculative.py
│   ├── bench_batch.py
│   ├── bench_full.py
│   └── report.py
├── tests/                        # Test suite
│   ├── test_model.py
│   ├── test_prefetch.py
│   ├── test_cache.py
│   ├── test_quant.py
│   ├── test_speculative.py
│   ├── test_batch.py
│   └── test_integration.py
├── colab/                        # Google Colab notebooks (Phase 7)
│   ├── setup.py
│   ├── real_model_bench.py
│   ├── quality_eval.py
│   └── demo.py
└── results/                      # Generated benchmark results
    └── .gitkeep
```

---

## LoC Summary

| Phase | Description | Est. LoC | Testable in Claude Code Web? |
|---|---|---|---|
| 0 | Tiny MoE + Baseline | ~2,150 | Yes (CPU) |
| 1 | Expert Prefetching | ~1,500 | Yes (CPU) |
| 2 | Predictive Caching | ~1,300 | Yes (CPU) |
| 3 | Hybrid Quantization | ~1,550 | Yes (CPU) |
| 4 | Speculative Decoding | ~2,600 | Yes (CPU) |
| 5 | Expert-Aware Batching | ~1,450 | Yes (CPU) |
| 6 | Full Pipeline Integration | ~1,500 | Yes (CPU) |
| 7 | Colab Real Model Validation | ~800 | Google Colab (GPU) |
| **Total** | | **~12,850** | **93% testable locally** |

---

## Speculative Decoding: Deep Dive on "Smarter & More"

Standard speculative decoding guesses ~4-5 tokens at ~70% acceptance = ~3 free tokens/step.

Our target: **8-12 effective tokens/step** through five enhancements:

### Enhancement 1: MoE-Self-Draft
- **Standard:** Separate 1.5B draft model. Different architecture, different weights, low agreement with target.
- **Ours:** Use the same model with top-1 expert instead of top-2. Same attention, same embeddings, same router. Agreement rate jumps from ~70% to ~85%+ because the draft IS a subset of the target model.
- **Bonus:** Zero extra memory. The draft model is just a configuration flag on the main model.

### Enhancement 2: Tree Speculation
- **Standard:** Generate linear chain: t1 → t2 → t3 → t4. If t2 is wrong, t3 and t4 are wasted.
- **Ours:** Generate a tree with branching factor 2-3. Verify all leaves in one pass. Accept longest valid path.
- **Math:** Linear chain with 70% acceptance, 8 candidates = ~5.6 accepted. Tree with 70% acceptance, 8 leaf paths = ~7.2 accepted (best path is longer because you have multiple chances).

### Enhancement 3: Adaptive Draft Length
- **Standard:** Fixed N=5 always.
- **Ours:** Track rolling acceptance rate per-layer. If last 10 verifications averaged 90% acceptance → increase N to 12-16. If acceptance dropped to 50% → reduce N to 3.
- **Why it matters:** Code completion = high predictability = speculate more. Creative writing = low predictability = speculate less. Saves wasted compute.

### Enhancement 4: Prefetch-During-Draft Synergy
- **Standard:** Draft finishes → load experts → verify. Sequential.
- **Ours:** While draft runs (top-1 expert, fast), we read the router logits at each layer. These logits tell us EXACTLY which top-2 experts the verification pass will need. We begin PCIe transfers immediately. By the time draft finishes, verification experts are already in VRAM.
- **Result:** Verification pass has near-zero I/O wait. The draft phase doubles as a prefetch scheduling phase.

### Enhancement 5: Parallel Multi-Sequence Drafting (API Mode)
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

## Open-Source Strategy

### What's Open (AGPL-3.0)
- Pillars 1, 2, 4, 5, 6 (single-user mode)
- Tiny model + all benchmarks
- CLI: `dana run --model qwen3-235b`

### What's Closed (Commercial License)
- **Pillar 3: Expert-Aware Request Batching** — useless for single-user, critical for API
- **Multi-sequence parallel drafting** (Enhancement 5 of speculative decoding)
- Managed deployment tooling

### Why This Split Works
- Single-user open-source goes viral: "Run 235B on your gaming PC at 40 TPS"
- Competitors can't use it for API serving (AGPL + no batching = single-user only)
- Enterprise customers need batching → commercial license

---

## Success Metrics (What We'll Prove in Each Phase)

| Phase | Metric | Target | How Measured |
|---|---|---|---|
| 0 | Baseline TPS | ~0.5 TPS (intentionally slow) | Wall clock / tokens generated |
| 0 | I/O bottleneck proof | >60% time in expert loading | Time breakdown profiling |
| 1 | Prefetch speedup | 4-6x over baseline | Wall clock comparison |
| 2 | Cache hit rate | >90% (predictive) vs ~60% (LRU) | Hit/miss counter |
| 3 | Load speed at Q2 | 4x faster than FP32 | Transfer time measurement |
| 3 | Quality at Q2 | <2% output divergence | Cosine similarity of outputs |
| 4 | Tokens accepted/step | >6 (tree) vs ~3.5 (standard) | Acceptance counter |
| 4 | Spec decode TPS multiplier | >5x | Wall clock comparison |
| 5 | Expert loads (10 users) | <100 (batched) vs ~500 (FIFO) | Load counter |
| 6 | Full pipeline TPS | >50x over baseline | Wall clock, end-to-end |
| 7 | Real model TPS | 30+ TPS on Colab T4/A100 | Real generation |
| 7 | Real model quality | <1% perplexity increase | Evaluation script |

---

## Technology Stack

### Proof-of-Concept (Phases 0-6, Claude Code Web)
- **Python 3.11+**
- **NumPy** — tensor operations, quantization math
- **asyncio** — async expert loading, prefetch pipeline
- **threading** — parallel I/O simulation
- **struct/mmap** — binary expert storage, memory-mapped files
- **time/cProfile** — benchmarking and profiling
- Zero external ML frameworks needed. Pure Python + NumPy.

### Production (Phase 7+, target)
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
