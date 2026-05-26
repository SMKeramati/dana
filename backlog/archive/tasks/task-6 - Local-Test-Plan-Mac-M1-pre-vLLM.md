---
id: TASK-6
title: 'Local Test Plan (Mac M1, pre-vLLM)'
status: Done
assignee: []
created_date: '2026-05-26 06:33'
labels:
  - archive
  - strategy
dependencies: []
ordinal: 6000
---

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
# Local Test Plan — Mac M1 / 16GB / 50GB free

> **Goal**: Validate as much of Dana engine as possible on M1 before paying for cloud GPU.
> Every bug found here is GPU-rent money saved.
> **Reproducibility**: `uv + pyproject.toml + uv.lock` — single source of truth for Mac (MPS) and Linux (CUDA).
> **Phases**: this doc covers infrastructure tests (L1–L5) used inside both phases. See [PHASE_1_QWEN35.md](PHASE_1_QWEN35.md) first, then [PHASE_2_TARGETS.md](PHASE_2_TARGETS.md) for production targets.

---

## TL;DR

Three layers of local testing, then gate to GPU:

| Layer | Runtime | What's validated | Time |
|---|---|---|---|
| **L1 — Correctness** | uv venv, CPU only | All 6 pillar unit + integration tests with mock pipelines | 1 day |
| **L2 — Real Inference** | uv venv, native MPS | Real models up to ~8B BF16 / 8B Int4; engine device-portability | 1 day |
| **L3 — MoE via mmap** | `llama.cpp` + Metal | Qwen3.6-35B-A3B end-to-end via page-on-demand | half day |
| **L4 — Architecture coverage** | mocks + small variants | Validate engine against all 3 production architectures | half day |
| **L5 — Gate check** | checklist | Confirm before renting GPU | 1 hour |

Pass L1–L5 → rent **4 hours of GPU**, not 10.

---

## Hard constraints on M1

```
Total RAM:     16 GB unified
macOS base:    ~4 GB
Native usable: ~10-12 GB
SSD free:      50 GB
GPU:           Metal/MPS (native only — NOT accessible from Docker)
CUDA:          none
```

**Why uv instead of Docker:** Docker on Apple Silicon cannot access MPS. We lose
real GPU testing if we containerize. Instead we use `uv` to lock dependencies
deterministically — the same `pyproject.toml + uv.lock` reproduces the env on
CI/Linux/cloud-GPU later.

**What can't be tested locally (GPU-only):**
1. CUDA kernels (GPTQ kernel, AWQ kernel, vLLM)
2. PCIe bandwidth on prefetch (Pillar 1's hot path)
3. VRAM fragmentation under load
4. Real wall-clock speedups (MTP, spec decode, batching)
5. The "30+ tok/s on 235B" claim

These five items define what the **first hour of paid GPU time** must do.

---

## Production targets recap

We test against architectural surrogates. See [PHASE_2_TARGETS.md](PHASE_2_TARGETS.md) for
the full mapping. Summary:

| Production tier | Architecture | M1 local surrogate |
|---|---|---|
| SMALL (Qwen3.6-35B-A3B) | `qwen3_5_moe` | Same model, llama.cpp Q4_K_M, mmap |
| MEDIUM (Qwen3-Coder-Next 80B) | `qwen3_next` | Arch-mocked or smaller Qwen3 variant |
| LARGE (DeepSeek-V4-Flash 158B) | `deepseek_v4` | DeepSeek-V2-Lite or arch-mocked |
| HUGE (Kimi-K2.6 ~1T) | `kimi_k25` | Arch-only mock — no small Kimi variant exists |

The engine must abstract over all four router/expert layouts.

---

## Phase 0 — Setup (1-2 hours)

### 0.1 — Install toolchain
```bash
# uv (modern Python project manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# llama.cpp with Metal backend (for L3)
brew install llama.cpp

# HF CLI
curl -LsSf https://hf.co/cli/install.sh | bash -s
hf auth login
```

### 0.2 — Initialize project env
```bash
cd dana/
uv init --name dana-local-test --python 3.11
uv add torch transformers accelerate sentencepiece numpy pytest pytest-benchmark
uv add --dev ruff mypy

# Install all engine packages in editable mode
for pkg in tiered-tensor-store expert-cache moe-quant moe-router-predict spec-decode-tree moe-self-draft dana-engine; do
  uv pip install -e "engine/$pkg[test]"
done

# Lock for reproducibility
uv lock
git add pyproject.toml uv.lock
```

This `uv.lock` is what makes it reproducible on cloud GPU — same packages, same versions.

### 0.3 — Confirm MPS available
```bash
uv run python -c "import torch; assert torch.backends.mps.is_available(); print('MPS ready')"
```

### 0.4 — Pre-download models in background
~25GB total, ~30-60 min depending on bandwidth.
```bash
# Smallest first — used everywhere
hf download Qwen/Qwen3-0.6B           --local-dir models/qwen3-0.6b &
hf download Qwen/Qwen3-1.7B           --local-dir models/qwen3-1.7b &
hf download unsloth/Qwen3-4B-GGUF     --local-dir models/qwen3-4b-gguf --include "*Q4_K_M*" &
hf download unsloth/Qwen3-8B-GGUF     --local-dir models/qwen3-8b-gguf --include "*Q4_K_M*" &

# Production SMALL target — fits via mmap
hf download unsloth/Qwen3.6-35B-A3B-GGUF --local-dir models/qwen3.6-35b-moe --include "*Q4_K_M*" &

wait
du -sh models/*
```

---

## Phase L1 — Correctness via uv venv (CPU only) — Day 1

### L1.1 — Pillar unit tests (all 6 packages)

```bash
# One command runs every test in every package
uv run pytest engine/ -v --tb=short
```

If a specific pillar fails, isolate:
```bash
uv run pytest engine/tiered-tensor-store/tests -v
uv run pytest engine/expert-cache/tests -v
uv run pytest engine/moe-quant/tests -v
uv run pytest engine/moe-router-predict/tests -v
uv run pytest engine/spec-decode-tree/tests -v
uv run pytest engine/moe-self-draft/tests -v
```

**Gate per pillar:** 100% green. Any red here is an architectural bug — fix
locally, never debug on paid GPU.

### L1.2 — Mock integration pipeline

Test that all 6 pillars wire together with dummy tensors. The engine must run
end-to-end on **CPU with mock weights** before we trust it on GPU.

`engine/dana-engine/tests/test_mock_pipeline.py`:
```python
import torch
from dana_engine import Engine

def test_mock_pipeline_cpu():
    """Engine runs end-to-end on CPU with synthetic weights."""
    engine = Engine.from_config({
        "num_experts": 8,
        "active_experts": 2,
        "hidden_dim": 128,
        "vocab_size": 1000,
        "device": "cpu",
        "tier_layout": {"fast": "ram", "slow": "disk"},  # M1 has no VRAM tier
    })
    fake_input = torch.randint(0, 1000, (1, 32))
    out = engine.generate(fake_input, max_new_tokens=8)
    assert out.shape == (1, 40), f"Expected (1, 40), got {out.shape}"
```

### L1.3 — Per-pillar local-vs-GPU validity matrix

For each pillar, write a CPU-runnable test that validates the part **independent
of CUDA kernels**:

| Pillar | Locally testable | GPU-only |
|---|---|---|
| 1 — router-predict | Predictor accuracy: small draft model vs target router on real traces | Prefetch latency |
| 2 — expert-cache | Cache hit rate on recorded traces (replay-based) | Wall-clock benefit |
| 3 — tiered-store | Migration policy correctness + spill ordering | VRAM↔RAM bandwidth |
| 4 — moe-quant | Numerical correctness of Int4/Int2 packing + perplexity delta | Quantized kernel speed |
| 5 — spec-decode-tree | Tree shape correctness + acceptance rate on small model | Wall-clock 2-3× speedup |
| 6 — moe-self-draft | Acceptance rate (same model as both draft + target) | Speedup |

**Output of L1:** `reports/L1_summary.md` with one row per test and pass/fail.

---

## Phase L2 — Native MPS / Real Small Inference — Day 2

Leave the uv venv. Use Python directly with MPS.

### L2.1 — Smoke test (Qwen3-0.6B on MPS)
```python
# scripts/02_smoke_mps.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "mps"
model = AutoModelForCausalLM.from_pretrained(
    "models/qwen3-0.6b", torch_dtype=torch.float16
).to(device)
tok = AutoTokenizer.from_pretrained("models/qwen3-0.6b")

inputs = tok("Explain MoE briefly:", return_tensors="pt").to(device)
out = model.generate(**inputs, max_new_tokens=50)
print(tok.decode(out[0], skip_special_tokens=True))
```

**Gate:** Output is coherent + no NaN + 5+ tok/s on M1.

### L2.2 — Quality baseline (Qwen3-4B BF16 on MPS)
Same script with `Qwen3-4B`. Expect 10-20 tok/s. This is the quality baseline
for benchmarks at L4.

### L2.3 — Wire engine to MPS device

The engine must accept `device="mps"` parametrically. If it doesn't, this is the
single biggest finding of the whole local-test exercise.

```python
from dana_engine import Engine
engine = Engine.from_config({..., "device": "mps", ...})
out = engine.generate(...)
```

**Gate:** Engine runs on MPS without CUDA-specific code paths. Any `device=="cuda"`
hardcoded in the engine = pre-GPU bug. Fix now.

### L2.4 — Per-expert quantization correctness

Test Pillar 4 against a real small model (Qwen3-4B, dense — surrogate for
per-expert quant logic):
1. Uniform Int4 quantization → measure perplexity on WikiText
2. Hybrid Int4/Int8 (half layers Int4, half Int8) → measure perplexity
3. Validate `moe-quant` package numerically produces same outputs as reference

---

## Phase L3 — Real MoE via llama.cpp + Metal + mmap — Half day

This is where Mac M1 becomes surprisingly capable. **Qwen3.6-35B-A3B Int4 is
18GB on disk, but with mmap only the active 1.5-2GB stays in RAM.** This is
*exactly* Dana's tiered-store pattern, just with `RAM↔SSD` instead of `VRAM↔RAM`.

### L3.1 — Baseline llama.cpp inference
```bash
llama-cli -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M \
    -p "Explain mixture-of-experts in one paragraph:" \
    -n 200 --mmap
```

**Gate:** Coherent output, 8-15 tok/s on M1 Max, RAM usage stays under 12GB.

### L3.2 — Capture router trace (ground truth)

Run `llama-server` with logging. Send 100 diverse prompts. Capture which experts
the model routes to at each token. This becomes ground truth for L3.3.

```bash
llama-server -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M \
    --port 8080 --log-format json --verbose > router_trace.jsonl &

# Send prompts (use a benchmark set or top 100 of WikiText)
uv run python scripts/03_send_prompts.py < prompts.txt
```

### L3.3 — Replay trace through Dana's expert-cache

```python
# scripts/04_replay_cache.py
from expert_cache import ExpertCache
import json

trace = [json.loads(line) for line in open("router_trace.jsonl")]
cache = ExpertCache(capacity=4)  # only 4 experts in fast tier (small budget)

hits = 0
for event in trace:
    for expert_id in event["routed_experts"]:
        if cache.access(expert_id):
            hits += 1
print(f"Hit rate: {hits / total:.1%}")
```

**Gate:** Hit rate > 80% with capacity = active_experts × 2. If lower, cache
policy is broken — fix before GPU.

### L3.4 — Self-draft validation (Pillar 6)

Qwen3.6-35B-A3B with 3B active params is a perfect self-draft candidate:
- Draft = subset (2 active experts)
- Target = full set (8 routed + 1 shared)

Measure acceptance rate over 500 tokens. **Gate:** > 60% acceptance.

---

## Phase L4 — Architecture Coverage Check — Half day

Production has 4 different MoE architectures. Engine must handle all 4.

### L4.1 — Inspect each target model's config

```bash
for m in moonshotai/Kimi-K2.6 Qwen/Qwen3-Coder-Next deepseek-ai/DeepSeek-V4-Flash Qwen/Qwen3.6-35B-A3B; do
  echo "=== $m ==="
  hf models info "$m" --json | jq '.config | {model_type, num_experts, num_experts_per_tok, hidden_size, num_hidden_layers}'
done
```

Capture: `num_experts`, `top_k`, `shared_experts`, `hidden_size` for each. These
go into a config matrix the engine's `Router` interface must accept.

### L4.2 — Mock-test engine against each topology

For each architecture, instantiate the engine with that topology + random
weights, run a mock generation:

```python
# scripts/05_arch_coverage.py
configs = {
    "qwen3_5_moe":  {"num_experts": 256, "top_k": 8, "shared": 1},
    "qwen3_next":   {"num_experts": ..., "top_k": ..., "shared": ...},
    "deepseek_v4":  {"num_experts": ..., "top_k": ..., "shared": ...},
    "kimi_k25":     {"num_experts": ..., "top_k": ..., "shared": ...},
}
for name, cfg in configs.items():
    engine = Engine.from_config({**cfg, "device": "mps", "dummy_weights": True})
    out = engine.generate(torch.randint(0, 1000, (1, 16)), max_new_tokens=4)
    assert out.shape[1] == 20, f"{name} failed"
    print(f"✓ {name}")
```

**Gate:** All 4 architectures load and run. Any failure = engine interface bug.

### L4.3 — Quantization coverage

Each architecture may have different expert weight shapes. Pillar 4's Int4
packing must work on all. Test with synthetic weights matching each shape.

---

## Phase L5 — Final Gate Check (1 hour)

Before clicking "rent GPU":

### Code gates
- [ ] L1.1: All 6 pillar unit tests green
- [ ] L1.2: Mock integration pipeline runs end-to-end
- [ ] L1.3: Each pillar has a CPU-validatable correctness test passing
- [ ] L2.1-2.2: Qwen3-0.6B and Qwen3-4B run on MPS
- [ ] L2.3: Engine accepts `device="mps"` and `device="cuda"` parametrically
- [ ] L2.4: Per-expert quant produces correct outputs vs reference
- [ ] L3.1: Qwen3.6-35B-A3B runs via llama.cpp Metal mmap (baseline)
- [ ] L3.3: Cache hit rate > 80% on real router trace
- [ ] L3.4: Self-draft acceptance rate > 60%
- [ ] L4.2: Engine loads + runs against all 4 production architectures

### Artifact gates
- [ ] `reports/L1_summary.md` (test results)
- [ ] `reports/router_trace.jsonl` (ground truth for cache tests)
- [ ] `reports/perplexity_quant.csv` (quality baseline)
- [ ] `reports/arch_coverage.md` (config matrix for all 4 target archs)
- [ ] `scripts/gpu_ready/` — all GPU-side scripts pre-written

### Cloud-prep gates
- [ ] `pyproject.toml + uv.lock` committed — env reproducible on cloud
- [ ] HF download script ready (no manual model fetching on GPU)
- [ ] `tmux` / `screen` workflow rehearsed
- [ ] SSH config + key ready
- [ ] Timer setup to track GPU billing

If all ✓: rent **4 hours of RTX 4090 with 90GB RAM**, not 10.

---

## What we still don't know after L5 (GPU-only)

These five items define the first hour of paid GPU time. Don't waste it on bugs
that should have been caught locally.

1. **GPTQ Int4 kernel speed** — CUDA-only kernels
2. **MTP speculative decoding** — vLLM/SGLang dependent
3. **PCIe bandwidth on prefetch** — different physical architecture
4. **VRAM fragmentation under sustained load** — CUDA-specific
5. **"30+ tok/s on 235B-class" claim** — requires the actual hardware tier

---

## Suggested folder structure

```
dana/
├── pyproject.toml             # uv-managed
├── uv.lock                    # reproducibility lock
├── PHASE_1_QWEN35.md          # phase 1 plan — Qwen3.5 family ladder
├── PHASE_2_TARGETS.md         # phase 2 plan — production serving targets
├── LOCAL_TEST_PLAN.md         # this file
├── scripts/
│   ├── 01_smoke_cpu.py
│   ├── 02_smoke_mps.py
│   ├── 03_send_prompts.py
│   ├── 04_replay_cache.py
│   ├── 05_arch_coverage.py
│   ├── 10_preflight_check.sh
│   └── gpu_ready/
│       ├── 01_smoke_cuda.py
│       ├── 02_pillar_bench.py
│       └── 03_push_targets.py
├── reports/                   # generated, gitignored except summaries
│   ├── L1_summary.md
│   ├── router_trace.jsonl
│   ├── perplexity_quant.csv
│   └── arch_coverage.md
└── models/                    # gitignored
    ├── qwen3-0.6b/
    ├── qwen3-4b-gguf/
    └── qwen3.6-35b-moe/
```

---

## Timeline

| Day | Phase | Output |
|---|---|---|
| Day 1 AM | Phase 0 + start downloads | Env ready, uv.lock committed |
| Day 1 PM | L1.1 + L1.2 | 6 pillars green, mock pipeline green |
| Day 1 EVE | L1.3 | Each pillar has a CPU-valid test |
| Day 2 AM | L2.1-2.3 | MPS inference works, engine device-portable |
| Day 2 PM | L2.4 + L3.1 | Quant correctness + 35B-A3B baseline |
| Day 2 EVE | L3.2-3.4 | Router trace + cache + self-draft validated |
| Day 3 AM | L4 | All 3 production archs load |
| Day 3 noon | L5 gate check | ✅ |
| Day 3 PM | **Rent GPU** | With confidence, 4h budget |

3 days local prep → 4 hours paid GPU instead of 10.
Estimated savings: ~550K toman + dramatically lower risk.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
migrated from Archive/LOCAL_TEST_PLAN.md — L1–L5 layered test plan when Dana was its own engine, superseded by PHASE_1_LOCAL.md
<!-- SECTION:NOTES:END -->
