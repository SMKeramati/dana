# Phase 2 — Production Serving Targets

> **Status**: this is **Phase 2** — the 4 models Dana engine must support in production.
> **Prerequisite**: [PHASE_1_QWEN35.md](PHASE_1_QWEN35.md) must pass its gate first.
> Phase 1 validates the engine on the Qwen3.5 family ladder (where we have size variants
> and same-family draft models). Phase 2 takes that validated engine and extends it to
> the production targets below.

---

## Production tiers

| Tier | Model | HF ID | Total Params | Arch | License |
|---|---|---|---|---|---|
| **HUGE** | Kimi-K2.6 | [`moonshotai/Kimi-K2.6`](https://huggingface.co/moonshotai/Kimi-K2.6) | ~1.06T | `kimi_k25` | Modified MIT |
| **LARGE** | DeepSeek-V4-Flash | [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | ~158B | `deepseek_v4` | DeepSeek License |
| **MEDIUM** | Qwen3-Coder-Next | [`Qwen/Qwen3-Coder-Next`](https://huggingface.co/Qwen/Qwen3-Coder-Next) | ~80B | `qwen3_next` | Apache 2.0 |
| **SMALL** | Qwen3.6-35B-A3B | [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | ~36B | `qwen3_5_moe` | Apache 2.0 |

> **About the `arch` column:** this is the `model_type` field from each model's
> `config.json` — the architecture class registered in `transformers`. New model
> versions often reuse the previous arch class (e.g. Qwen3.6 reuses `qwen3_5_moe`,
> Kimi-K2.6 reuses `kimi_k25`). The engine dispatches on this string, not the
> marketing name.

## Architectural coverage

These four architectures are **not interchangeable** at the engine level. Router,
expert count, and routing top-k differ. Dana must abstract over all four:

| Arch | Expected Expert Layout | Router Style |
|---|---|---|
| `qwen3_5_moe` (Qwen3.6-35B-A3B) | 256 experts, 8 routed + 1 shared active (~3B active) | top-k softmax |
| `qwen3_next` (Qwen3-Coder-Next) | TBD — new architecture, verify on first load | TBD |
| `deepseek_v4` (DeepSeek-V4-Flash) | TBD — historically DeepSeek uses fine-grained experts | TBD |
| `kimi_k25` (Kimi-K2.6) | TBD — Moonshot K2.5 family, ~1T params, MoE | TBD |

**Implications for engine code:**
- The `Router` interface must accept arbitrary `(num_experts, top_k, shared)` configs
- Expert cache eviction policy must be invariant to expert count
- Per-expert quantization (Pillar 4) must support varying expert sizes
- Attention dispatch must handle gated attention (qwen3_next) and MLA (deepseek_v4)

## VRAM sizing (Int4, rough)

| Tier | Total Int4 | Active Int4 | Min GPU | Recommended |
|---|---|---|---|---|
| SMALL (Qwen3.6-35B-A3B) | ~18 GB | ~1.5 GB | RTX 3090 24GB | RTX 4090 24GB |
| MEDIUM (Qwen3-Coder-Next 80B) | ~40 GB | TBD | A6000 48GB | A100 40GB |
| LARGE (DeepSeek-V4-Flash 158B) | ~80 GB | TBD | A100 80GB | H100 80GB |
| HUGE (Kimi-K2.6 ~1T) | ~500 GB | TBD | multi-node | H100 cluster or aggressive RAM+SSD tiering |

> SMALL/MEDIUM/LARGE all deployable on **single H100** (141 GB) or **2× A100 80GB**.
> HUGE (Kimi-K2.6 1T) is **the real stress test** — only Dana's RAM+SSD tiering on a
> 4090+90GB-RAM node could even attempt it. This is the engine's reason to exist.

## Local-test surrogate mapping

We cannot run any production target on Mac M1 at native size. Validate against
**architecturally identical** smaller models, or mock:

| Production Target | Local Surrogate (M1) | Why |
|---|---|---|
| Qwen3.6-35B-A3B (SMALL) | Same model via `llama.cpp + Metal + mmap` Q4_K_M | Fits in 16GB via mmap |
| Qwen3-Coder-Next 80B (MEDIUM) | Qwen3-Coder smaller variant or arch-only mock | 80B too big even mmap |
| DeepSeek-V4-Flash 158B (LARGE) | DeepSeek-V2-Lite or arch-only mock | Same router family |
| Kimi-K2.6 ~1T (HUGE) | Arch-only mock — no small Kimi variant exists | 1T impossible locally |

If no smaller variant exists, use **mock pipelines with the architectural shape
preserved** — same `num_experts`, `top_k`, `hidden_dim` ratios, on tiny weights.

---

## Phase 2A — Qwen3.6-35B-A3B (SMALL prod tier)

Nearly free given Phase 1. Architecture is `qwen3_5_moe` (same class as Qwen3.5-35B-A3B
that Phase 1 already validated). Only weights differ. Should slot into the engine
with no code change.

**Risk level: LOW** — Phase 1 already validated this architecture.

**Gates:**
- [ ] Loads via Dana engine without code changes
- [ ] Outputs coherent for 100 prompts vs HF reference
- [ ] All 6 pillars active, throughput within 10% of Phase 1 Qwen3.5-35B-A3B numbers

## Phase 2B — Qwen3-Coder-Next 80B (MEDIUM prod tier)

New architecture (`qwen3_next`). Different attention pattern (gated). Engine code
needs to dispatch differently.

**Open questions for engine work:**
- Does Pillar 1 (router prediction) work the same way on `qwen3_next`?
- Does Pillar 4 (per-expert quant) need topology-specific code?

**Risk level: MEDIUM** — new arch, but same vendor as Qwen3.5 (similar code style).

**Gates:**
- [ ] Engine dispatches `qwen3_next` arch correctly
- [ ] Gated attention path works under Pillar 1 prefetch
- [ ] Throughput ≥ 80% of equivalent-sized Qwen3.5 model

## Phase 2C — DeepSeek-V4-Flash 158B (LARGE prod tier)

DeepSeek's own architecture (MLA attention, fine-grained experts).

**Risk level: MEDIUM–HIGH** — completely different attention mechanism. Engine
needs deepseek-specific code paths.

**Gates:**
- [ ] MLA attention runs correctly through engine
- [ ] Fine-grained expert layout handled by Pillar 2/3 caching
- [ ] Pillar 4 per-expert quant works on DeepSeek's expert shapes
- [ ] End-to-end serving works on A100 80GB

## Phase 2D — Kimi-K2.6 ~1T (HUGE prod tier) — the hard one

Two compounding problems:
1. **No small variant.** Spec decode strategy unclear — see open research below.
2. **Size.** 1T params = ~500GB Int4. Single-node infeasible even with aggressive
   tiering. Either multi-node, or Int2/Int3 (which may break quality).

**Open research for spec decode on Kimi:**

| Strategy | Approach | Risk |
|---|---|---|
| Cross-family small draft | Use `Qwen3.5-0.8B` or similar | low acceptance rate likely |
| Aggressive quant self-draft | Use Kimi at Int2 as draft for Int4 Kimi | quality of Int2 unknown (Phase 1D answers this) |
| Subset-of-experts (Pillar 6) | Self-draft using fewer experts | only works if Kimi has many small experts |
| Distillation | Train a small Kimi-shaped draft | expensive, out of scope |
| No spec decode | Accept lower throughput | viable fallback |

**Action item before serving Kimi:** prove which spec-decode strategy survives at
production quality. This is its own research mini-project, probably 1–2 weeks of
GPU work.

**Risk level: HIGH** — multiple unknowns. May need to fund this separately.

---

## File / script structure

```
dana/scripts/phase2/
├── 2a_qwen36_smoke.py
├── 2b_qwen3_coder_next.py
├── 2c_deepseek_v4.py
└── 2d_kimi_k26_research.py

dana/reports/phase2/
└── (after Phase 1 gate passes)
```

---

## Bottom line

Phase 1 ([PHASE_1_QWEN35.md](PHASE_1_QWEN35.md)) is the **right tool to debug the engine**
because we have size variants and same-family drafts.

Phase 2 (this doc) is the **right tool to validate serving** because that's what
we'll actually serve.

Do not reverse the order. Debugging the engine on a 1T-parameter Kimi with no draft
model is exactly the failure mode this plan is designed to avoid.
