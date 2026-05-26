---
id: TASK-4
title: Phase 1 — Qwen3.5 Family Ladder (original AI-written plan)
status: Done
assignee: []
created_date: '2026-05-26 06:33'
labels:
  - archive
  - strategy
dependencies: []
ordinal: 4000
---

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
# Phase 1 — Qwen3.5 Family Ladder

> **Status**: this is **Phase 1** — engine validation across the Qwen3.5 size spectrum.
> **Next**: once Phase 1 passes the gate (see bottom of this doc),
> move to [PHASE_2_TARGETS.md](PHASE_2_TARGETS.md) for production serving targets.
> **Infrastructure tests** (L1–L5) referenced throughout: [LOCAL_TEST_PLAN.md](LOCAL_TEST_PLAN.md).

---

## Why Qwen3.5 first

The Phase 2 production targets (Kimi-K2.6, DeepSeek-V4-Flash, Qwen3-Coder-Next,
Qwen3.6-35B-A3B) are great for **serving** but **terrible for engine validation**:

| Target | Smallest variant | Spec-decode pair? |
|---|---|---|
| Kimi-K2.6 (~1T) | none | ❌ no small Kimi exists |
| DeepSeek-V4-Flash (158B) | V2-Lite from different generation | ⚠️ cross-version risk |
| Qwen3-Coder-Next (80B) | no Coder variants released | ❌ |
| Qwen3.6-35B-A3B (36B) | itself is smallest | ❌ |

vs the [Qwen3.5 collection](https://huggingface.co/collections/Qwen/qwen35):

```
0.8B  →  2B  →  4B  →  9B  →  27B  →  35B-A3B  →  122B-A10B  →  397B-A17B
 ──── dense ────  ──── dense ──  ──── MoE ────  ──── MoE ──────  ── MoE ──
```

Same tokenizer, same architecture family, same training distribution. This gives us:
1. **Engine validation at every scale** — bugs that only show up at scale can be caught
   incrementally rather than all at once on a 397B model
2. **Real speculative decoding pairs** — 0.8B drafts for 27B, 4B drafts for 122B-A10B
3. **Quality continuum** — compare quant impact on the same family at different sizes
4. **No reason to debug arch quirks** — one architecture across the whole ladder

---

## Models in scope

| Model | Total | Active | Int4 | M1 16GB? | Cloud (4090 + 90GB RAM) |
|---|---|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` | 0.8B | dense | ~0.5GB | ✅ everywhere | ✅ |
| `Qwen/Qwen3.5-2B` | 2B | dense | ~1.2GB | ✅ everywhere | ✅ |
| `Qwen/Qwen3.5-4B` | 4B | dense | ~2.5GB | ✅ MPS | ✅ |
| `Qwen/Qwen3.5-9B` | 9B | dense | ~5GB | ✅ MPS (Int4) | ✅ |
| `Qwen/Qwen3.5-27B-GPTQ-Int4` | 27B | dense | ~14GB | ⚠️ tight (Int4 only) | ✅ |
| `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | 35B | MoE 3B | ~18GB | ✅ via llama.cpp mmap | ✅ |
| `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` | 122B | MoE 10B | ~62GB | ❌ | ✅ (RAM tier) |
| `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` | 397B | MoE 17B | ~200GB | ❌ | ⚠️ stress test |

---

## Phase 1A — Dense ladder on M1 (Day 1)

Run each dense model end-to-end on MPS, climbing:

```
0.8B BF16  →  2B BF16  →  4B BF16  →  9B Int4  →  27B Int4 (tight)
```

**Per-model gates:**
- [ ] Loads without OOM on M1
- [ ] Coherent generation (50+ tokens make sense)
- [ ] Dana engine matches HF reference output (or known sampling)
- [ ] tok/s recorded for baseline curve

**Why this matters:** any engine bug that scales with model size shows up before you
spend money. If 9B works but 27B leaks, that's a $0 finding.

## Phase 1B — MoE on M1 via mmap (Day 1 EVE)

```bash
llama-cli -hf unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M -p "..." -n 200 --mmap
```

Then wire Dana engine to do the same with the actual `qwen3_5_moe` config:
- 256 experts, 8 routed + 1 shared active
- Engine must page experts in/out from disk (M1 has no VRAM tier, but `RAM↔SSD`
  is architecturally identical to `VRAM↔RAM`)

**Gates:**
- [ ] Qwen3.5-35B-A3B runs via llama.cpp baseline
- [ ] Dana engine reproduces same outputs via its own tiered loader
- [ ] RAM usage stays under 14GB (with mmap working)
- [ ] Capture router trace for L3.2 of [LOCAL_TEST_PLAN.md](LOCAL_TEST_PLAN.md)

## Phase 1C — Speculative decoding within family (Day 2)

This is the **strategic test** of the whole family approach. Try every reasonable
draft↔target pairing:

| Draft | Target | Hypothesis | Test |
|---|---|---|---|
| `Qwen3.5-0.8B` | `Qwen3.5-9B` | small-same-family drafts well for medium dense | acceptance rate > 60% |
| `Qwen3.5-0.8B` | `Qwen3.5-27B` Int4 | small drafts for large dense | acceptance rate > 50% |
| `Qwen3.5-0.8B` | `Qwen3.5-35B-A3B` | dense drafts for MoE target | acceptance rate > 40% |
| `Qwen3.5-2B` | `Qwen3.5-35B-A3B` | bigger draft, better acceptance | acceptance > 55% |
| `Qwen3.5-4B` | `Qwen3.5-35B-A3B` | diminishing returns? | acceptance vs cost |
| Self-draft (Pillar 6) | `Qwen3.5-35B-A3B` | subset of experts as draft | compare to external draft |

**Output:** `reports/spec_decode_qwen35_matrix.md` with the full matrix.

**Decisive insight expected:** which draft size is the sweet spot for which target
size in the same family. That answer directly informs how to serve the Phase 2 targets.

## Phase 1D — Quantization ladder (Day 2 PM)

Same model, different quants. Measure quality + speed:

| Model | Quants to test |
|---|---|
| `Qwen3.5-9B` | BF16, FP8, Int8, Int4 (GPTQ), Int4 (AWQ), Int2 (extreme) |
| `Qwen3.5-35B-A3B` | FP8, Int4 GPTQ, mixed per-expert (Pillar 4) |

For each: perplexity on WikiText + 20-prompt generation quality score + tok/s.

**Critical question for Phase 2:** at what quant level does quality break down?
This sets the floor for what we can use as **self-draft material for Kimi-K2.6**
(where no same-family small model exists — see Phase 2D).

## Phase 1E — GPU stress (when paid GPU starts)

Once Phase 1A–1D pass locally, the rented GPU validates:
- 122B-A10B on RTX 4090 + 90GB RAM tier (Pillar 3 stress test)
- 397B-A17B as a stretch goal (or fall back to A6000)
- All 6 pillars active simultaneously
- Compare local spec-decode matrix to GPU spec-decode matrix — does the pattern hold?

---

## Phase 1 → Phase 2 Gate

Move to [PHASE_2_TARGETS.md](PHASE_2_TARGETS.md) only when:

- [ ] Phase 1A: dense ladder 0.8B–9B all run on M1 with Dana engine
- [ ] Phase 1B: Qwen3.5-35B-A3B runs via Dana engine on M1 (mmap-backed)
- [ ] Phase 1C: spec-decode matrix complete, at least one pair > 60% acceptance
- [ ] Phase 1D: quant ladder complete, breakdown floor identified
- [ ] Phase 1E: at least one Qwen3.5 model > 30B running on rented GPU with all 6 pillars active
- [ ] Final report: `reports/phase1/dense_ladder_results.md` + `spec_decode_qwen35_matrix.md` + `quant_ladder.csv`

If any gate fails, **fix the engine on Qwen3.5** before touching any Phase 2 model.

---

## File / script structure

```
dana/scripts/phase1/
├── 1a_dense_ladder.py
├── 1b_moe_mmap.py
├── 1c_spec_decode_matrix.py
└── 1d_quant_ladder.py

dana/reports/phase1/
├── dense_ladder_results.md
├── spec_decode_qwen35_matrix.md
└── quant_ladder.csv
```
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
migrated from Archive/strategy/PHASE_1_QWEN35_orig.md — first AI-drafted Phase 1, superseded by lean PHASE_1_LOCAL.md (2026-05-25)
<!-- SECTION:NOTES:END -->
