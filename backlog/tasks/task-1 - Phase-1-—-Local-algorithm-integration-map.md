---
id: TASK-1
title: 'Phase 1 — Local: algorithm + integration map'
status: In Progress
assignee: []
created_date: '2026-05-26 06:32'
updated_date: '2026-05-26 07:21'
labels: []
dependencies: []
priority: high
ordinal: 1000
---

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
# Phase 1 — Local: algorithm + integration map (M1, $0)

> **Goal:** prove pillars 4/5/6 work in isolation and pin down the exact vLLM hook points for Phase 2. Zero GPU rental.
> **Output:** 3 green packages on a synthetic MoE + `reports/vllm_hook_points.md` with file:line references.
> **Next:** [PHASE_2_GPU.md](PHASE_2_GPU.md) once this doc's exit gate passes.

---

## What changed vs yesterday's plan

The "Dana engine" is not the runtime anymore — **vLLM is**. Dana ships as plugins on top of vLLM (`pip install dana-quant`, `pip install dana-spec`). That kills three assumptions from the original plan:

1. M1 cannot meaningfully run vLLM. Anything CUDA-shaped goes to Phase 2.
2. The dense-ladder benchmark on MPS (0.8B → 9B) was testing HF `transformers` + Apple Metal, not your code. Cut.
3. Pillar 1 (router-predict prefetch) has no bug class on M1 — unified memory has no PCIe. Cut from Phase 1.

What stays: standalone algorithm correctness for pillars 4, 5, 6 + one real-model sanity check via `llama.cpp` Metal.

## Stack

- `uv` for everything Python — single `pyproject.toml` + `uv.lock` at repo root
- `llama.cpp` (Metal) for the one real-model run
- `python 3.11`
- HF `transformers` only where unavoidable (model loading for tiny baselines)

## Day 1 — Workspace + llama.cpp baseline (4h)

```bash
# Toolchain
curl -LsSf https://astral.sh/uv/install.sh | sh
brew install llama.cpp
curl -LsSf https://hf.co/cli/install.sh | bash -s && hf auth login

# uv workspace at repo root
cd /Users/seyed/Projects/personal-projects/dana
uv init --name dana --python 3.11
uv add --dev pytest pytest-benchmark ruff mypy
# Add the 4 surviving pillars as workspace members
for p in moe-quant moe-self-draft spec-decode-tree moe-router-predict; do
  uv add --editable "engine/$p"
done
uv lock
git add pyproject.toml uv.lock && git commit -m "uv workspace bootstrap"
```

**llama.cpp baseline** — the only real-model run on M1:

```bash
mkdir -p models reports
hf download unsloth/Qwen3.5-35B-A3B-GGUF \
  --local-dir models/qwen35-35b-a3b --include "*Q4_K_M*"

llama-cli -m models/qwen35-35b-a3b/*Q4_K_M*.gguf \
  -p "Explain mixture-of-experts in one paragraph:" \
  -n 200 --mmap > reports/llamacpp_baseline.txt
```

**Gate:** coherent output, ≥ 8 tok/s, RAM < 14 GB. If this fails, the rest of the plan is moot — fix the model load first.

## Day 2 — Pillar 4 standalone (6h, `engine/moe-quant`)

Re-implement per-expert PTQ following EAQuant (Feb 2026 revision) + the April 2026 *Efficient Quantization of MoE* paper.

- Hot tier → Q8, warm → Q4, cold → Q2, group-size 128
- Calibration: 64 prompts of WikiText, capture per-expert activation
- Validate round-trip cosine vs FP16 on synthetic 8-expert MoE
- All under `pytest engine/moe-quant`, CPU-only

**Output:** `reports/quant_roundtrip.csv` (one row per (expert, bits, cosine)).

**Gate:** Q4 cosine ≥ 0.999, Q2 ≥ 0.97 on synthetic, all tests green.

## Day 3 — Pillars 5 + 6 standalone (6h)

Bring the existing 206 tests back to green under uv:

```bash
uv run pytest engine/spec-decode-tree engine/moe-self-draft -v
```

Then add one new test each:

- **spec-decode-tree:** batched tree-verification path (the known bug from old `NEXT_STEPS.md` — one forward per path → one batched forward).
- **moe-self-draft:** end-to-end self-draft acceptance on the existing tiny MoE, > 60% on deterministic routing.

**Output:** `reports/spec_decode_synth.md`.

**Gate:** all pillar 5+6 tests green + batched tree verify lands as one PR-sized commit.

## Day 4 — vLLM hook-point map (4h)

`uv add vllm` (CPU build — slow but you only need to read source).

Map exactly where each plugin will subclass:

| Pillar | File to read in vLLM | Subclass point |
|---|---|---|
| 4 quant | `vllm/model_executor/layers/quantization/__init__.py` + `gptq_marlin.py` | `QuantizationConfig` |
| 5 tree spec | `vllm/spec_decode/spec_decode_worker.py` | `SpeculativeProposer` |
| 6 self-draft | same as 5 | same |
| 1 router-predict | `vllm/distributed/expert_parallel.py` + `vllm/model_executor/layers/fused_moe/` | **no clean hook — fork or upstream PR** |

Write four empty plugin shells (≤ 50 lines each) that register against stock vLLM via `entry_points`. They do nothing yet — but `vllm serve --quantization dana` must accept the flag.

**Output:** `reports/vllm_hook_points.md`.

**Gate:** `python -c "import vllm; from dana_quant import DanaQuantConfig"` succeeds. Plugin shells importable.

## What NOT to do on M1

- Don't run vLLM end-to-end. CPU build is too slow to be informative — use it only as a source-reading dependency.
- Don't run Qwen3.5-27B on MPS via `transformers`. The numbers don't transfer to a 3090.
- Don't test pillar 1's prefetch logic. M1 has unified memory, no PCIe. No bug class lives here.
- Don't write CUDA kernels yet. Phase 2.

## Exit gate → Phase 2

- [ ] `uv.lock` committed; `uv sync` reproduces the env on any machine
- [ ] `reports/llamacpp_baseline.txt` recorded
- [ ] `pytest engine/moe-quant engine/spec-decode-tree engine/moe-self-draft` all green
- [ ] `reports/quant_roundtrip.csv` meets the gates above
- [ ] `reports/vllm_hook_points.md` lists exact file:line per pillar
- [ ] Four plugin shells importable against stock vLLM

When all six boxes are ticked, move to [PHASE_2_GPU.md](PHASE_2_GPU.md).

---

## Folder layout after Phase 1

```
dana/
├── pyproject.toml          # uv workspace
├── uv.lock
├── PHASE_1_LOCAL.md        # this file
├── PHASE_2_GPU.md
├── engine/
│   ├── moe-quant/          # pillar 4 — vLLM plugin
│   ├── spec-decode-tree/   # pillar 5 — vLLM plugin
│   ├── moe-self-draft/     # pillar 6 — vLLM plugin
│   └── moe-router-predict/ # pillar 1 — fork candidate, kept for Phase 2 work
├── models/                 # gitignored, GGUFs for llama.cpp
├── reports/                # gitignored except summaries
└── Archive/                # everything stale (don't touch)
```
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
migrated from PHASE_1_LOCAL.md (kept in place at repo root as canonical spec)
<!-- SECTION:NOTES:END -->
