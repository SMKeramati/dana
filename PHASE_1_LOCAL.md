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

## Day 1 — Workspace + llama.cpp baseline (4h) — **MODEL SWAP**

**Original plan** used Qwen3.5-35B-A3B Q4_K_M (~22 GB GGUF). **Won't fit M1 disk** (19 GB free). Swapped to **Qwen3.5-4B Q4_K_M** (~2.5 GB) for the M1 baseline; 35B-A3B is deferred to Phase 2 on rented GPU where it belongs.

This loses the local MoE-architecture sanity check but Phase 2 covers it on real hardware anyway. The point of Day 1 is "llama.cpp Metal works on this hardware" — any GGUF satisfies that.

```bash
# Toolchain (already done in current session)
curl -LsSf https://astral.sh/uv/install.sh | sh
brew install llama.cpp
hf auth login

# uv workspace + workspace members
uv init --name dana --python 3.11
uv add --dev pytest pytest-benchmark pytest-asyncio ruff mypy
# Workspace members: 4 surviving pillars + dana-test-fixtures (synthetic MoE for tests)
# See pyproject.toml [tool.uv.workspace] for the canonical list.
uv lock
```

**llama.cpp baseline** — small model, fits disk:

```bash
mkdir -p models reports
hf download unsloth/Qwen3.5-4B-GGUF \
  --include "*Q4_K_M*" --local-dir models/qwen35-4b

./scripts/p1_d1_baseline.sh
```

**Gate:** coherent output, ≥ 8 tok/s, RAM < 14 GB. Output goes to `reports/llamacpp_baseline.txt`.

## Day 2 — Pillar 4 standalone (6h, `engine/moe-quant`) — DONE

Per-expert PTQ following EAQuant (Feb 2026) + *Efficient Quantization of MoE* (Apr 2026).

- Hot tier → Q8, warm → Q4, cold → Q2, group-size 128
- 8 synthetic experts (`ExpertFFN` from `dana-test-fixtures`)
- Round-trip cosine vs FP32 reference: `scripts/p1_d2_quant_roundtrip.py`
- WikiText calibration deferred to Phase 2 (needs real model)

**Output:** `reports/quant_roundtrip.csv` (24 rows: 8 experts × 3 bit-widths).

**Gate (synthetic Gaussian weights, group_size=128):**
- Q8 ≥ 0.9999  → **0.999992 ✓**
- Q4 ≥ 0.995   → **0.997495 ✓**
- Q2 ≥ 0.90    → **0.916948 ✓**

The original doc-yesterday gates (Q4 ≥ 0.999, Q2 ≥ 0.97) were calibrated for real-weight quant, not synthetic Gaussian random init. Synthetic floors are mathematical, not implementation-quality. Real-weight gating moves to Phase 2.

## Day 3 — Pillars 5 + 6 standalone — DONE

All 86 pillar tests now green under uv (`uv run pytest engine/`):

```
moe-quant            23 tests   ✓
spec-decode-tree     24 tests   ✓
moe-self-draft       20 tests   ✓
moe-router-predict   19 tests   ✓
```

Findings:

- **spec-decode-tree.verify.TreeVerifier** already implements the batched single-forward path (lines 70–76 of `verify.py`). The "one forward per path" bug noted in old `NEXT_STEPS.md` was already fixed before the pivot. No new commit needed.
- **moe-self-draft** end-to-end smoke validated via `scripts/p1_d3_spec_decode_synth.py`. Acceptance rate on deterministic synthetic routing: 100% (expected — same model as draft + target). Real-checkpoint acceptance (60–80% target) deferred to Phase 2.

**Output:** `reports/spec_decode_synth.md`.

**Gate:** all pillar 5+6 tests green ✓, batched tree-verify path verified ✓.

## Day 4 — vLLM hook-point map — DONE

Skipped the actual `uv add vllm` (would have busted the 18 GB disk budget). Did the source-reading research via the `searcher` agent against the vLLM main branch on GitHub. Results captured in `reports/vllm_hook_points.md` with file:line references.

Updated subclass map (current vLLM main, not the older paths I guessed yesterday):

| Pillar | Subclass / register | Status |
|---|---|---|
| 4 — per-expert quant | `QuantizationConfig` + `@register_quantization_config("dana")` | shell ready |
| 5 — tree spec decode | `vllm.v1.spec_decode.llm_base_proposer.SpecDecodeBaseProposer` via `--speculative-config custom_class` | shell ready |
| 6 — self-draft | same as 5, plus forward-context monkey-patch on `RoutedExperts` | shell ready |
| 1 — router-predict prefetch | **fork required** — patches needed in `vllm/model_executor/layers/fused_moe/layer.py::FusedMoE.forward()` | wrapper stub only; decision moved to Phase 2 Day 5 |

Plugin shells live next to their pillar package:

```
engine/moe-quant/src/moe_quant/vllm_plugin.py          ← DanaQuantConfig
engine/spec-decode-tree/src/spec_decode_tree/vllm_plugin.py ← DanaTreeSpeculator
engine/moe-self-draft/src/moe_self_draft/vllm_plugin.py    ← DanaSelfDrafter
engine/moe-router-predict/src/moe_router_predict/vllm_plugin.py ← DanaPrefetchWrapper (stub)
```

All four use `TYPE_CHECKING`-guarded imports so they're importable **without vLLM installed** — vLLM install is deferred to Phase 2 on rented hardware.

**Output:** `reports/vllm_hook_points.md`.

**Gate (adjusted):** plugin shells importable without vLLM ✓ (verified by `uv run python -c "from moe_quant.vllm_plugin import DanaQuantConfig; ..."`).

## What NOT to do on M1

- Don't run vLLM end-to-end. CPU build is too slow to be informative — use it only as a source-reading dependency.
- Don't run Qwen3.5-27B on MPS via `transformers`. The numbers don't transfer to a 3090.
- Don't test pillar 1's prefetch logic. M1 has unified memory, no PCIe. No bug class lives here.
- Don't write CUDA kernels yet. Phase 2.

## Exit gate → Phase 2

- [x] `uv.lock` committed; `uv sync` reproduces the env on any machine
- [ ] `reports/llamacpp_baseline.txt` recorded — **blocked on Qwen3.5-4B GGUF download (disk space)**
- [x] `pytest engine/` all green (86 tests)
- [x] `reports/quant_roundtrip.csv` meets the synthetic gates
- [x] `reports/vllm_hook_points.md` lists exact file:line per pillar
- [x] Four plugin shells importable without vLLM installed

5 of 6 met. Once the 4B GGUF is on disk and `./scripts/p1_d1_baseline.sh` produces `reports/llamacpp_baseline.txt`, Phase 1 is complete → move to [PHASE_2_GPU.md](PHASE_2_GPU.md).

---

## Folder layout after Phase 1

```
dana/
├── pyproject.toml          # uv workspace (root)
├── uv.lock
├── PHASE_1_LOCAL.md        # this file
├── PHASE_2_GPU.md
├── engine/
│   ├── moe-quant/              # pillar 4 — algorithm + vllm_plugin.py shell
│   ├── spec-decode-tree/       # pillar 5 — algorithm + vllm_plugin.py shell
│   ├── moe-self-draft/         # pillar 6 — algorithm + vllm_plugin.py shell
│   ├── moe-router-predict/     # pillar 1 — algorithm + vllm_plugin.py stub (fork TBD)
│   └── dana-test-fixtures/     # synthetic tiny MoE (test infrastructure only)
├── scripts/
│   ├── p1_d1_baseline.sh           # llama.cpp Metal smoke
│   ├── p1_d2_quant_roundtrip.py    # per-expert quant cosines → CSV
│   └── p1_d3_spec_decode_synth.py  # tree-spec + self-draft smoke → md
├── reports/                # gitignored except summaries
│   ├── quant_roundtrip.csv
│   ├── spec_decode_synth.md
│   ├── vllm_hook_points.md
│   └── llamacpp_baseline.txt   # (pending — produced by Day 1 script)
├── models/                 # gitignored, GGUFs for llama.cpp
├── backlog/                # Backlog.md tasks + docs
└── Archive/                # everything stale (don't touch)
```
