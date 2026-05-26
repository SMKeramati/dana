# Phase 2 — GPU: vLLM + Dana plugins on rented hardware

> **Prerequisite:** [PHASE_1_LOCAL.md](PHASE_1_LOCAL.md) exit gate passed.
> **Goal:** prove Dana plugins beat stock vLLM on the Qwen3.5 ladder. Stretch: production targets.
> **Budget cap:** 16 GPU-hours total (≈ 1.36M toman on 3090 @ 85k/hr).

---

## Stack

- `uv sync` from the Phase 1 lockfile — same env, no surprises
- vLLM stable (latest as of 2026-05) via `uv add vllm`
- 4 Dana plugin packages installed editable from `engine/`
- GPU: RTX 3090 24GB (Qom-7, 85k toman/hr) for first runs; A6000 48GB only when a model won't fit

## Pre-rental checklist (do on M1, before paying)

Every script for this phase must already exist on disk before you SSH into the rental. **No dev-on-rental.** Editing CUDA code on a paid clock is how 4 hours become 10.

- [ ] `scripts/p2_setup.sh` — uv sync + hf download + warmup
- [ ] `scripts/p2_baseline.py` — stock vLLM, 50 prompts, 3 batch sizes
- [ ] `scripts/p2_pillar4.py` — same suite under DanaQuantConfig
- [ ] `scripts/p2_pillar5.py` — tree spec decode plugin
- [ ] `scripts/p2_pillar6.py` — self-draft plugin
- [ ] tmux config + SSH key tested

## Hour 1 — Setup (one-time per session)

```bash
ssh root@rental
tmux new -s dana
git clone <repo> && cd dana
uv sync                                            # locked from Phase 1
uv pip install -e engine/moe-quant engine/spec-decode-tree engine/moe-self-draft
hf download Qwen/Qwen3.5-27B-GPTQ-Int4 \
            Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
python -c "import torch, vllm; print(torch.cuda.is_available(), vllm.__version__)"
```

**Gate:** CUDA visible, vLLM imports, both models downloaded.

## Hours 2-3 — Stock vLLM baseline (the floor every pillar must beat)

```bash
bash scripts/p2_baseline.py \
  --models Qwen/Qwen3.5-27B-GPTQ-Int4,Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --batch-sizes 1,4,16 --prompts 50 \
  > reports/vllm_floor.csv
```

Record: tok/s, first-token latency, peak VRAM, output sample.

**This is the number every pillar must beat. Don't skip. Commit it.**

## Day 2 — Pillar 4 (per-expert quantization)

- Register `DanaQuantConfig` as a vLLM quant backend (Phase 1 wired the shell)
- Re-quantize Qwen3.5-35B-A3B with hot=Q8 / warm=Q4 / cold=Q2 from the activation profile captured in Phase 1
- Same benchmark suite as the floor

**Gate:** ≥ 10% throughput gain on batch ≥ 4 (memory-bound regime), < 0.5% perplexity delta on WikiText.

## Day 3 — Pillar 5 (tree spec decode for MoE)

- Subclass vLLM's `SpeculativeProposer` with the batched tree variant from Phase 1
- Self-target draft (same model, top-1 routing) on Qwen3.5-27B-Int4
- Adaptive depth/width controller from the existing `spec-decode-tree` package

**Gate:** ≥ 1.5× tokens-per-step on coding prompts, ≥ 60% acceptance.

## Day 4 — Pillar 6 (self-draft on MoE)

- Same `SpeculativeProposer` framework, top-1 expert routing as the draft signal
- Compare against pillar 5 (external vs self draft)
- Decide which is the default for production

**Gate:** ≥ 60% acceptance, ≥ 1.3× throughput vs floor.

## Day 5 — Pillar 1 (router-predict prefetch) — the fork decision

vLLM exposes no clean hook for expert-offload prefetch. Two paths:

1. **Upstream PR:** open an RFC against vLLM with the draft-model router predictor. ~2 weeks of review cycles.
2. **Thin fork:** branch vLLM, patch `vllm/distributed/expert_parallel.py`, rebase weekly.

Decision criterion: if pillars 4+5+6 already give ≥ 30% combined gain over floor, pillar 1 is a stretch — defer to a Phase 3. If they give < 30%, pillar 1 is the only path to the original 30 tok/s thesis, so commit to the fork.

## Stretch — Production-target smoke

Once Qwen3.5 numbers are in, try the Phase 2 production targets on whatever GPU fits:

| Model | Min GPU | Phase-2 risk |
|---|---|---|
| `Qwen/Qwen3.6-35B-A3B` | 3090 24GB (Int4) | LOW — same arch as 3.5-35B-A3B |
| `Qwen/Qwen3-Coder-Next` (80B) | A6000 48GB or A100 40GB | MED — new arch `qwen3_next` |
| `deepseek-ai/DeepSeek-V4-Flash` (158B) | A100 80GB | MED-HIGH — MLA attention |
| `moonshotai/Kimi-K2.6` (~1T) | multi-node | HIGH — separate sprint |

All 4 verified on HF (May 2026). Run stock-vLLM first; plug-ins only if stock works.

## Exit gate → "real product"

- [ ] `reports/vllm_floor.csv` committed
- [ ] Dana plugins combined throughput ≥ floor + 30% on Qwen3.5-35B-A3B
- [ ] Plug-ins installable via `uv pip install dana-quant dana-spec` (standalone, not just editable)
- [ ] At least one Phase-2 production target runs end-to-end under Dana
- [ ] Full benchmark matrix in `reports/phase2/`

## What "real product" looks like

If you pass the exit gate, you have:

- 2-3 OSS `pip`-installable packages with measurable speedup vs stock vLLM
- A clear story: "vLLM is the engine, Dana is the optimization layer"
- An optional thin serving layer (FastAPI in front of `vllm serve`) — this is where the **platform** decision from the old plan comes back, but only when there is real demand

## What NOT to do in Phase 2

- Don't refactor vLLM internals without an issue/PR opened first.
- Don't optimize pillar 1 on rental hours — design it locally, only measure on GPU.
- Don't pull in Phase-2 production targets before Qwen3.5 numbers prove the floor-beating gain.
- Don't restart the platform/ work. It stays in Archive/ until the engine pays for itself.
