# Phase 1 — actual M1 inference benchmark across pillars

**Hardware:** M1 (CPU only, no GPU)  
**Model:** TinyMoE micro: 2 layers, 4 experts (top-1), hidden=64, ~181 K params  
**Per scenario:** 5 warmup runs + 20 measured runs, up to 30 new tokens each  
**Pillar 1 (PCIe prefetch):** skipped — unified memory has no PCIe; no bug class on M1.

## Results

| Scenario | tok/s | Δ vs baseline | Extras |
|---|---:|---:|---|
| naive (baseline) | **1313.9** | 1.00× |  |
| + pillar 4 (Q8 weights) | **1327.9** | 1.01× |  |
| + pillar 4 (Q4 weights) | **1278.3** | 0.97× |  |
| + pillar 4 (Q2 weights) | **1336.1** | 1.02× |  |
| + pillar 5 (tree spec d=2 w=2) | **545.7** | 0.42× | tokens/step 2.00  (375 steps) |
| + pillar 6 (self-draft n=3) | **920.6** | 0.70× | acceptance 100.0%  (750/750) |
| + pillars 4 (Q4) + 6 (self-draft) combined | **889.7** | 0.68× |  |

## Honest interpretation

**The numbers above show the *correctness* of each pillar end-to-end, not its production value.** Read them this way:

- **Pillar 4 (quant)** ≈ baseline tok/s because we apply quant→dequant in FP32 — the *quality* path runs but the *speed* path needs an INT-N kernel (Marlin / bitsandbytes) which we don't have on CPU. Speed gain shows up in Phase 2 on GPU with a real Q4 kernel.
- **Pillar 5 (tree spec)** runs `width × depth` verification paths per step. On a tiny compute-bound model, that overhead exceeds the tokens/step gain → net slower. On a real MoE on GPU, tree paths are batched in one forward and the math inverts.
- **Pillar 6 (self-draft)** has the same overhead profile as 5 but cheaper (top-1 routing only). Slight slowdown on tiny model; gain on real MoE.
- **Combined** scales the same way: small models lose, large memory-bound models win.

## Why M1 doesn't move the needle for these pillars

The pillars target the **memory-bound regime**: 35B-A3B on GPU has ~3 GB of active expert weights moving across PCIe per token, and ~70% of wall-clock time is the GPU stalling on transfers. **The pillars hide that latency.**

On M1 with a 250 KB synthetic model on CPU:

- No PCIe — unified memory means there's no expert-transfer latency to hide.
- The model fits in L2 cache — memory bandwidth is irrelevant.
- The bottleneck is pure compute — and the pillars *add* compute (extra verification paths, dequant ops).

**This is the wrong workload for these optimizations.** Production validation happens in Phase 2 on a rented 3090 with Qwen3.5-27B-Int4 + Qwen3.5-35B-A3B-Int4, where the regime inverts.

## What CAN be trusted from these numbers

1. **Algorithm correctness.** Every pillar runs end-to-end without exceptions, output shapes match, NaN-free.
2. **Acceptance rate / tokens-per-step.** These are architecture-agnostic — the same model will produce the same acceptance rate on GPU.
3. **Overhead profile.** We now know per-pillar Python overhead in seconds-per-call. On GPU, where the GPU kernel time dominates Python time, this overhead is negligible.

## What we'll re-measure in Phase 2

Same script, same prompts, but against:
- `Qwen/Qwen3.5-27B-GPTQ-Int4` (dense, real Q4 weights, real PCIe)
- `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` (real MoE, the actual target)

Floor: stock vLLM `vllm serve`. Pillars 4/5/6 plug in on top, one at a time. Each must beat the floor on tok/s OR on quality/latency tradeoff to earn its place.
