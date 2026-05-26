# Dana Engine — Next Steps

Priority-ordered list of what to build, test, and fix next.
Each item includes: what it is, why it matters, and how to do it.

---

## Priority 1 — Validate core claims on real hardware

These are blockers before any performance number can be trusted.

### 1a. Attach a real HuggingFace MoE model

**Why:** Every TPS number we have is from a 1MB synthetic model. The memory-bandwidth
bottleneck that justifies all six pillars only exists on models with billions of
parameters.

**What to do:**
- Write a `ModelAdapter` that wraps any `transformers` model exposing `MoELayer`
  hooks (Qwen3-MoE, DeepSeek-V3, Mixtral)
- Map HuggingFace expert weight tensors into `tiered-tensor-store`
- Replace the synthetic tokenizer in `api/server.py` with the model's real tokenizer

**Best candidate:** `Qwen/Qwen3-30B-A3B` (fits A100 80GB, real MoE, free Colab tier).

---

### 1b. Measure real PCIe bandwidth and expert load latency

**Why:** The ×5 prefetch gain is predicated on PCIe being the bottleneck.
We need to measure it, not assume it.

**What to do:**
```python
import time, torch
x = torch.randn(expert_size).pin_memory()
t0 = time.perf_counter()
y = x.cuda()
torch.cuda.synchronize()
print(f"{expert_size * 4 / (time.perf_counter() - t0) / 1e9:.1f} GB/s")
```
Run this for each expert size and compare to PCIe theoretical max (A100: ~32 GB/s
practical over PCIe 4.0 × 16).

---

### 1c. Measure naïve vs prefetch TPS on real model

**Sequence:**
1. Load Qwen3-30B-A3B, put all experts on CPU RAM
2. Run 50-token greedy generation, record wall-clock TPS
3. Enable `moe-router-predict` prefetcher, repeat
4. Compare: if prefetch gain < ×2, the bottleneck is elsewhere (compute or CPU-GPU sync)

---

## Priority 2 — Fix CPU/GPU tier gap

### 2a. Wire CUDA tensors into `tiered-tensor-store`

**Why:** Currently the "hot tier" is a CPU tensor. Nothing actually goes to GPU.

**What to do:** In `tier_manager.py`, replace:
```python
# current hot tier store
self._hot[key] = tensor
```
with:
```python
self._hot[key] = tensor.cuda() if torch.cuda.is_available() else tensor
```
And wire `AsyncExpertLoader` to use `tensor.cuda(non_blocking=True)` + CUDA
streams for concurrent DMA.

---

### 2b. Implement real async H2D in `AsyncExpertLoader`

**Why:** The current `async_loader.py` has a TODO comment where the GPU transfer
should be. Without this, prefetching is a no-op.

**What to do:**
```python
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    gpu_tensor = cpu_tensor.cuda(non_blocking=True)
# Store stream handle so caller can synchronize before use
```

---

### 2c. CUDA kernel for Q4/Q2 dequantization

**Why:** `moe-quant` currently unpacks with numpy on CPU. On GPU, dequant must
happen on the GPU side. Otherwise every Q4 expert requires round-trip to CPU
for unpacking.

**What to do:** Write a minimal Triton kernel or use `bitsandbytes` INT4 kernels
for the dequant step. At expert-level granularity (≈1–4MB per expert at Q4)
this can overlap with compute.

---

## Priority 3 — Spec decode tuning

### 3a. Measure acceptance rate on real model

**Why:** Our benchmark shows 100% acceptance on the tiny model (deterministic
routing). Real acceptance rates drive all spec decode ROI calculations.

**What to do:**
- Generate 200 prompts from a real dataset (e.g. OpenHermes)
- Run self-draft with num_draft_tokens=4, record per-prompt acceptance rate
- Plot distribution. Expect 55–80% for in-distribution prompts, lower for OOD

**If acceptance < 50%:** Reduce `num_draft_tokens` to 2; consider larger draft model.

---

### 3b. Cap tree spec decode depth/width adaptively

**Why:** The adaptive controller scaled to depth=4, width=4 = 256 paths at
100% acceptance. On GPU, batched tree verification handles this fine. On CPU
it kills TPS. Add a computational budget cap:

```python
max_verify_cost = 32  # max total paths
actual_paths = width ** depth
if actual_paths > max_verify_cost:
    # reduce depth first, then width
```

---

### 3c. Evaluate draft model alternatives

Current: self-draft (same model, top-1 routing). Alternatives worth testing:
- **Layer-skipping draft**: run only first N layers as draft
- **Smaller MoE draft**: use 3B-A0.3B as draft for 30B-A3B target
- **Medusa heads**: add parallel prediction heads to the target model

---

## Priority 4 — Expert cache tuning

### 4a. Profile real expert access frequency distribution

**Why:** `ExpertClassifier` hot/warm/cold thresholds are currently defaults.
Real MoE models show strong power-law distribution (top 5% experts used 70%+
of the time).

**What to do:**
- Run 500-token generation on 50 prompts
- Log every `(layer, expert_id)` activation
- Fit power-law to access counts
- Auto-tune cache thresholds using `CacheAnalytics.suggest_thresholds()`

---

### 4b. Validate Jaccard batching assumption

**Why:** `ExpertGrouper` uses Jaccard similarity of predicted expert sets.
This is only useful if requests from similar domains actually share experts.

**What to do:**
- Sample 100 pairs of (coding prompt, coding prompt) and (coding prompt, math prompt)
- Measure predicted expert overlap for each pair
- Verify that same-domain pairs have higher Jaccard than cross-domain pairs
- Threshold tune: if same-domain Jaccard < 0.3, batching wins are minimal

---

## Priority 5 — Production hardening

### 5a. Replace synthetic tokenizer

`api/server.py` encodes prompts as random integers and decodes tokens as digit
strings. Replace with:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
```
Map the real vocabulary through `tiered-tensor-store`'s embedding table.

---

### 5b. Concurrent request handling

`DanaInferencePipeline` is not thread-safe for concurrent requests. The
`ExpertAwareBatchScheduler` was built for this but isn't wired to the pipeline.

**What to do:**
- Add a worker thread that pulls from `scheduler.next_batch()` and runs
  batched inference
- Expose a `submit_async(request)` method on the pipeline
- Add concurrency stress test with `httpx.AsyncClient` firing 20 parallel requests

---

### 5c. Memory budget enforcement

`VRAMBudgetManager` exists but is not connected to the pipeline's runtime
eviction loop. When VRAM fills up, the pipeline currently has no eviction
pressure signal.

**What to do:**
- Register each expert load with `VRAMBudgetManager.register(expert_id, size_bytes)`
- On each step, call `enforce()` to evict cold experts before loading hot ones
- Monitor `cache_analytics.eviction_count` to tune budget

---

## Priority 6 — Colab validation (end-to-end smoke test)

Recommended test sequence for a free A100 Colab session (40GB):

```
Step 1: pip install all packages (15 min)
Step 2: Load Qwen3-30B-A3B (8 min download)
Step 3: Measure naive TPS (baseline)
Step 4: Enable prefetch only → measure TPS delta
Step 5: Enable expert cache → measure TPS delta
Step 6: Enable Q8 quantization → measure VRAM freed and TPS delta
Step 7: Enable self-draft (num_draft_tokens=3) → measure acceptance rate and TPS
Step 8: Enable all → final TPS
Step 9: Run 10 concurrent requests via API → throughput/user
```

**Expected outcome:**
- If PCIe is the bottleneck: prefetch alone gives ×3–5
- If compute is the bottleneck: spec decode gives ×2–3, prefetch gives <×1.5
- Combined should reach ×5–15 on 30B-A3B vs naïve

---

## Known Issues / Bugs to Fix

| Issue | Location | Impact |
|---|---|---|
| Tree verifier runs one forward per path (should batch) | `spec-decode-tree/verify.py` | High — tree TPS collapses on CPU |
| `_prefetch_from_draft()` is a no-op | `pipeline.py:253` | Medium — prefetch has no effect yet |
| `VRAMBudgetManager` not connected to eviction loop | `pipeline.py` | Medium — unbounded VRAM growth |
| Synthetic tokenizer produces garbage text output | `api/server.py` | Low (test model only) |
| `AsyncExpertLoader` double-buffer logic is CPU-only | `moe-router-predict/async_loader.py` | High — no real async on GPU |

---

## Quick Win — Tree Verifier Batching

The biggest immediate code improvement: batch tree path verification.

Current (one forward per path):
```python
for path in tree_paths:
    logits = model(torch.cat([context, path], dim=1))
```

Correct (one batched forward for all paths):
```python
batch = torch.stack([torch.cat([context, p], dim=1) for p in tree_paths])
logits = model(batch)  # (num_paths, seq_len, vocab)
```

This alone would make tree spec decode viable on CPU and dramatically faster on GPU.
