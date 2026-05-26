# Dana Engine — Documentation

> Custom MoE inference engine built from scratch.
> **Goal:** Run Qwen3-235B-MoE at 30+ tokens/sec on a $20K workstation.

---

## Quick Start

```bash
cd dana/engine

# Install all packages (CPU-only, fast)
pip install -e tiered-tensor-store[test]
pip install -e expert-cache[test]
pip install -e moe-quant[test]
pip install -e moe-router-predict[test]
pip install -e spec-decode-tree[test]
pip install -e moe-self-draft[test]
pip install -e dana-engine[test]

# Run all tests
python -m pytest dana-engine/tests/ -v -s
```

For GPU work, see [GPU_TODO.md](../engine/GPU_TODO.md) or open [`dana_collab.ipynb`](../../dana_collab.ipynb).

---

## Architecture

```
dana/engine/
├── tiered-tensor-store/    Pillar 3: VRAM/RAM/SSD tensor hierarchy
├── expert-cache/           Pillar 2: Frequency + predictive caching
├── moe-quant/              Pillar 4: Per-expert Q2/Q4/Q8 quantization
├── moe-router-predict/     Pillar 1: Async expert prefetching
├── spec-decode-tree/       Pillar 5: Tree speculative decoding
├── moe-self-draft/         Pillar 6: Self-draft speculative decode
└── dana-engine/            Private integrator: wires all 6 pillars
    ├── src/dana_engine/
    │   ├── model/          TinyMoETransformer (synthetic test model)
    │   ├── batch/          ExpertGrouper + ExpertAwareBatchScheduler
    │   ├── speculative/    OptimizedSelfDraftRunner + OptimizedTreeRunner
    │   ├── api/            FastAPI server (OpenAI-compatible)
    │   ├── pipeline.py     DanaInferencePipeline — main entry point
    │   └── naive_inference.py  Greedy baseline
    └── tests/
        └── test_integration_benchmark.py
```

---

## Package Reference

### `tiered-tensor-store`

Three-tier storage hierarchy: VRAM → RAM → SSD.

```python
from tiered_tensor_store import TieredTensorStore

store = TieredTensorStore(base_dir="/tmp/dana")
store.store("expert_0", weight_tensor, tier="ram")
tensor = store.load("expert_0")      # auto-promotes to hot
store.evict("expert_0", to_tier="ssd")
stats = store.stats()                # access counts, tier distribution
```

**Tiers:**
| Tier | Storage | Latency | Notes |
|------|---------|---------|-------|
| `"hot"` | CPU tensor (→ CUDA when wired) | 0 ms | Resident |
| `"ram"` | numpy.memmap | ~1 ms PCIe | 512 GB DDR5 |
| `"ssd"` | torch.save/load file | ~5 ms NVMe | 2–4 TB |

---

### `expert-cache`

Frequency-aware and predictive caching, replacing naive LRU.

```python
from expert_cache import FrequencyExpertCache, PredictiveExpertCache

# Frequency cache (LFU + sliding-window recency)
cache = FrequencyExpertCache(capacity=32, window_size=1000)
cache.put(expert_id, weight_tensor)
tensor = cache.get(expert_id)        # None on miss
analytics = cache.analytics()        # hit rate, eviction count

# Predictive cache (hint-driven pre-loading)
pcache = PredictiveExpertCache(capacity=32)
pcache.hint([3, 5, 7])              # pre-load before needed
pcache.put_hinted(3, weight_tensor)
```

**Hit rate:** LRU ~60% → Predictive 90%+

---

### `moe-quant`

Per-expert quantization at Q2/Q4/Q8, with sensitivity-based tier assignment.

```python
from moe_quant import quantize, dequantize
from moe_quant.sensitivity import SensitivityAnalyzer
from moe_quant.tier_assigner import TierAssigner

qt = quantize(weight, bits=4)       # QuantizedTensor
restored = dequantize(qt)           # approximate reconstruction

# Sensitivity-based tier assignment
analyzer = SensitivityAnalyzer(model)
sensitivity = analyzer.analyze()    # {layer: {expert_id: sensitivity_score}}
assigner = TierAssigner()
tiers = assigner.assign(sensitivity) # {expert_id: bits}  (2, 4, or 8)
```

**Compression ratios:** Q8 = 4×, Q4 = 8×, Q2 = 16×

---

### `moe-router-predict`

Lightweight lookahead predictor + async DMA prefetch manager.

```python
from moe_router_predict.predictor import RouterPredictor
from moe_router_predict.async_loader import AsyncExpertLoader
from moe_router_predict.residency import ExpertResidencyTracker
from moe_router_predict.scheduler import PrefetchScheduler

predictor = RouterPredictor(model, num_steps=3)
predictions = predictor.predict(hidden_state, num_steps=3)
# predictions[i].expert_ids  — predicted expert IDs for step i+1
# predictions[i].confidence  — router softmax confidence

# Async loader (CPU→GPU when CUDA streams wired)
loader = AsyncExpertLoader()
loader.enqueue(expert_ids=[3, 5, 7], priority=1)
```

---

### `spec-decode-tree`

Tree-structured speculative decoding with adaptive depth/width.

```python
from spec_decode_tree.tree_spec import TreeSpecDecoder
from spec_decode_tree.acceptance import SpeculativeAcceptor
from spec_decode_tree.adaptive import AdaptiveTreeController

decoder = TreeSpecDecoder(model, depth=2, width=2)
tree = decoder.generate_tree(input_ids)   # candidate tree

acceptor = SpeculativeAcceptor(model)
accepted, meta = acceptor.verify(input_ids, tree)

# Adaptive controller tunes depth/width based on acceptance rate
controller = AdaptiveTreeController(max_depth=4, max_width=4)
controller.update(meta["acceptance_rate"])
depth, width = controller.current_depth, controller.current_width
```

---

### `moe-self-draft`

Self-draft speculative decoding using the same model in top-1 routing mode.

```python
from moe_self_draft.self_draft import MoeSelfDrafter
from moe_self_draft.verify import SelfDraftVerifier

drafter  = MoeSelfDrafter(model, num_active_override=1)  # sparse routing
verifier = SelfDraftVerifier(model)                       # full routing

draft = drafter.draft(input_ids, num_draft_tokens=4)
accepted = verifier.verify(input_ids, draft)
# len(accepted) / len(draft.token_ids) = acceptance rate
```

---

### `dana-engine` (private integrator)

```python
from dana_engine import DanaInferencePipeline, PipelineConfig
from dana_engine.model.config import TinyMoEConfig

pipeline = DanaInferencePipeline(PipelineConfig(
    model_config=TinyMoEConfig.tiny(),
    enable_spec_decode=True,
    num_draft_tokens=4,
    enable_prefetch=True,
    max_new_tokens=64,
))

result = pipeline.generate(input_ids)
# result.tokens_per_second
# result.avg_tokens_per_step
# result.spec_decode_used
```

#### Key classes

| Class | File | Purpose |
|-------|------|---------|
| `TinyMoETransformer` | `model/transformer.py` | Synthetic test model (MoE architecture) |
| `TinyMoEConfig` | `model/config.py` | Model hyperparameters. `.micro()` for tests |
| `DanaInferencePipeline` | `pipeline.py` | Main entry point, all 6 pillars |
| `PipelineConfig` | `pipeline.py` | Feature flags: spec_decode, prefetch, etc. |
| `OptimizedSelfDraftRunner` | `speculative/self_draft_runner.py` | Self-draft + stats |
| `OptimizedTreeRunner` | `speculative/tree_runner.py` | Adaptive tree decode |
| `ExpertGrouper` | `batch/expert_grouper.py` | Groups requests by expert overlap (Jaccard) |
| `ExpertAwareBatchScheduler` | `batch/scheduler.py` | Pull-based batch queue |

---

## Test Configurations

```python
TinyMoEConfig.micro()  # 2L, 4E, 64H — <1ms per forward, for unit tests
TinyMoEConfig.tiny()   # 4L, 8E, 256H — ~5MB model, for integration tests
```

---

## Benchmark Results (CPU, synthetic model)

Results from `test_integration_benchmark.py` on a standard laptop CPU:

| Stage | TPS | Notes |
|-------|-----|-------|
| Naïve greedy | baseline | 1 token/step |
| Self-draft (k=4) | ~same on CPU | Draft overhead ≈ verify savings |
| Tree (d=2, w=2) | varies | Adaptive depth |
| Full pipeline | ≥ naïve | Spec+prefetch combined |

> CPU benchmarks validate correctness only. Real gains require GPU hardware.
> See `GPU_TODO.md` for expected GPU numbers (×5–60 over naïve).

---

## Known Issues

| ID | Issue | Location | Impact |
|----|-------|----------|--------|
| BUG-1 | Tree verifier runs one forward per path | `spec-decode-tree/verify.py` | High |
| BUG-2 | `_prefetch_from_draft()` is a no-op | `pipeline.py:253` | Medium |
| BUG-3 | `VRAMBudgetManager` not wired to eviction | `pipeline.py` | Medium |
| BUG-4 | Synthetic tokenizer (garbage text output) | `api/server.py` | Low |
| BUG-5 | `AsyncExpertLoader` is CPU-only | `async_loader.py` | High |

---

## Roadmap

See [`NEXT_STEPS.md`](../engine/NEXT_STEPS.md) for the full priority-ordered list.

**Top 5 next actions:**
1. Fix `BUG-1` — batch tree path verification (5-min, high impact)
2. `P2a` — wire `.cuda()` into `tier_manager.py` hot tier
3. `P2b` — real async H2D with `torch.cuda.Stream()`
4. `P1a` — write `QwenMoEAdapter` for HuggingFace Qwen3-MoE
5. `P6`  — run Colab end-to-end smoke test

---

## Collaboration

Open `dana_collab.ipynb` (repo root) in Colab or Jupyter.
It contains runnable cells for every section above, plus all GPU TODO items.
