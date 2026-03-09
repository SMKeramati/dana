# GPU TODO — Run These on Colab / GPU Machine

Everything in this file requires real GPU hardware and/or large model downloads.
All other engine code is implemented and tested on CPU.

## Setup

```bash
# Install CUDA-enabled torch
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install all engine packages
cd /path/to/dana/engine
pip install -e tiered-tensor-store[test]
pip install -e expert-cache[test]
pip install -e moe-quant[test]
pip install -e moe-router-predict[test]
pip install -e spec-decode-tree[test]
pip install -e moe-self-draft[test]
pip install -e dana-engine[test]
```

---

## 1. PCIe Transfer Benchmarking

Measure actual RAM→VRAM transfer latency for expert weight tensors.
This is the core bottleneck the engine solves.

```python
import torch, time
# Simulate an expert weight tensor (typical MoE expert ~50MB)
tensor = torch.randn(4096, 4096)  # ~64MB FP32

# Warm up
tensor.cuda()

# Benchmark
times = []
for _ in range(100):
    t0 = time.perf_counter()
    tensor.cuda()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

print(f"RAM→VRAM: {sum(times)/len(times)*1000:.2f}ms avg")
# Expected: ~1-3ms on PCIe 4.0 x16
```

---

## 2. tiered-tensor-store GPU Mode

Test the `"hot"` tier actually uses GPU memory:

```python
from tiered_tensor_store import TieredTensorStore
store = TieredTensorStore(hot_budget_bytes=4 * 1024**3)  # 4GB VRAM

tensor = torch.randn(4096, 4096)
store.store("expert_0", tensor, tier="ram")
loaded = store.load("expert_0")  # should auto-promote to GPU
assert loaded.device.type == "cuda"
print(f"VRAM used: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
```

---

## 3. Download & Test Real MoE Model

```bash
# Download Qwen1.5-MoE-A2.7B (~14GB, or Q4_K_M GGUF ~8GB)
pip install huggingface_hub
huggingface-cli download Qwen/Qwen1.5-MoE-A2.7B --local-dir /tmp/qwen-moe
```

Adapt TinyMoETransformer interface to Qwen's architecture:
- Qwen has 60 experts, top-k=4, hidden_dim=2048
- Map `MoERouter` → Qwen's `MoeSparseMoeBlock`
- Run `RouterPredictor` against real Qwen router layers

---

## 4. Quantization Speed Benchmark

Compare dequant speed at each bit-width on GPU:

```python
from moe_quant import quantize, dequantize
import time, torch

expert_weight = torch.randn(4096, 4096).cuda()

for bits in [2, 4, 8]:
    qt = quantize(expert_weight.cpu(), bits=bits)
    qt_gpu = qt  # move quantized data to GPU

    times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        dequantize(qt_gpu)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg_us = sum(times)/len(times) * 1e6
    compression = expert_weight.element_size() * expert_weight.numel() / qt.nbytes
    print(f"Q{bits}: {avg_us:.1f}µs dequant, {compression:.1f}x compression")
```

---

## 5. Baseline vs Full Pipeline Benchmark

Compare tokens/sec with each optimization enabled:

```python
from dana_engine import DanaEngine, DanaConfig
from dana_engine.model.config import TinyMoEConfig

# Use real Qwen weights instead of tiny synthetic model
config = DanaConfig(
    model_path="/tmp/qwen-moe",
    hot_budget_bytes=20 * 1024**3,   # 20GB VRAM (2x RTX 4090)
    ram_budget_bytes=200 * 1024**3,  # 200GB RAM
    enable_prefetch=True,
    enable_spec_decode=True,
    enable_quant=True,
)

engine = DanaEngine(config)

# Benchmark: 100 tokens, measure time
import time
input_ids = torch.randint(0, 1000, (1, 32)).cuda()

for label, flags in [
    ("naive",      dict(enable_prefetch=False, enable_spec_decode=False, enable_quant=False)),
    ("+prefetch",  dict(enable_prefetch=True,  enable_spec_decode=False, enable_quant=False)),
    ("+cache",     dict(enable_prefetch=True,  enable_spec_decode=False, enable_quant=False)),
    ("+quant",     dict(enable_prefetch=True,  enable_spec_decode=False, enable_quant=True)),
    ("+spec",      dict(enable_prefetch=True,  enable_spec_decode=True,  enable_quant=True)),
]:
    for k, v in flags.items():
        setattr(config, k, v)
    t0 = time.perf_counter()
    result = engine.generate(input_ids, max_new_tokens=100)
    elapsed = time.perf_counter() - t0
    tps = 100 / elapsed
    print(f"{label:15s}: {tps:.1f} tok/s")
```

Expected progression (from the plan):
- naive: ~0.5 TPS
- +prefetch: ~2.5 TPS
- +cache: ~6 TPS
- +quant: ~15 TPS
- +spec: ~40-60 TPS

---

## 6. Expert Activation Heatmap

Which of Qwen's 60 experts are hot across Persian/English prompts?

```python
from moe_router_predict import RouterPredictor
import json

prompts = [
    "سلام، چطوری؟",          # Persian greeting
    "Hello, how are you?",    # English greeting
    "برنامه‌نویسی پایتون چیست؟",  # Persian technical
    "What is Python programming?",  # English technical
]

expert_counts = {}
predictor = RouterPredictor(model)

for prompt in prompts:
    tokens = tokenizer.encode(prompt)
    hidden = model.embed(tokens)
    predictions = predictor.predict(hidden, num_steps=len(tokens))
    for pred in predictions:
        for eid in pred.expert_ids:
            expert_counts[eid] = expert_counts.get(eid, 0) + 1

# Print top-10 hot experts
sorted_experts = sorted(expert_counts.items(), key=lambda x: -x[1])
print("Top 10 hot experts:", sorted_experts[:10])
```

---

## 7. CUDA Kernel: Tree Verification

The reference `spec-decode-tree` uses pure Python. For production speed,
implement a CUDA kernel for the verification step.

File to create: `dana-engine/csrc/tree_verify.cu`

The kernel needs to:
1. Accept flattened tree candidates (batch_size × tree_size × vocab_size logits)
2. Apply acceptance criterion in parallel across all tree nodes
3. Return the longest valid path per sequence

See `spec-decode-tree/src/spec_decode_tree/verify.py` for the reference algorithm.

---

## 8. Production API Stress Test

```bash
pip install locust

# locustfile.py
from locust import HttpUser, task
class DanaUser(HttpUser):
    @task
    def chat(self):
        self.client.post("/v1/chat/completions", json={
            "model": "qwen-moe",
            "messages": [{"role": "user", "content": "سلام"}],
            "max_tokens": 100,
        })

# Run with 10 concurrent users
locust -f locustfile.py --host=http://localhost:8000 --users 10 --spawn-rate 2
```

Target: 20-30 TPS/user at 10 concurrent users on 2x RTX 4090 + 512GB DDR5.

---

## 9. GPU Memory Profiling

```python
import torch
torch.cuda.memory._record_memory_history()

# Run inference...
result = engine.generate(input_ids, max_new_tokens=50)

snapshot = torch.cuda.memory._snapshot()
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
# Open in Chrome at chrome://tracing or use torch._C._cuda_memorySnapshot()
```

---

## Notes

- All CPU-based tests in `engine/*/tests/` pass locally — no GPU needed
- GPU tests above are integration/perf tests, not correctness tests
- The tiny synthetic MoE model is architecture-compatible with Qwen-style MoE
- Adapt `dana_engine.model.transformer.TinyMoETransformer` → `QwenMoEAdapter` when
  connecting to real Qwen weights
