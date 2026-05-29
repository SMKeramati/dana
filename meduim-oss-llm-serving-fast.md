# Fastest OSS LLM Serving for ≤30B Models on GPU Servers (2025–2026)

## TL;DR
- **Best general-purpose throughput today (≤30B dense models on H100/L40S/4090):** SGLang ≥ vLLM ≈ LMDeploy/TurboMind > TensorRT‑LLM (faster but harder to operate); for single-user batch‑1 on a consumer GPU, ExLlamaV2 (EXL2 4 bpw) via TabbyAPI still leads.
- **The single biggest practical wins:** FP8 W8A8 (Hopper/Ada) or INT4 AWQ/GPTQ with Marlin/Machete kernels, FP8 KV‑cache, continuous batching + chunked prefill + prefix caching (defaults in vLLM V1 / SGLang), and EAGLE‑3 speculative decoding — Red Hat reports vLLM "boosting inference performance by up to 2.5X across diverse scenarios" [Red Hat](https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding) (developers.redhat.com, July 1 2025).
- **TGI is in maintenance mode as of 12/11/2025** — Hugging Face's own docs recommend migrating to vLLM or SGLang. [TECHSY](https://techsy.io/en/blog/vllm-vs-sglang) [GitHub](https://github.com/huggingface/hf-endpoints-documentation/blob/main/docs/source/engines/tgi.md) For multi‑node / disaggregated serving above a single box, NVIDIA Dynamo (GTC 2025) [NVIDIA Developer](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) and llm‑d (CNCF sandbox, May 2025) [llm-d](https://llm-d.ai/blog/llm-d-announce) are the emerging orchestration layers.

---


## Key Findings

1. **Framework hierarchy for ≤30B on a single GPU (H100/L40S/4090):**
   - **SGLang** wins on prefix‑heavy workloads (RAG, multi‑turn chat, agents) and structured output. On Llama‑3.1‑8B‑Instruct with 1,000 ShareGPT prompts on H100, PremAI (via AIMultiple) measured **SGLang 16,215 tok/s vs vLLM 12,553 tok/s — a 29.2% advantage**, and SGLang reports up to 6.4× over baselines on heavy prefix-reuse. [ChatForest](https://chatforest.com/reviews/sglang-structured-generation-llm-serving/) [Qiyanjun](https://qiyanjun.github.io/2025sp-GenAI-overview//contents/S0-L21/) Tail TTFT p95 5–8% lower than vLLM at every concurrency tested (techsy.io H100 benchmark).
   - **vLLM V1** (vLLM 0.11+ has fully removed V0) [Vllm](https://vllm.ai/blog/2025-12-17-large-scale-serving) is the best general‑purpose default — broadest model/hardware/quant coverage, biggest community.
   - **TensorRT‑LLM** typically wins an additional 15–30% over vLLM on NVIDIA hardware when properly compiled, but demands engine builds and per‑shape tuning.
   - **LMDeploy/TurboMind** is the dark horse for INT4 (AWQ/GPTQ) workloads on H100/4090/L40S — the TurboMind paper (arXiv:2508.15601, Aug 2025) reports "up to 61% lower serving latency (30% on average) and up to 156% higher throughput (58% on average)" [arXiv](https://arxiv.org/abs/2508.15601) vs other mixed‑precision frameworks, [arXiv](https://arxiv.org/html/2508.15601v2) and 13–31% over vLLM+MARLIN on the same models. [arXiv](https://arxiv.org/pdf/2508.15601) Single-batch RTX 4090 W4A16: Llama‑2‑7B 206 tok/s, Llama‑2‑13B 116 tok/s. [readthedocs](https://lmdeploy.readthedocs.io/en/latest/quantization/w4a16.html)
   - **ExLlamaV2 + TabbyAPI** still beats every server engine at strict batch=1 on a single consumer GPU; turboderp's official README shows ~**211 tok/s for Llama‑2‑7B EXL2 4 bpw on RTX 4090**, ~114 tok/s for Llama‑13B GPTQ.
   - **llama.cpp / llama‑server** is the most portable and the only practical choice for CPU/Apple Silicon or hybrid CPU+GPU; ~95–110 tok/s on Llama‑3.1‑8B Q4_K_M on RTX 4090 [SitePoint](https://www.sitepoint.com/mac-m3-max-vs-rtx-4090-local-llm-performance-showdown-2026/) single user.
   - **MLC‑LLM** is fast but ecosystem activity thinned in 2025; remains a good single‑user option on heterogeneous backends (WebGPU, Vulkan, Metal).
   - **Aphrodite Engine** is a vLLM fork with broader sampler/quantization support (AQLM, BitNet, QuIP#, ExLlamaV3, MXFP4, [GitHub](https://github.com/aphrodite-engine/aphrodite-engine) etc.) — favored by RP/community; slightly behind upstream vLLM on raw speed.
   - **TGI** — maintenance mode since Dec 11 2025; HF docs explicitly recommend vLLM or SGLang. [GitHub](https://github.com/huggingface/text-generation-inference)

2. **Quantization (≤30B):**
   - **FP8 W8A8** (vLLM/SGLang/LMDeploy/TRT‑LLM) on Hopper/Ada: vLLM docs claim "2× reduction in model memory requirements and up to a 1.6x improvement in throughput with minimal impact on accuracy." [vLLM](https://docs.vllm.ai/en/latest/features/quantization/fp8/) Neural Magic's "Give Me BF16 or Give Me Death?" study (arXiv:2411.02355v3) calls W8A8‑FP "essentially lossless."
   - **AWQ‑INT4** with Marlin (Ampere) / Machete (Hopper) kernels: best perf/$ for high‑concurrency synchronous serving. Per ermolushka's RTX 4090 vLLM benchmark, "AWQ excels in inference speed, delivering 579.1 tokens/second — a 70% improvement over FP16" [Ermolushka](https://ermolushka.github.io/posts/vllm-benchmark-4090/) (Llama‑3‑8B class). Neural Magic's Machete blog: on Llama‑3.1‑70B-W4A16 on one H100, "up to 5 user requests per second while maintaining a median TTFT of <250 ms and a median TPOT of <100 ms."
   - **GPTQ‑INT4**: essentially interchangeable with AWQ in vLLM/SGLang/LMDeploy via the same Marlin/Machete paths.
   - **GGUF Q4_K_M / IQ4_XS** is llama.cpp‑only; Unsloth UD‑Q4_K_XL is competitive on quality but lags dedicated GPU stacks on throughput.
   - **EXL2** (ExLlamaV2/V3): best quality‑per‑bit at 2.5–4 bpw on consumer GPUs; single‑GPU, single‑user only.
   - **FP4 / NVFP4 / MXFP4**: useful only on Blackwell (B200, RTX 5090, RTX PRO 6000). On older GPUs vLLM silently falls back to Marlin and you lose the speedup. [GitHub](https://github.com/vllm-project/vllm/issues/30135) On Hopper, set `VLLM_USE_FLASHINFER_MOE_FP8=0` so DeepGEMM wins selection (FlashInfer is Blackwell‑tuned). [GitHub](https://github.com/vllm-project/vllm/issues/34249)
   - **KV‑cache FP8** in vLLM nearly halves decode ITL slope vs BF16 on Llama‑3.1‑8B (vLLM blog: "fitted ITL slope drops from 4.37e‑05 to 2.37e‑05 ms/token," [vLLM Blog](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html) "14.9% output throughput increase at concurrency 8"). [Vllm](https://vllm.ai/blog/2026-04-22-fp8-kvcache) SGLang supports FP8 E5M2/E4M3 and FP4 KV; warning per SGLang docs — if the attention backend doesn't fuse dequant, perf can be "extremely slow."

3. **Speculative decoding:**
   - **EAGLE‑3 is the SOTA.** vLLM supports it from v0.8.5+ (CUDA graphs added in v0.9.1). Per Marques/Flynn/Kurtz (Red Hat Developer, July 1 2025): "boosting inference performance by up to 2.5X across diverse scenarios"; "latency of each request is reduced by up to 1.8x for the 8B model … 1.6x for the 70B model at low request rates" [redhat](https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding) (Llama‑3‑8B on 1×A100; Llama‑3.3‑70B on 4×A100). Speedup is workload‑dependent: math/RAG ~2.1×, translation ~0×.
   - SGLang, TensorRT‑LLM, and LMDeploy all support EAGLE/EAGLE‑3 (TRT‑LLM also has Medusa, ReDrafter, [GitHub](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html) MTP, Lookahead, n‑gram, draft‑model).
   - vLLM intentionally **does not support tree decoding** — it hurts at higher concurrency [Red Hat](https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding) (Red Hat figure 5).
   - **Speculators v0.3.0** (vLLM blog, Dec 13 2025) standardizes EAGLE/EAGLE‑3/HASS draft‑model training/serving: "reduce model latency by 1.5–3x." [vllm](https://blog.vllm.ai/2025/12/13/speculators-v030.html)

4. **Attention/runtime infra:**
   - **PagedAttention + continuous batching**: default in vLLM, SGLang, LMDeploy, TGI, Aphrodite.
   - **Chunked prefill**: default in vLLM V1 and SGLang; vLLM V1's unified scheduler treats prefill/decode uniformly [Red Hat](https://developers.redhat.com/articles/2025/01/28/vllm-v1-a-major-upgrade-vllms-core-architecture) (vLLM blog, Jan 27 2025).
   - **Prefix caching (APC) / RadixAttention**: vLLM uses hash‑based prefix caching; [Squeezebits](https://blog.squeezebits.com/37065) SGLang's RadixAttention is a multi-level radix tree [ChatForest](https://chatforest.com/reviews/sglang-structured-generation-llm-serving/) and the reason SGLang dominates on shared‑prompt workloads — LMSYS reports up to 6.4× over baselines.
   - **FlashAttention 3** is default on Hopper in vLLM v0.6+ and SGLang v0.4+; FA2 still used on Ampere/Ada. Per NVIDIA: "FlashAttention‑3 achieved 1.5–2.0x faster performance than FlashAttention‑2 with FP16, reaching up to 740 TFLOPS, and up to 1.2 PFLOPS with FP8." [NVIDIA Developer](https://developer.nvidia.com/blog/next-generation-of-flashattention/) FlashAttention‑4 (arXiv:2603.05451) targets Blackwell B200/GB200. [arXiv](https://arxiv.org/html/2603.05451v1)
   - **FlashInfer** kernels for FP4/FP8 MoE on Blackwell; on Hopper, DeepGEMM/Marlin/Machete usually win.
   - **CUDA graphs / torch.compile**: enabled by default in vLLM V1 and SGLang; major decode‑side win for small models where CPU overhead dominates.

5. **Parallelism, LoRA, disaggregation:**
   - **Tensor parallelism** is standard everywhere. **Pipeline parallelism**: vLLM, SGLang, TRT‑LLM. **Expert parallelism**: vLLM `--enable-expert-parallel`, SGLang large‑scale EP, TRT‑LLM, LMDeploy — mostly matters above 30B (MoE).
   - **Multi‑LoRA** is mature in vLLM (Punica SGMV kernel with hash‑aware KV cache integration), [Squeezebits](https://blog.squeezebits.com/37065) SGLang, LMDeploy, TGI. ExLlamaV2/TabbyAPI supports multi‑LoRA with independent scaling [GitHub](https://github.com/gittb/tabbyAPI-function) for single‑user.
   - **P/D disaggregation**: production‑grade in SGLang (since v0.4.7, June 2025) [ChatForest](https://chatforest.com/reviews/sglang-structured-generation-llm-serving/) and LMDeploy (via DLSlime/Mooncake, June 2025); orchestrated above any engine by **NVIDIA Dynamo** (GTC 2025) and **llm‑d** (CNCF sandbox, May 2025). [Rafay](https://rafay.co/ai-and-cloud-native-blog/nvidia-dynamo-turning-disaggregated-inference-into-a-production-system) Below ~30B on one GPU, disaggregation is unnecessary.

6. **Structured output:**
   - **xgrammar** is the default fast backend for SGLang and vLLM since late 2024. Per Particula's 2026 benchmark of SGLang vs vLLM, SGLang delivers "3x faster constrained decoding and 96–98% compliance rate" [Particula Tech](https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison) via xgrammar's optimized FSM masking. Outlines and lm‑format‑enforcer remain available as alternates.

7. **Emerging projects worth watching:**
   - **NVIDIA Dynamo** — orchestration above vLLM/SGLang/TRT‑LLM for disaggregated, KV‑aware, multi‑node serving. [Rafay](https://rafay.co/ai-and-cloud-native-blog/nvidia-dynamo-turning-disaggregated-inference-into-a-production-system)
   - **llm‑d** — Kubernetes‑native distributed inference on vLLM; CNCF sandbox. [GitHub](https://github.com/llm-d/llm-d)
   - **ik_llama.cpp** — llama.cpp fork with SOTA quant types (IQ4_NL, Q8_K_R8), FlashMLA, and a "split‑mode graph" tensor‑parallel multi‑GPU mode reported as 3–4× faster than upstream layer/row splits. [Medium](https://medium.com/@jagusztinl/llama-cpp-performance-breakthrough-for-multi-gpu-setups-04c83a66feb2)
   - **ExLlamaV3** — successor to ExLlamaV2; Marlin‑inspired GEMM kernel near‑memory‑bound at 4 bpw on RTX 4090. [GitHub](https://github.com/turboderp-org/exllamav3)
   - **Speculators v0.3** (Red Hat / vLLM) — standardized HF‑format training/serving of EAGLE/EAGLE‑3/HASS draft models.
   - **P‑EAGLE** (AWS, late 2025) — parallel drafter; "up to 1.69x speedup over vanilla EAGLE‑3 on GPT‑OSS 20B over MT‑Bench, HumanEval, and SpeedBench" [AWS](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) on B200.

---

## Details

### Framework comparison (≤30B, single H100 80GB or 1× RTX 4090, late 2025/early 2026)

| Engine | Best at | Throughput vs vLLM (8B‑class) | TTFT p95 | Ease of use | Key quants |
|---|---|---|---|---|---|
| **vLLM V1** | General default; broadest models/hardware/quants | 1.0× | baseline | Easy (1 cmd, OpenAI API) | FP8, AWQ, GPTQ, BNB, GGUF (limited), FP4 (Blackwell) |
| **SGLang** | Prefix‑heavy chat/RAG/agents, structured output | **1.29× on 8B**; up to 6.4× on prefix workloads | 5–8% lower | Easy | FP4/FP8/INT4/AWQ/GPTQ [PyPI](https://pypi.org/project/sglang/) |
| **TensorRT‑LLM** | Max raw NVIDIA throughput; engine‑build OK | ~1.15–1.30× post engine‑build | best steady state | Hard | FP8, FP4/NVFP4, AWQ, GPTQ, SmoothQuant INT8 |
| **LMDeploy/TurboMind** | INT4 weight‑only on H100/4090/L40S | +13–31% over vLLM+Marlin; up to 1.8× on persistent batching | competitive | Moderate | AWQ, GPTQ, FP8, MXFP4, INT4/INT8 KV |
| **ExLlamaV2 + TabbyAPI** | Single‑GPU, single‑user, INT4 batch=1 | ~211 tok/s 7B @ 4090 | best at batch=1 | Easy | EXL2 (2–8 bpw), GPTQ |
| **llama.cpp / llama‑server** | CPU/Apple/hybrid; portability | slower at multi‑user | great at batch=1 | Easiest | GGUF (Q2–Q8, IQ, K) |
| **MLC‑LLM** | Cross‑backend single‑user (WebGPU/Vulkan/Metal) | ≈ vLLM batch=1 | great at batch=1 | Moderate | Q4F16, AWQ |
| **Aphrodite Engine** | vLLM fork with extra quants/samplers | ≈ vLLM (sometimes behind) | ≈ vLLM | Easy | Almost everything (AQLM, BNB, ExLlamaV3, MXFP4…) [GitHub](https://github.com/aphrodite-engine/aphrodite-engine) |
| **TGI** | HF ecosystem (legacy) | ~0.8–1.0× | similar | Easy | FP8, AWQ, GPTQ, EETQ — **maintenance mode since 12/11/2025** [Hugging Face](https://huggingface.co/docs/inference-endpoints/engines/tgi) |

### Speed‑up techniques and framework support

| Technique | vLLM V1 | SGLang | TRT‑LLM | LMDeploy | TGI | llama.cpp | ExLlamaV2 |
|---|---|---|---|---|---|---|---|
| PagedAttention / continuous batching | ✅ default | ✅ default | ✅ | ✅ persistent | ✅ | partial | ✅ (FA 2.5+) |
| Chunked prefill | ✅ default | ✅ default | ✅ | ✅ split‑fuse | ✅ | – | – |
| Prefix caching / APC | ✅ hash | ✅ RadixAttention (best) | ✅ | ✅ | ✅ | basic | smart prompt cache |
| FlashAttention 2 / 3 | ✅ auto | ✅ auto | ✅ XQA | ✅ | ✅ | own | ✅ |
| FlashInfer | ✅ Blackwell MoE | ✅ | – (own) | partial | partial | – | – |
| CUDA graphs / torch.compile | ✅ default | ✅ default | ✅ engine | ✅ | partial | – | ✅ |
| Tensor parallelism | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (ik split-graph) | ✅ |
| Pipeline parallelism | ✅ | ✅ | ✅ | ✅ | partial | – | – |
| Expert parallelism (MoE) | ✅ | ✅ large‑scale EP | ✅ | ✅ | – | – | – |
| FP8 W8A8 | ✅ | ✅ | ✅ | ✅ | ✅ | – | – |
| AWQ Marlin/Machete | ✅ | ✅ | ✅ | ✅ | ✅ | – | – |
| GPTQ‑INT4 | ✅ | ✅ | ✅ | ✅ | ✅ | – | ✅ legacy |
| GGUF | partial | – | – | – | – | ✅ native | – |
| FP4/NVFP4/MXFP4 (Blackwell) | ✅ | ✅ | ✅ | ✅ MXFP4 | – | – | – |
| INT8 SmoothQuant | ✅ | ✅ | ✅ | ✅ | ✅ | – | – |
| KV‑cache quant | ✅ FP8 E4M3/E5M2 | ✅ FP8/FP4 | ✅ FP8/INT8 | ✅ INT8/FP8 | partial | ✅ Q8/Q4 | ✅ Q4/Q8 |
| Speculative decoding | ✅ EAGLE‑1/3, MLP, n‑gram, draft | ✅ EAGLE, MTP, Medusa | ✅ EAGLE‑1/2/3, Medusa, ReDrafter, MTP, Lookahead, n‑gram | ✅ MTP, draft | ✅ draft | ✅ draft | ✅ draft |
| Multi‑LoRA | ✅ Punica/SGMV | ✅ | ✅ | ✅ | ✅ | basic | ✅ |
| Structured output (xgrammar) | ✅ default | ✅ default (≈3× vLLM) | partial | basic | basic | basic | – |
| P/D disaggregation | ✅ via Dynamo/llm‑d | ✅ native | ✅ | ✅ DLSlime/Mooncake | – | – | – |

### "If you want max speed for X on Y, do this"

- **7B–14B dense (Llama‑3‑8B, Qwen2.5‑14B, Mistral‑7B, Gemma‑2‑9B) on 1× H100 (80 GB):**
  Run **vLLM V1**: `vllm serve <model> --quantization fp8 --kv-cache-dtype fp8 --enable-prefix-caching --speculative-config '{"model":"RedHatAI/<model>-speculator.eagle3","method":"eagle3","num_speculative_tokens":5}'`. Expect 1.5–2.5× over BF16 baseline (Red Hat). Or **SGLang**: `python -m sglang.launch_server --quantization fp8 --enable-torch-compile` — preferred for shared prompts (29% throughput edge on 8B). For absolute single‑stream latency, **TensorRT‑LLM** with FP8 + EAGLE‑3 on a fixed‑shape engine.

- **27B–32B dense (Gemma‑2‑27B, Qwen2.5‑32B) on 1× H100:**
  Same as above with FP8 W8A8. If memory is tight, switch to AWQ‑INT4 via vLLM (Machete) or LMDeploy TurboMind W4A16 — LMDeploy often wins at high concurrency.

- **7B–14B on 1× RTX 4090 (24 GB):**
  Multi‑user serving: **vLLM AWQ‑INT4** or **LMDeploy TurboMind AWQ** (~500–600 tok/s aggregate; ermolushka measured 579 tok/s for AWQ vs 340 tok/s for FP16). Single user, max tok/s: **ExLlamaV2 + TabbyAPI**, EXL2 4–5 bpw, paired with a small draft (e.g., Llama‑3.2‑1B EXL2 4.5 bpw) for spec decoding (~200+ tok/s on 7B at batch=1). CPU+GPU hybrid: **llama.cpp / ik_llama.cpp** with Q4_K_M or IQ4_XS.

- **13B–14B on 1× RTX 5090 / RTX PRO 6000 / L40S:**
  **vLLM V1** with FP8 (Ada/Blackwell) or AWQ. On Blackwell, enable FlashInfer MoE FP8/FP4 paths and consider MXFP4 (gpt‑oss). On RTX PRO 6000, MXFP4 currently falls back to Marlin (known vLLM issue #30135). [GitHub](https://github.com/vllm-project/vllm/issues/30135)

- **70B‑quant (Llama‑3.3‑70B) on 1× H100:**
  AWQ‑INT4 (LMDeploy TurboMind or vLLM Machete). FP8 W8A8 also fits 80 GB but leaves less KV headroom; pair with `--kv-cache-dtype fp8`.

### Gotchas / fine print

- **vLLM AWQ/GPTQ doesn't appear to save VRAM** at the engine level — vLLM preallocates KV cache to `gpu_memory_utilization` (default 0.9). [Ermolushka](https://ermolushka.github.io/posts/vllm-benchmark-4090/) The real benefit is bigger KV / more concurrency.
- **FP8 KV cache in vLLM defaults to per‑tensor scale = 1.0 (uncalibrated)** — calibrate via `llm-compressor` for production accuracy. [vLLM](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) vLLM docs: "Per-tensor (scalar) scaling factors are supported. Development is ongoing to support scaling factors of a finer granularity."
- **FP8 attention's two‑level accumulator slows prefill for head_dim > 128** vs BF16 — verify on your model [vLLM Blog](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html) (vLLM blog, FP8 KV‑cache).
- **SGLang quantized KV cache only pays off when the attention backend fuses dequant**; otherwise it can be slower than BF16 [SGLang](https://sgl-project-sglang-93.mintlify.app/optimization/quantized-kv-cache) (SGLang docs).
- **TensorRT‑LLM engines are shape‑specialized** — wrong `max_input_len`/`max_batch_size` makes a "fast" engine slow.
- **EAGLE‑3 speculators currently cap at 2048 context** for many released checkpoints — check the speculator card.
- **vLLM intentionally does not support tree‑style spec decoding** (perf collapses past synchronous request rates).
- **ik_llama.cpp's `-rtr` repacking forces CPU matmul** for some k‑quants — disable for hybrid GPU offload.
- **TGI is in maintenance mode** (12/11/2025); HF Inference Endpoints recommend vLLM or SGLang. [GitHub](https://github.com/huggingface/hf-endpoints-documentation/blob/main/docs/source/engines/tgi.md)

---

## Recommendations (staged)

1. **Default stack today (≤30B, NVIDIA single GPU):** vLLM V1 (≥0.11) with FP8 W8A8 on Hopper/Ada (or AWQ‑INT4 on Ampere/4090), FP8 KV cache, prefix caching on, chunked prefill on, CUDA graphs on, EAGLE‑3 spec decoding via a Speculators (RedHatAI) draft for your verifier.
2. **If traffic is prefix‑heavy (chat, RAG, agents):** switch to SGLang for the RadixAttention win + xgrammar structured output. Benchmark for 24 h before committing.
3. **If you need max single‑GPU INT4 throughput at high concurrency:** evaluate LMDeploy TurboMind W4A16 against vLLM Machete on your traffic — TurboMind paper claims +13–31%.
4. **If you need batch‑1 minimum latency on a consumer GPU:** ExLlamaV2 + TabbyAPI with EXL2 4–5 bpw + small EXL2 draft model.
5. **If you need every last % and can afford the ops cost:** TensorRT‑LLM with FP8 + EAGLE‑3 on a tuned engine.
6. **For Kubernetes / multi‑node fleets:** llm‑d above vLLM, or NVIDIA Dynamo above vLLM/SGLang/TRT‑LLM. Only worth it once you outgrow a single node.

**Switching thresholds (when to re‑evaluate):**
- p95 TTFT > SLO at peak QPS → enable chunked prefill / lower `max_num_seqs` / try SGLang.
- Output throughput plateau at ~80% GPU memory bandwidth utilization → try FP8 KV cache or move to AWQ‑INT4.
- Acceptance length < 2 with EAGLE‑3 → workload off‑distribution; retrain the draft or disable spec decoding.
- > 70% prompts share a prefix → SGLang (or vLLM with `--enable-prefix-caching` if not already on).
- > 1 node → llm‑d / Dynamo with P/D disaggregation.

---

## Caveats

- "Fastest" depends heavily on workload (prompt/output length, concurrency, prefix overlap), GPU generation (Ampere vs Ada vs Hopper vs Blackwell), and quantization scheme. Cited numbers come from 2025 benchmarks (Spheron, Particula, AIMultiple, CloudRift, ermolushka, hardware‑corner, LMSYS, Neural Magic/Red Hat, lmdeploy docs) and may not reproduce on your stack.
- Several rapidly moving features (P‑EAGLE, EAGLE‑3.1, FlashAttention‑4, FP4 paths on Blackwell, MXFP4 kernel selection) are still maturing — re‑benchmark every 1–2 months.
- Independent benchmarks often originate from companies selling adjacent products (cloud providers, frameworks); cross‑reference with at least two sources before adopting.
- TGI numbers reflect its pre‑maintenance state; new deployments should not use it.
- Some references in this report (vLLM FP8 KV‑cache blog April 2026, EAGLE 3.1 blog May 2026, FlashAttention‑4 paper) are dated 2026 — they describe forward‑looking optimizations on Hopper/Blackwell; treat as leading indicators rather than universally available defaults.