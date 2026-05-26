# vLLM hook-point map for the 4 Dana pillars

> **Generated:** 2026-05-26 (Phase 1 Day 4)
> **Source:** vLLM main branch as of late May 2026
> **Goal:** identify the exact subclass / registration points before writing real plugin code on a rented GPU. Three pillars are plugin-shaped (no fork). One needs a fork.

---

## Pillar 4 — per-expert quantization

| | |
|---|---|
| Shape | **library** (pip-installable plugin) |
| Subclass | `vllm.model_executor.layers.quantization.base_config::QuantizationConfig` |
| Register | `@register_quantization_config("dana")` decorator |
| Activate via | `vllm serve --quantization dana` |

**Methods to implement on `QuantizationConfig`:**

- `get_name() -> str` — return `"dana"`
- `get_supported_act_dtypes() -> list[torch.dtype]` — e.g. `[torch.bfloat16, torch.half]`
- `@classmethod get_min_capability() -> int` — minimum GPU SM, e.g. `80` for Ampere
- `@staticmethod get_config_filenames() -> list[str]` — e.g. `["dana_quant_config.json"]`
- `@classmethod from_config(config: dict) -> QuantizationConfig`
- `get_quant_method(layer: nn.Module, prefix: str) -> QuantizeMethodBase | None`
  - Return a custom `FusedMoEMethodBase` subclass for `RoutedExperts` layers
  - Return `None` for non-MoE layers (vLLM falls through to default)

**Per-expert bit-width hook (the actual point of this pillar):**

The `FusedMoEMethodBase` subclass overrides:

- `create_weights(layer, ...)` — allocate per-expert weight tensors with **per-expert bit-widths** (e.g. `[num_experts]` array of bits, stored on `layer`)
- `apply(layer, x, router_logits, ...)` — dispatch the MoE kernel using the per-expert scales

**Reference implementations to imitate:**

- `vllm/model_executor/layers/quantization/experts_int8.py` — `ExpertsInt8Config` (closest existing analog, uniform Int8)
- `vllm/model_executor/layers/quantization/gptq_marlin.py` — `GPTQMarlinConfig` (registration template)

**Gotchas:**

- Per-expert scales live as a `[num_experts, n_groups]` tensor parameter on the layer, named `expert_scales` by convention
- The `from_config()` call happens BEFORE the model is loaded, so the bit-width policy must be encoded in the model directory's config JSON, not inferred at runtime
- Calibration (which expert gets which bit-width) is a **separate offline step** — the plugin only reads the result

**Minimal stub** — see `engine/moe-quant/src/moe_quant/vllm_plugin.py`

---

## Pillar 5 — tree speculative decoding

| | |
|---|---|
| Shape | **library** |
| Subclass | `vllm.v1.spec_decode.llm_base_proposer::SpecDecodeBaseProposer` |
| Activate via | `--speculative-config '{"method": "custom_class", "model": "moe_self_draft.vllm_plugin:DanaTreeSpeculator"}'` |

**Methods to implement:**

- `__init__(vllm_config: VllmConfig, device: torch.device, runner=None)`
- `propose(target_hidden_states: Tensor, sampling_metadata: SamplingMetadata, slot_mappings=None) -> Tensor`
  - Return `[batch, num_speculative_tokens]` int32 token IDs
  - For trees: flatten paths to `depth` tokens per batch element, OR stack as `[batch, depth, width]` and reshape
- `load_model(target_model: nn.Module) -> None` — optional; set up draft model

**Dispatcher location:** `vllm/v1/worker/gpu_model_runner.py` (~line 551 in current main) has an `if/elif` chain on `speculative_config.method`. The `"custom_class"` branch imports our class via dotted-path string. No vLLM patch needed.

**Reference implementations:**

- `vllm/v1/spec_decode/medusa.py` — multi-head, no separate model
- `vllm/v1/spec_decode/eagle.py` — separate small draft model
- `vllm/v1/spec_decode/mlp_speculator.py` — simplest template

**Gotchas:**

- `num_speculative_tokens` from `vllm_config.speculative_config` is the **total** tokens proposed per step. For a tree with `depth × width`, you decide whether to expose `depth` paths to vLLM (and accept along the longest verified path) or all `depth × width` leaves
- Tree verification batched as `[num_paths, seq_len]` happens inside vLLM's target-model forward, not in your `propose()`. Your job is only to PRODUCE the tree
- `pass_hidden_states_to_model=True` is required if your tree uses hidden-state-conditioned drafting

**Minimal stub** — see `engine/spec-decode-tree/src/spec_decode_tree/vllm_plugin.py`

---

## Pillar 6 — self-draft (same model, top-1 routing as draft)

| | |
|---|---|
| Shape | **library** — reuses pillar 5's framework |
| Subclass | `SpecDecodeBaseProposer` (same as pillar 5) |
| Activate via | `--speculative-config '{"method": "custom_class", "model": "moe_self_draft.vllm_plugin:DanaSelfDrafter"}'` |

**Key difference from pillar 5:** no separate draft model. The draft signal comes from the target model's own router output (top-1 instead of top-k). This needs the target model's router outputs to be **observable** by the proposer.

**Two ways to wire this:**

1. **Forward-context channel** (clean, no fork): the target model's `RoutedExperts` layer writes router top-1 IDs into `vllm.forward_context` during its forward pass; `DanaSelfDrafter.propose()` reads them out. Requires a small monkey-patch on the `RoutedExperts.forward()` to call `set_forward_context({"router_top1": ...})` — done at plugin load time, no vLLM patch.

2. **Target-model wrapper** (slightly more invasive): subclass the target `nn.Module`, expose router outputs as an attribute. Heavier but more explicit.

We'll start with (1).

**Gotchas:**

- Self-draft tokens are **expert IDs (0..num_experts-1)**, not language tokens. The verification step must compare expert routing predictions against actual routing, not token-level argmax
- Acceptance rate is therefore measured at expert-routing granularity, not token granularity
- If the target model is sharded across TP ranks, router outputs need to be all-gathered before the proposer sees them

**Minimal stub** — see `engine/moe-self-draft/src/moe_self_draft/vllm_plugin.py`

---

## Pillar 1 — router-predict prefetch — **FORK REQUIRED**

| | |
|---|---|
| Shape | **fork or upstream PR** |
| Reason | no clean hook in stock vLLM for async expert prefetch |

**Where the patch lands:**

- `vllm/model_executor/layers/fused_moe/layer.py::FusedMoE.forward()` — add a **pre-forward callback** that takes a list of predicted expert IDs and kicks off async H2D transfer
- `vllm/distributed/expert_async_prefetch.py` — **new module** to be added, implements the async loader with CUDA stream
- `vllm/config/speculative.py::SpeculativeConfig` — add field `enable_expert_prefetch: bool = False` to gate the feature
- `vllm/model_executor/layers/fused_moe/expert_map_manager.py` — already tracks expert→device mapping; reuse its API rather than reinventing

**Less invasive alternative (testable in Phase 2 before the real fork):**

Subclass `FusedMoE`, override `forward()` to call a prefetch hook before `super().forward()`. Works as a wrapper but doesn't fix the upstream behavior — it only proves the gain exists. If the gain is real, open the upstream PR; if not, drop pillar 1 entirely.

**Decision criterion** (per Phase 2 day 5):

- If pillars 4 + 5 + 6 combined give ≥ 30% over the vLLM floor → pillar 1 is a stretch, open RFC against vLLM and don't block on it
- If < 30% → pillar 1 is the only path to the 30 tok/s thesis. Commit to the fork.

**Gotchas:**

- Prefetch must finish **before** the expert kernel launches; this requires explicit CUDA stream synchronization, not just `non_blocking=True`
- The predicted expert IDs come from a different rank (typically TP rank 0); communication cost between predict-rank and compute-rank can wipe out the prefetch gain on small models. Don't measure on Qwen3.5-0.8B
- Existing expert offload RFC #38256 (Feb 2026) is the closest mainline analog — read its PR conversation before opening yours

**Minimal stub** — see `engine/moe-router-predict/src/moe_router_predict/vllm_plugin.py` (placeholder only; real implementation is fork-conditional)

---

## Summary

| Pillar | Hook | Status |
|---|---|---|
| 4 — per-expert quant | `@register_quantization_config("dana")` | plug-in shell ready |
| 5 — tree spec decode | `SpecDecodeBaseProposer` subclass, `--speculative-config custom_class` | plug-in shell ready |
| 6 — self-draft | same as 5, plus a forward-context monkey-patch | plug-in shell ready |
| 1 — router prefetch | wrapper (Phase 2 measurement) → fork or PR (Phase 3) | **shell is a stub; decision deferred** |

All four shells are importable without vLLM installed (TYPE_CHECKING-guarded imports). Real vLLM dependency lands when Phase 2 boots on the rental.

## Verification (Phase 1 Day 4 gate)

```bash
uv run python -c "
from moe_quant.vllm_plugin import DanaQuantConfig
from spec_decode_tree.vllm_plugin import DanaTreeSpeculator
from moe_self_draft.vllm_plugin import DanaSelfDrafter
from moe_router_predict.vllm_plugin import DanaPrefetchWrapper
print('all four shells importable')
"
```

Expected: `all four shells importable`.
