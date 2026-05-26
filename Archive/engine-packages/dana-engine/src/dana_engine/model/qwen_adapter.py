"""QwenMoEAdapter — bridge HuggingFace Qwen3.5-MoE to Dana engine interfaces.

Maps:
  model.model.layers[i].mlp.experts[j].{gate,up,down}_proj.weight
      → TieredTensorStore (RAM tier on construction, VRAM on demand)
  model.model.layers[i].mlp.gate.weight
      → RouterPredictor-compatible interface

Architecture notes (Qwen3.5-35B-A3B):
  - 256 experts per MoE layer (vs 60 in Qwen1.5-MoE)
  - Hybrid: Gated Delta Network layers interleaved with standard attention
  - Expert FFN: gate_proj + up_proj + down_proj (SwiGLU activation)
  - Routing: top-K sparse, K varies by layer config

Usage (Colab Section 6a):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from tiered_tensor_store import TieredTensorStore
    from dana_engine.model.qwen_adapter import QwenMoEAdapter

    store   = TieredTensorStore()
    adapter = QwenMoEAdapter(model_hf, store=store)

    # generate (model runs without VRAM-resident expert weights;
    # Dana prefetch pipeline promotes predicted experts ahead of each step)
    out = adapter.generate(input_ids, max_new_tokens=64)
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

import torch

logger = logging.getLogger(__name__)

# Projection names present in each Qwen3.5 expert FFN (SwiGLU)
_EXPERT_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")


class QwenMoEAdapter:
    """Wraps a HuggingFace Qwen3.5-MoE model with Dana engine interfaces.

    On construction every expert FFN weight is offloaded from VRAM to the
    RAM tier of ``TieredTensorStore``, freeing ~90% of expert VRAM.  Dana's
    prefetch pipeline then promotes predicted-hot experts back to VRAM ahead
    of each forward pass via ``AsyncExpertLoader.create_with_cuda_stream()``.

    The adapter is intentionally model-agnostic: it probes the HF model's
    layer structure at runtime so it works with any Qwen3.5-class checkpoint
    (35B-A3B, 7B-A1.4B, etc.) without hard-coded layer counts.
    """

    def __init__(
        self,
        hf_model: Any,
        store: Optional[Any] = None,
        offload_experts: bool = True,
    ) -> None:
        """
        Args:
            hf_model: HuggingFace ``AutoModelForCausalLM`` instance (Qwen3.5 MoE).
            store: ``TieredTensorStore`` for tiered weight management.
                   If None, offload_experts is ignored and weights stay in VRAM.
            offload_experts: Move expert FFN weights to RAM tier on init.
                             Set False to skip offload (e.g. for profiling).
        """
        self.model = hf_model
        self.store = store
        self._expert_keys: list[str] = []
        self._num_moe_layers: int = 0

        if offload_experts and store is not None:
            self._offload_experts()

        self._num_moe_layers = self._count_moe_layers()
        logger.info(
            "QwenMoEAdapter: %d MoE layers, %d expert projections offloaded to RAM",
            self._num_moe_layers,
            len(self._expert_keys),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens using the underlying HF model.

        Expert weights must be resident in VRAM before calling.  When used
        through ``DanaInferencePipeline``, the prefetch loop handles this
        automatically via ``AsyncExpertLoader``.

        Args:
            input_ids: (1, T) token tensor on the model's device (CUDA).
            max_new_tokens: number of tokens to generate.
            **kwargs: forwarded verbatim to ``model.generate()``.

        Returns:
            (1, T + max_new_tokens) output token tensor.
        """
        self.model.eval()
        with torch.no_grad():
            return self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                **kwargs,
            )

    def expert_keys(self) -> list[str]:
        """Return all expert storage keys registered in TieredTensorStore."""
        return list(self._expert_keys)

    def num_moe_layers(self) -> int:
        """Return the number of MoE layers in the model."""
        return self._num_moe_layers

    def router_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield ``(key, weight_tensor)`` for every router gate matrix.

        Router weights are small (hidden_dim × num_experts) and always kept
        in VRAM — they are the "cheap" part that RouterPredictor uses to
        predict which experts the *next* token will activate.
        """
        for layer_idx, layer in enumerate(self.model.model.layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            gate = getattr(mlp, "gate", None)
            if gate is not None and hasattr(gate, "weight") and gate.weight is not None:
                yield f"L{layer_idx}_router_gate", gate.weight

    def restore_expert(self, key: str) -> None:
        """Promote a single expert projection from RAM tier back to VRAM.

        Called by Dana's prefetch loop immediately before the expert is needed.
        ``key`` must match one of the keys returned by ``expert_keys()``.
        """
        if self.store is None:
            return
        # layer_idx, exp_idx, proj_name encoded in key: "L{i}_E{j}_{proj}"
        parts = key.split("_", 2)
        if len(parts) != 3:
            logger.warning("restore_expert: unexpected key format %r", key)
            return
        _, exp_part, proj_name = parts
        layer_idx = int(parts[0][1:])
        exp_idx = int(exp_part[1:])

        try:
            weight: torch.Tensor = self.store.retrieve(key)
        except Exception as exc:
            logger.warning("restore_expert: retrieve %r failed: %s", key, exc)
            return

        try:
            layer = self.model.model.layers[layer_idx]
            expert = layer.mlp.experts[exp_idx]
            proj = getattr(expert, proj_name)
            device = next(self.model.parameters()).device
            proj.weight = torch.nn.Parameter(weight.to(device), requires_grad=False)
        except (IndexError, AttributeError) as exc:
            logger.warning("restore_expert: failed to set weight for %r: %s", key, exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _offload_experts(self) -> None:
        """Move all expert FFN weights from VRAM to the RAM tier.

        After offload the module's ``weight`` attribute is set to ``None``
        (freed from VRAM).  Use ``restore_expert(key)`` to promote back.

        Key format: ``"L{layer_idx}_E{exp_idx}_{proj_name}"``
          e.g. ``"L3_E127_gate_proj"``
        """
        offloaded = 0
        for layer_idx, layer in enumerate(self.model.model.layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None or not hasattr(mlp, "experts"):
                continue

            for exp_idx, expert in enumerate(mlp.experts):
                for proj_name in _EXPERT_PROJ_NAMES:
                    proj = getattr(expert, proj_name, None)
                    if proj is None or not hasattr(proj, "weight") or proj.weight is None:
                        continue

                    key = f"L{layer_idx}_E{exp_idx}_{proj_name}"
                    weight_cpu = proj.weight.detach().cpu()
                    self.store.store(key, weight_cpu, tier="ram")
                    proj.weight = None   # release VRAM immediately
                    self._expert_keys.append(key)
                    offloaded += 1

        logger.debug("Offloaded %d expert projections to RAM tier", offloaded)

    def _count_moe_layers(self) -> int:
        count = 0
        for layer in self.model.model.layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "experts"):
                count += 1
        return count
