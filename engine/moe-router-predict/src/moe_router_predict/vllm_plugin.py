"""vLLM plug-in shell for Pillar 1 — router-predict prefetch.

**THIS PILLAR REQUIRES A vLLM FORK** (no clean hook in stock vLLM as of
late May 2026). The shell below is a *wrapper* that subclasses
``FusedMoE`` to prove the prefetch gain in Phase 2 before committing to
the fork. If the gain is real, file the upstream PR. If not, drop the
pillar entirely. See ``reports/vllm_hook_points.md`` for the decision
criterion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from vllm.model_executor.layers.fused_moe import FusedMoE

try:
    from vllm.model_executor.layers.fused_moe import FusedMoE as _Base

    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _Base = object  # type: ignore[assignment,misc]
    _VLLM_AVAILABLE = False


class DanaPrefetchWrapper(_Base):  # type: ignore[valid-type,misc]
    """Wrap FusedMoE to issue async H2D for predicted expert weights.

    Predicted IDs come from a draft-model lookahead (the existing
    ``moe_router_predict.predictor.RouterPredictor``). This wrapper is
    *measurement-only* — production behaviour requires patching the
    expert-offload module upstream (see hook_points doc).
    """

    def forward(  # type: ignore[override]
        self,
        hidden_states: Any,
        router_logits: Any = None,
        predicted_expert_ids: Any = None,
    ) -> Any:
        # TODO Phase 2: kick async H2D for predicted_expert_ids before super().forward()
        #   stream = torch.cuda.Stream()
        #   with torch.cuda.stream(stream):
        #       for eid in predicted_expert_ids:
        #           self._stage_expert_to_gpu(eid)
        if _VLLM_AVAILABLE:
            return super().forward(hidden_states, router_logits)  # type: ignore[misc]
        raise NotImplementedError("Phase 2 implementation deferred to GPU rental.")

    def _stage_expert_to_gpu(self, expert_id: int) -> None:
        # TODO Phase 2: read offset from FusedMoE.expert_map_manager, async copy
        raise NotImplementedError
