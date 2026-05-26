"""vLLM plug-in shell for Pillar 6 — self-draft on MoE.

Importable without vLLM. Activated by ``vllm serve --speculative-config
'{"method": "custom_class", "model": "moe_self_draft.vllm_plugin:DanaSelfDrafter"}'``.
See ``reports/vllm_hook_points.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

try:
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer as _Base

    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _Base = object  # type: ignore[assignment,misc]
    _VLLM_AVAILABLE = False


class DanaSelfDrafter(_Base):  # type: ignore[valid-type,misc]
    """Same-model draft via top-1 expert routing.

    No separate draft model. The target's ``RoutedExperts`` layer publishes
    its top-1 routing into ``vllm.forward_context``; this proposer reads
    it back as the draft signal.

    Acceptance is measured at **expert-routing granularity**, not token
    argmax — see ``moe_self_draft.verify.SelfDraftVerifier``.
    """

    def __init__(self, vllm_config: Any, device: Any, runner: Any = None) -> None:
        if _VLLM_AVAILABLE:
            super().__init__(  # type: ignore[call-arg]
                vllm_config,
                device,
                pass_hidden_states_to_model=False,
                runner=runner,
            )
        # TODO Phase 2: install RoutedExperts.forward monkey-patch that writes
        #   router top-1 into forward_context["dana_router_top1"]

    def propose(
        self,
        target_hidden_states: Any,
        sampling_metadata: Any,
        slot_mappings: Any = None,
    ) -> Any:
        """Return ``[batch, num_speculative_tokens]`` draft expert IDs."""
        # TODO Phase 2:
        #   ctx = get_forward_context()
        #   return ctx.get("dana_router_top1")
        raise NotImplementedError("Phase 2 implementation deferred to GPU rental.")

    def load_model(self, target_model: Any) -> None:
        # No-op: target IS the draft.
        pass
