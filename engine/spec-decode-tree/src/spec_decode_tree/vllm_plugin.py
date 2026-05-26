"""vLLM plug-in shell for Pillar 5 — tree speculative decoding.

Importable without vLLM. Activated by ``vllm serve --speculative-config
'{"method": "custom_class", "model": "spec_decode_tree.vllm_plugin:DanaTreeSpeculator"}'``.
See ``reports/vllm_hook_points.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

try:
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer as _Base

    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _Base = object  # type: ignore[assignment,misc]
    _VLLM_AVAILABLE = False


class DanaTreeSpeculator(_Base):  # type: ignore[valid-type,misc]
    """Tree-shaped speculative decoder.

    Generates ``depth × width`` candidate paths, vLLM verifies them in one
    batched forward, accepts the longest verified path. Adaptive controller
    from ``spec_decode_tree.adaptive`` tunes depth/width per session.
    """

    def __init__(self, vllm_config: Any, device: Any, runner: Any = None) -> None:
        if _VLLM_AVAILABLE:
            super().__init__(  # type: ignore[call-arg]
                vllm_config,
                device,
                pass_hidden_states_to_model=True,
                runner=runner,
            )
        # TODO Phase 2: read depth/width from vllm_config.speculative_config

    def propose(
        self,
        target_hidden_states: Any,
        sampling_metadata: Any,
        slot_mappings: Any = None,
    ) -> Any:
        """Return ``[batch, num_speculative_tokens]`` int32 draft tokens."""
        # TODO Phase 2: wire spec_decode_tree.tree_spec.TreeSpeculator here
        raise NotImplementedError("Phase 2 implementation deferred to GPU rental.")

    def load_model(self, target_model: Any) -> None:
        # TODO Phase 2: load same-family small draft (e.g. Qwen3.5-0.8B → Qwen3.5-27B)
        pass
