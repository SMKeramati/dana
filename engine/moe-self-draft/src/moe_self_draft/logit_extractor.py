"""RouterLogitExtractor — capture router logits via forward hooks.

Registers forward hooks on every MoERouter in the model. After each forward
pass, get_logits() returns the captured router distributions.

These logits tell us exactly which experts the next verification pass will
need → free prefetch window with zero extra compute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from dana_engine.model.transformer import TinyMoETransformer


class RouterLogitExtractor:
    """Extract router logits from every MoE layer via forward hooks.

    Usage:
        extractor = RouterLogitExtractor()
        extractor.attach(model)
        model(input_ids)
        logits = extractor.get_logits()  # list of (B, T, num_experts) per layer
        extractor.clear()                # clear before next forward
    """

    def __init__(self) -> None:
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._captured: list[torch.Tensor] = []
        self._attached = False

    def attach(self, model: "TinyMoETransformer") -> None:
        """Register hooks on all MoERouter modules in the model."""
        if self._attached:
            self.detach()

        from dana_engine.model.moe_layer import MoERouter

        for module in model.modules():
            if isinstance(module, MoERouter):
                hook = module.register_forward_hook(self._capture_hook)
                self._hooks.append(hook)

        self._attached = True

    def detach(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._attached = False

    def clear(self) -> None:
        """Clear captured logits (call before each forward pass)."""
        self._captured.clear()

    def get_logits(self) -> list[torch.Tensor]:
        """Return captured router logits.

        Returns:
            List of (batch, seq, num_experts) tensors, one per MoE layer.
        """
        return list(self._captured)

    def get_top_experts(self, k: int = 1) -> list[list[int]]:
        """Return top-k expert IDs per layer from the last captured forward pass.

        Returns:
            List of expert ID lists, one per layer.
        """
        result: list[list[int]] = []
        for logits in self._captured:
            # logits: (B, T, num_experts) — take last token
            last = logits[:, -1, :]  # (B, num_experts)
            _, top_ids = torch.topk(last[0], k)
            result.append(top_ids.tolist())
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_hook(
        self,
        module: nn.Module,
        inputs: tuple,
        output: object,
    ) -> None:
        """Forward hook: captures router logits from RouterOutput."""
        # RouterOutput is a NamedTuple with .logits field
        if hasattr(output, "logits"):
            self._captured.append(output.logits.detach())
