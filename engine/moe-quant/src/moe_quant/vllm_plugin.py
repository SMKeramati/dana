"""vLLM plug-in shell for Pillar 4 — per-expert quantization.

Importable without vLLM. Real registration happens when ``vllm`` is on
``sys.path`` (Phase 2 on rented GPU). See ``reports/vllm_hook_points.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

try:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig as _Base
    from vllm.model_executor.layers.quantization import register_quantization_config

    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover — stand-in when vllm isn't installed
    _Base = object  # type: ignore[assignment,misc]

    def register_quantization_config(name: str):  # type: ignore[no-redef]
        def deco(cls):
            return cls

        return deco

    _VLLM_AVAILABLE = False


@register_quantization_config("dana")
class DanaQuantConfig(_Base):  # type: ignore[valid-type,misc]
    """Per-expert sensitivity-aware MoE quantization.

    Each expert can carry a different bit-width: hot=Q8, warm=Q4, cold=Q2.
    Bit-width assignment is calibrated offline and stored in
    ``dana_quant_config.json`` alongside the model weights.
    """

    def get_name(self) -> str:
        return "dana"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[Any]:
        import torch

        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80  # Ampere

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["dana_quant_config.json"]

    @classmethod
    def from_config(cls, config: dict) -> DanaQuantConfig:
        return cls()

    def get_quant_method(self, layer: Any, prefix: str) -> Any:
        # TODO Phase 2: return DanaFusedMoEMethod for RoutedExperts, None otherwise
        return None
