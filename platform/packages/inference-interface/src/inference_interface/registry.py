"""Engine registry — maps engine names to adapter classes.

Usage in inference-gateway:
    registry = EngineRegistry()
    engine = registry.load(os.environ["ENGINE"])  # "dana" | "sglang" | "mock"
    await engine.startup()
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import InferenceEngine

# Maps ENGINE env var value → (module path, class name)
_ADAPTERS: dict[str, tuple[str, str]] = {
    "dana":    ("inference_gateway.adapters.dana_engine", "DanaEngineAdapter"),
    "sglang":  ("inference_gateway.adapters.sglang",      "SGLangAdapter"),
    "mock":    ("inference_gateway.adapters.mock",         "MockEngineAdapter"),
}


class EngineRegistry:
    def load(self, engine_name: str) -> InferenceEngine:
        key = engine_name.lower()
        if key not in _ADAPTERS:
            available = list(_ADAPTERS.keys())
            raise ValueError(
                f"Unknown engine '{engine_name}'. Available: {available}"
            )
        module_path, class_name = _ADAPTERS[key]
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls()

    def available(self) -> list[str]:
        return list(_ADAPTERS.keys())
