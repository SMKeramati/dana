"""Dana Engine — custom MoE inference engine.

Public API (grows with each phase):
    Phase 1: TinyMoETransformer, TinyMoEConfig, greedy_generate
    Phase 7: DanaEngine, DanaConfig
"""

from dana_engine.model.config import TinyMoEConfig
from dana_engine.model.transformer import TinyMoETransformer, ForwardOutput
from dana_engine.naive_inference import greedy_generate, NaiveGenerationResult

__all__ = [
    "TinyMoEConfig",
    "TinyMoETransformer",
    "ForwardOutput",
    "greedy_generate",
    "NaiveGenerationResult",
]

__version__ = "0.1.0"
