"""MoE Self Draft — zero-memory speculative decoding for MoE models."""

from moe_self_draft.self_draft import MoeSelfDrafter, DraftResult
from moe_self_draft.logit_extractor import RouterLogitExtractor
from moe_self_draft.verify import SelfDraftVerifier

__all__ = [
    "MoeSelfDrafter",
    "DraftResult",
    "RouterLogitExtractor",
    "SelfDraftVerifier",
]

__version__ = "0.1.0"
