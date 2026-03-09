"""Spec Decode Tree — tree-based speculative decoding (reference implementation)."""

from spec_decode_tree.tree_spec import TreeSpeculator, DraftTree
from spec_decode_tree.verify import TreeVerifier
from spec_decode_tree.adaptive import AdaptiveDraftLength
from spec_decode_tree.acceptance import AcceptanceTracker

__all__ = [
    "TreeSpeculator",
    "DraftTree",
    "TreeVerifier",
    "AdaptiveDraftLength",
    "AcceptanceTracker",
]

__version__ = "0.1.0"
