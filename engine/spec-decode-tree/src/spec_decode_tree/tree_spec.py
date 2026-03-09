"""TreeSpeculator — draft multiple candidate token sequences as a tree.

Reference implementation (unoptimized). The production version in dana-engine
has CUDA-optimized tree construction and flattening.

Tree structure:
    root: current token
    depth 1: top-width continuations
    depth 2: top-width continuations of each depth-1 node
    ...
    depth D: D levels of branching

The tree is flattened to a batch for single-pass verification by TreeVerifier.

Example tree (depth=2, width=2):
    position 0: [root]
    position 1,2: continuations of root
    position 3,4: continuations of pos 1
    position 5,6: continuations of pos 2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from dana_engine.model.transformer import TinyMoETransformer


@dataclass
class DraftNode:
    token_id: int
    parent_idx: int          # index of parent in flat list (-1 for root)
    depth: int
    logprob: float           # log probability under draft model


@dataclass
class DraftTree:
    """A flat representation of the speculation tree."""
    nodes: list[DraftNode]
    paths: list[list[int]]   # each path is a list of token_ids from root to leaf
    input_ids: torch.Tensor  # original prompt (1, T)

    def num_candidates(self) -> int:
        return len(self.paths)

    def max_depth(self) -> int:
        return max((len(p) for p in self.paths), default=0)


class TreeSpeculator:
    """Generate draft trees using top-k sampling from a draft model.

    Uses the same model as the target (the "self-draft" idea) or a
    separate draft model. In both cases, the interface is the same.

    Usage:
        spec = TreeSpeculator(draft_model, depth=3, width=2)
        tree = spec.draft(input_ids)
        # → DraftTree with width^depth leaf paths
    """

    def __init__(
        self,
        model: "TinyMoETransformer",
        depth: int = 3,
        width: int = 2,
    ) -> None:
        self.model = model
        self.depth = depth
        self.width = width

    def draft(self, input_ids: torch.Tensor) -> DraftTree:
        """Generate a draft tree by running the model greedily with branching.

        Args:
            input_ids: (1, T) current token sequence

        Returns:
            DraftTree with up to width^depth leaf paths
        """
        self.model.eval()
        nodes: list[DraftNode] = []
        # Queue entries: (current_ids, parent_node_idx, depth)
        queue: list[tuple[torch.Tensor, int, int]] = [(input_ids, -1, 0)]
        paths: list[list[int]] = []

        with torch.no_grad():
            while queue:
                current_ids, parent_idx, current_depth = queue.pop(0)

                if current_depth >= self.depth:
                    # This is a leaf: collect path
                    path = self._extract_path(nodes, parent_idx)
                    paths.append(path)
                    continue

                # Run model to get next token distribution
                out = self.model(current_ids)
                logits = out.logits[:, -1, :]         # (1, vocab)
                log_probs = torch.log_softmax(logits, dim=-1)

                # Take top-width tokens
                top_logprobs, top_tokens = torch.topk(log_probs[0], self.width)

                for i in range(self.width):
                    token_id = top_tokens[i].item()
                    logprob = top_logprobs[i].item()

                    node = DraftNode(
                        token_id=int(token_id),
                        parent_idx=parent_idx,
                        depth=current_depth + 1,
                        logprob=logprob,
                    )
                    node_idx = len(nodes)
                    nodes.append(node)

                    next_ids = torch.cat([current_ids, top_tokens[i:i+1].unsqueeze(0)], dim=1)
                    queue.append((next_ids, node_idx, current_depth + 1))

        # If no paths were collected (depth=0), create a dummy path
        if not paths:
            paths = [[]]

        return DraftTree(nodes=nodes, paths=paths, input_ids=input_ids)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_path(self, nodes: list[DraftNode], leaf_idx: int) -> list[int]:
        """Walk from leaf back to root, returning token IDs in order."""
        if leaf_idx == -1:
            return []
        path = []
        idx = leaf_idx
        while idx != -1:
            path.append(nodes[idx].token_id)
            idx = nodes[idx].parent_idx
        return list(reversed(path))
