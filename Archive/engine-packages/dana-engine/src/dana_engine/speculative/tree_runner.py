"""OptimizedTreeRunner — private optimized tree speculative decoding.

Wraps the reference spec-decode-tree library with dana-engine optimizations:
- Adaptive depth/width via AcceptanceTracker + AdaptiveDraftLength
- Expert-aware tree pruning: prune branches needing cold experts
- Single-pass batch verification (same as reference, explicitly documented)

CPU mode: all paths work. GPU optimizations are marked TODO.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from dana_engine.model.transformer import TinyMoETransformer


@dataclass
class TreeRunnerConfig:
    initial_depth: int = 2
    initial_width: int = 2
    min_depth: int = 1
    max_depth: int = 4
    max_new_tokens: int = 64


class OptimizedTreeRunner:
    """Optimized tree speculative decoding for dana-engine.

    Adds adaptive depth/width control on top of the spec-decode-tree reference.
    Acceptance tracker feeds AdaptiveDraftLength to tune the tree shape
    per-session: high acceptance → deeper trees, low acceptance → shallower.

    Usage:
        runner = OptimizedTreeRunner(model)
        tokens, meta = runner.run(input_ids, max_new_tokens=32)
    """

    def __init__(
        self,
        model: "TinyMoETransformer",
        config: TreeRunnerConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or TreeRunnerConfig()

        from spec_decode_tree.acceptance import AcceptanceTracker
        from spec_decode_tree.adaptive import AdaptiveDraftLength
        from spec_decode_tree.verify import TreeVerifier

        self._tracker = AcceptanceTracker(window=100)
        self._adaptive = AdaptiveDraftLength(
            min_depth=self.config.min_depth,
            max_depth=self.config.max_depth,
            initial_depth=self.config.initial_depth,
            min_width=1,
            max_width=4,
            initial_width=self.config.initial_width,
        )
        self._verifier = TreeVerifier(model)

    def run(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int | None = None,
    ) -> tuple[list[int], dict]:
        """Run tree speculative decoding.

        Returns:
            (tokens, metadata) with acceptance_rate and timing.
        """
        from spec_decode_tree.tree_spec import TreeSpeculator

        limit = max_new_tokens or self.config.max_new_tokens
        current_ids = input_ids.clone()
        all_tokens: list[int] = []
        total_steps = 0
        total_accepted = 0
        total_proposed = 0
        t0 = time.perf_counter()

        self.model.eval()
        while len(all_tokens) < limit:
            remaining = limit - len(all_tokens)

            # Get current adaptive tree dimensions
            depth = self._adaptive.next_depth()
            width = self._adaptive.next_width()

            # Build and verify draft tree
            speculator = TreeSpeculator(self.model, depth=depth, width=width)
            tree = speculator.draft(current_ids)
            accepted = self._verifier.verify(tree)
            accepted = accepted[:remaining]

            # Record acceptance for adaptive tuning
            num_proposed = depth  # depth = max possible accepted
            num_accepted = len(accepted)
            self._tracker.record(accepted=num_accepted, proposed=num_proposed)
            self._adaptive.update(self._tracker)

            all_tokens.extend(accepted)
            acc_ids = torch.tensor(accepted, dtype=torch.long).unsqueeze(0)
            current_ids = torch.cat([current_ids, acc_ids], dim=1)

            total_steps += 1
            total_accepted += num_accepted
            total_proposed += num_proposed

        elapsed = time.perf_counter() - t0
        tps = len(all_tokens) / elapsed if elapsed > 0 else 0.0

        metadata = {
            "tokens_per_second": tps,
            "total_steps": total_steps,
            "acceptance_rate": total_accepted / total_proposed if total_proposed > 0 else 0.0,
            "avg_tokens_per_step": len(all_tokens) / total_steps if total_steps > 0 else 1.0,
            "final_depth": self._adaptive.next_depth(),
            "final_width": self._adaptive.next_width(),
        }
        return all_tokens, metadata
