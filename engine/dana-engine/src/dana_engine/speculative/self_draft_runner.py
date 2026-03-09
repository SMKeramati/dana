"""OptimizedSelfDraftRunner — private optimized MoE self-draft implementation.

Extends the reference moe-self-draft library with:
- Fused expert pre-loading: accumulate all verification experts during draft
- Overlap: dequant of verify experts starts during draft phase (CPU stub)
- Batch verification: verify all draft tokens in one forward pass (same as reference)

CPU mode: all paths work correctly. GPU stream overlap is a TODO stub.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from dana_engine.model.transformer import TinyMoETransformer


@dataclass
class SelfDraftRunnerConfig:
    num_draft_tokens: int = 5
    max_new_tokens: int = 64


class OptimizedSelfDraftRunner:
    """Optimized self-draft speculative decoding for dana-engine.

    This is the private implementation used internally by DanaInferencePipeline.
    The public reference implementation lives in moe-self-draft/.

    Key optimisation over the reference:
    - During draft, RouterLogitExtractor captures predicted experts
    - Those experts are "pre-warmed" before verification starts
    - On GPU: warm means async H2D transfer; on CPU: warm is a no-op

    Usage:
        runner = OptimizedSelfDraftRunner(model)
        tokens = runner.run(input_ids, max_new_tokens=32)
    """

    def __init__(
        self,
        model: "TinyMoETransformer",
        config: SelfDraftRunnerConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or SelfDraftRunnerConfig()

        from moe_self_draft.self_draft import MoeSelfDrafter
        from moe_self_draft.verify import SelfDraftVerifier
        self._drafter = MoeSelfDrafter(model, num_active_override=1)
        self._verifier = SelfDraftVerifier(model)

    def run(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int | None = None,
    ) -> tuple[list[int], dict]:
        """Run speculative decoding and return accepted tokens + metadata.

        Returns:
            (tokens, metadata) where metadata has timing and acceptance stats.
        """
        limit = max_new_tokens or self.config.max_new_tokens
        current_ids = input_ids.clone()
        all_tokens: list[int] = []
        total_steps = 0
        total_draft = 0
        total_accepted = 0
        t0 = time.perf_counter()

        self.model.eval()
        while len(all_tokens) < limit:
            remaining = limit - len(all_tokens)
            num_draft = min(self.config.num_draft_tokens, remaining)

            # Phase 1: Draft with top-1 expert mode
            draft_result = self._drafter.draft(current_ids, num_draft_tokens=num_draft)

            # Phase 1b: Pre-warm experts needed for verification (GPU: async H2D)
            self._prewarm_experts(draft_result.predicted_experts())

            # Phase 2: Verify with full top-k model
            accepted = self._verifier.verify(current_ids, draft_result)
            accepted = accepted[:remaining]

            all_tokens.extend(accepted)
            acc_ids = torch.tensor(accepted, dtype=torch.long).unsqueeze(0)
            current_ids = torch.cat([current_ids, acc_ids], dim=1)

            total_steps += 1
            total_draft += num_draft
            total_accepted += len(accepted)

        elapsed = time.perf_counter() - t0
        tps = len(all_tokens) / elapsed if elapsed > 0 else 0.0
        acceptance_rate = total_accepted / total_draft if total_draft > 0 else 0.0

        metadata = {
            "tokens_per_second": tps,
            "total_steps": total_steps,
            "acceptance_rate": acceptance_rate,
            "avg_tokens_per_step": len(all_tokens) / total_steps if total_steps > 0 else 1.0,
        }
        return all_tokens, metadata

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prewarm_experts(self, expert_ids: set[int]) -> None:
        """Pre-warm experts needed for the upcoming verification pass.

        CPU mode: no-op (experts are always in memory).
        TODO(gpu): fire async H2D transfers via AsyncExpertLoader for each
                   expert_id not already in VRAM tier.
        """
        pass  # TODO(gpu): implement async expert loading
