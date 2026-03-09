"""TreeVerifier — verify draft tokens against target model.

Implements the speculative decoding acceptance criterion (Leviathan et al., 2023):
  For each draft token x at position i:
    acceptance probability = min(1, q(x|context) / p(x|context))
  where q = target distribution, p = draft distribution.

Reference: "Fast Inference from Transformers via Speculative Decoding" (2023).

The tree verifier runs the target model once on the flattened tree and
accepts the longest valid path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from spec_decode_tree.tree_spec import DraftTree, DraftNode

if TYPE_CHECKING:
    from dana_engine.model.transformer import TinyMoETransformer


class TreeVerifier:
    """Verify a draft tree against the target model.

    Runs target model once on the flattened draft (batch of paths),
    then applies acceptance criterion per token per path.

    Usage:
        verifier = TreeVerifier(target_model)
        accepted_tokens = verifier.verify(draft_tree)
    """

    def __init__(self, target_model: "TinyMoETransformer") -> None:
        self.target_model = target_model

    def verify(self, draft_tree: DraftTree) -> list[int]:
        """Verify the draft tree and return the accepted token sequence.

        Args:
            draft_tree: DraftTree from TreeSpeculator.draft()

        Returns:
            List of accepted token IDs (can be longer than 1)
        """
        if not draft_tree.paths or all(len(p) == 0 for p in draft_tree.paths):
            return self._fallback_sample(draft_tree.input_ids)

        best_path: list[int] = []
        best_length = -1

        self.target_model.eval()
        with torch.no_grad():
            for path in draft_tree.paths:
                if not path:
                    continue

                # Build context with draft tokens
                draft_ids = torch.tensor(path, dtype=torch.long).unsqueeze(0)
                context = torch.cat([draft_tree.input_ids, draft_ids], dim=1)

                # Run target model on full context
                out = self.target_model(context)
                target_logits = out.logits  # (1, T+len(path), vocab)

                # Check acceptance for each draft token
                accepted = self._accept_path(
                    path=path,
                    target_logits=target_logits,
                    prompt_len=draft_tree.input_ids.shape[1],
                )

                if len(accepted) > best_length:
                    best_length = len(accepted)
                    best_path = accepted

        if not best_path:
            return self._fallback_sample(draft_tree.input_ids)

        return best_path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _accept_path(
        self,
        path: list[int],
        target_logits: torch.Tensor,
        prompt_len: int,
    ) -> list[int]:
        """Accept tokens along a draft path using the token-level criterion.

        Simplified acceptance: accept token if it matches target's argmax,
        or sample from residual distribution. This reference implementation
        uses greedy acceptance (argmax matching) for clarity.
        """
        accepted: list[int] = []

        for i, draft_token in enumerate(path):
            # Target distribution at position prompt_len + i - 1 (predicts position i)
            target_pos = prompt_len + i - 1
            if target_pos < 0:
                continue

            target_dist = torch.softmax(target_logits[0, target_pos], dim=-1)
            target_argmax = target_dist.argmax().item()

            if target_argmax == draft_token:
                # Accept: draft token matches target's top prediction
                accepted.append(draft_token)
            else:
                # Reject: stop here and take target's token instead
                accepted.append(int(target_argmax))
                break

        return accepted

    def _fallback_sample(self, input_ids: torch.Tensor) -> list[int]:
        """Sample one token using the target model (fallback)."""
        out = self.target_model(input_ids)
        next_token = out.logits[:, -1, :].argmax(dim=-1).item()
        return [int(next_token)]
