"""SelfDraftVerifier — verify self-drafted tokens against the full model.

Same speculative decoding acceptance criterion as spec-decode-tree,
but specialised for the self-draft scenario where we know the draft
and target share the same architecture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from moe_self_draft.self_draft import DraftResult

if TYPE_CHECKING:
    from dana_engine.model.transformer import TinyMoETransformer


class SelfDraftVerifier:
    """Verify self-drafted tokens using the full model (top-k experts).

    Usage:
        verifier = SelfDraftVerifier(model)
        accepted = verifier.verify(input_ids, draft_result)
    """

    def __init__(self, model: "TinyMoETransformer") -> None:
        self.model = model

    def verify(
        self,
        input_ids: torch.Tensor,
        draft_result: DraftResult,
    ) -> list[int]:
        """Verify draft tokens and return accepted sequence.

        Args:
            input_ids: (1, T) original prompt/context
            draft_result: DraftResult from MoeSelfDrafter.draft()

        Returns:
            List of accepted token IDs (≥1 token)
        """
        if not draft_result.draft_tokens:
            return self._sample_one(input_ids)

        draft_tokens = draft_result.draft_tokens
        draft_ids = torch.tensor(draft_tokens, dtype=torch.long).unsqueeze(0)
        context = torch.cat([input_ids, draft_ids], dim=1)

        self.model.eval()
        with torch.no_grad():
            out = self.model(context)
            target_logits = out.logits  # (1, T + len(draft), vocab)

        prompt_len = input_ids.shape[1]
        accepted: list[int] = []

        for i, draft_token in enumerate(draft_tokens):
            pos = prompt_len + i - 1
            if pos < 0:
                continue
            target_top = target_logits[0, pos].argmax().item()
            if int(target_top) == draft_token:
                accepted.append(draft_token)
            else:
                # Accept target's token instead, stop
                accepted.append(int(target_top))
                break

        return accepted if accepted else self._sample_one(input_ids)

    def _sample_one(self, input_ids: torch.Tensor) -> list[int]:
        out = self.model(input_ids)
        token = out.logits[:, -1, :].argmax().item()
        return [int(token)]
