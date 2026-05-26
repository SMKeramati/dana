"""Baseline naive autoregressive inference loop.

This is the unoptimised reference implementation — greedy decoding with
no caching, no prefetching, no speculative decoding. It establishes the
performance floor that each engine optimisation must beat.

Every phase benchmarks against this baseline to quantify improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from dana_engine.model.transformer import ForwardOutput, TinyMoETransformer


@dataclass
class NaiveGenerationResult:
    """Output of a naive greedy generation run."""

    tokens: torch.Tensor                     # (1, prompt_len + num_new_tokens)
    num_new_tokens: int
    all_router_logits: list[list[torch.Tensor]]  # [step][layer] → (1, 1, num_experts)
    steps: int                               # equals num_new_tokens (1 token/step)

    @property
    def tokens_per_step(self) -> float:
        return self.num_new_tokens / max(self.steps, 1)


def greedy_generate(
    model: TinyMoETransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
) -> NaiveGenerationResult:
    """Greedy autoregressive generation — one token per forward pass.

    Args:
        model: The TinyMoETransformer (or any model with the same interface)
        input_ids: (1, prompt_len) integer tensor
        max_new_tokens: number of tokens to generate

    Returns:
        NaiveGenerationResult with generated tokens and per-step router logits
    """
    assert input_ids.dim() == 2 and input_ids.size(0) == 1, \
        "input_ids must be (1, seq_len) — batch size 1 only for naive inference"

    model.eval()
    all_router_logits: list[list[torch.Tensor]] = []

    current_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out: ForwardOutput = model(current_ids)

            # Greedy: pick argmax of last-token logits
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
            current_ids = torch.cat([current_ids, next_token], dim=1)

            # Capture router logits at the last position only
            step_logits = [rl[:, -1:, :] for rl in out.all_router_logits]
            all_router_logits.append(step_logits)

    num_new = max_new_tokens
    return NaiveGenerationResult(
        tokens=current_ids,
        num_new_tokens=num_new,
        all_router_logits=all_router_logits,
        steps=num_new,
    )
