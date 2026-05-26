"""Custom speculative decoding engine.

Daneshbonyan: Internal Design & Development

Draft-verify pipeline: a small draft model predicts N tokens ahead, then the
large target model verifies all candidates in a single parallel forward pass.
Accepted tokens are emitted immediately; on rejection the target model's
distribution is used to resample from the residual.  Configurable lookahead
depth (gamma) and acceptance threshold.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    gamma: int = 5  # lookahead depth (draft tokens per step)
    temperature: float = 1.0
    top_k: int = 50
    max_rejection_streak: int = 3
    adaptive_gamma: bool = True
    min_gamma: int = 1
    max_gamma: int = 12


@dataclass
class DecodingStats:
    """Runtime statistics for speculative decoding."""

    total_draft_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    total_steps: int = 0
    gamma_history: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_draft_tokens

    @property
    def tokens_per_step(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return (self.accepted_tokens + self.total_steps) / self.total_steps


# Type alias: a model forward function takes token ids (1-D int array) and
# returns logits of shape (seq_len, vocab_size).
ModelForwardFn = Callable[[np.ndarray], np.ndarray]


def _sample_from_logits(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 50,
    rng: np.random.Generator | None = None,
) -> tuple[int, np.ndarray]:
    """Sample a single token from logits; return (token_id, probabilities)."""
    if rng is None:
        rng = np.random.default_rng()

    logits = logits.astype(np.float64)

    if temperature > 0:
        logits = logits / temperature
    else:
        # Greedy
        token = int(np.argmax(logits))
        probs = np.zeros_like(logits)
        probs[token] = 1.0
        return token, probs

    # Top-k filtering
    if top_k > 0 and top_k < len(logits):
        indices_to_remove = np.argsort(logits)[: -top_k]
        logits[indices_to_remove] = -np.inf

    # Stable softmax
    logits -= np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / (exp_logits.sum() + 1e-12)

    token = int(rng.choice(len(probs), p=probs))
    return token, probs


def _compute_residual_distribution(
    target_probs: np.ndarray,
    draft_probs: np.ndarray,
) -> np.ndarray:
    """Compute the residual distribution max(0, p_target - p_draft) normalised."""
    residual = np.maximum(target_probs - draft_probs, 0.0)
    total = residual.sum()
    if total < 1e-12:
        # Fall back to target distribution
        return target_probs / (target_probs.sum() + 1e-12)
    return residual / total


class SpeculativeDecoder:
    """Speculative decoding engine with adaptive lookahead.

    Daneshbonyan: Internal Design & Development
    """

    def __init__(
        self,
        draft_forward: ModelForwardFn,
        target_forward: ModelForwardFn,
        vocab_size: int,
        config: SpeculativeConfig | None = None,
        seed: int = 0,
    ) -> None:
        self._draft_forward = draft_forward
        self._target_forward = target_forward
        self._vocab_size = vocab_size
        self.config = config or SpeculativeConfig()
        self._rng = np.random.default_rng(seed)
        self.stats = DecodingStats()
        self._current_gamma = self.config.gamma
        self._rejection_streak = 0

    # ------------------------------------------------------------------
    # Adaptive gamma
    # ------------------------------------------------------------------
    def _adapt_gamma(self, accepted: int, proposed: int) -> None:
        """Adjust lookahead depth based on recent acceptance rate."""
        if not self.config.adaptive_gamma:
            return

        rate = accepted / max(proposed, 1)
        if rate > 0.8:
            self._current_gamma = min(
                self._current_gamma + 1, self.config.max_gamma
            )
            self._rejection_streak = 0
        elif rate < 0.3:
            self._current_gamma = max(
                self._current_gamma - 1, self.config.min_gamma
            )
            self._rejection_streak += 1
        else:
            self._rejection_streak = 0

        self.stats.gamma_history.append(self._current_gamma)

    # ------------------------------------------------------------------
    # Core speculative step
    # ------------------------------------------------------------------
    def speculative_step(
        self,
        prefix_ids: np.ndarray,
    ) -> tuple[list[int], int]:
        """Run one speculative decoding step.

        Parameters
        ----------
        prefix_ids : 1-D int array
            Token ids generated so far (the prompt / context).

        Returns
        -------
        new_tokens : list[int]
            Newly accepted token ids (1 .. gamma+1 tokens).
        num_accepted_draft : int
            How many of the draft tokens were accepted.
        """
        gamma = self._current_gamma
        draft_tokens: list[int] = []
        draft_probs_list: list[np.ndarray] = []

        # --- Draft phase: generate gamma candidate tokens autoregressively ---
        current = prefix_ids.copy()
        for _ in range(gamma):
            logits = self._draft_forward(current)
            # Take logits for the last position
            last_logits = logits[-1] if logits.ndim == 2 else logits
            token, probs = _sample_from_logits(
                last_logits,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                rng=self._rng,
            )
            draft_tokens.append(token)
            draft_probs_list.append(probs)
            current = np.append(current, token)

        # --- Verify phase: run target model on prefix + all draft tokens ---
        verify_input = np.concatenate([prefix_ids, np.array(draft_tokens, dtype=prefix_ids.dtype)])
        target_logits = self._target_forward(verify_input)
        # target_logits shape: (len(verify_input), vocab_size)
        # We need logits at positions len(prefix_ids)-1 .. len(prefix_ids)+gamma-1
        # which predict tokens at positions len(prefix_ids) .. len(prefix_ids)+gamma

        accepted_tokens: list[int] = []
        num_accepted = 0

        for i in range(gamma):
            target_pos = len(prefix_ids) - 1 + i
            if target_logits.ndim == 2:
                t_logits = target_logits[target_pos].copy()
            else:
                t_logits = target_logits.copy()

            # Compute target probs
            t_logits_f = t_logits.astype(np.float64)
            if self.config.temperature > 0:
                t_logits_f /= self.config.temperature
            t_logits_f -= np.max(t_logits_f)
            exp_t = np.exp(t_logits_f)
            target_probs = exp_t / (exp_t.sum() + 1e-12)

            draft_probs = draft_probs_list[i]
            draft_token = draft_tokens[i]

            # Acceptance criterion: accept with probability
            # min(1, p_target(x) / p_draft(x))
            p_target = target_probs[draft_token]
            p_draft = draft_probs[draft_token]

            if p_draft < 1e-12:
                # Draft assigned ~zero probability, reject
                acceptance_prob = 0.0
            else:
                acceptance_prob = min(1.0, p_target / p_draft)

            r = float(self._rng.random())
            if r < acceptance_prob:
                accepted_tokens.append(draft_token)
                num_accepted += 1
            else:
                # Rejection: sample from residual distribution
                residual = _compute_residual_distribution(target_probs, draft_probs)
                resampled = int(self._rng.choice(len(residual), p=residual))
                accepted_tokens.append(resampled)
                break
        else:
            # All gamma tokens accepted -- bonus: sample one more from target
            bonus_pos = len(prefix_ids) - 1 + gamma
            if target_logits.ndim == 2 and bonus_pos < target_logits.shape[0]:
                bonus_logits = target_logits[bonus_pos]
                bonus_token, _ = _sample_from_logits(
                    bonus_logits.copy(),
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    rng=self._rng,
                )
                accepted_tokens.append(bonus_token)

        # Update stats
        self.stats.total_draft_tokens += gamma
        self.stats.accepted_tokens += num_accepted
        self.stats.rejected_tokens += gamma - num_accepted
        self.stats.total_steps += 1

        self._adapt_gamma(num_accepted, gamma)

        return accepted_tokens, num_accepted

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 128,
        eos_token_id: int = 2,
    ) -> list[int]:
        """Generate tokens using speculative decoding.

        Parameters
        ----------
        prompt_ids : 1-D int array
        max_new_tokens : int
        eos_token_id : int

        Returns
        -------
        list of generated token ids (excluding the prompt).
        """
        generated: list[int] = []
        current = prompt_ids.copy()

        while len(generated) < max_new_tokens:
            new_tokens, _ = self.speculative_step(current)
            for tok in new_tokens:
                generated.append(tok)
                current = np.append(current, tok)
                if tok == eos_token_id:
                    return generated
                if len(generated) >= max_new_tokens:
                    break

        return generated

    def reset_stats(self) -> None:
        self.stats = DecodingStats()
        self._current_gamma = self.config.gamma
        self._rejection_streak = 0
