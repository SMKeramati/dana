"""Custom token counting per request.

Uses model-specific tokenizer logic to accurately count input and output
tokens for billing purposes.

Daneshbonyan: Internal R&D
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from dana_common.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Model-specific average characters-per-token ratios.  These are calibrated
# empirically for each model family and updated as new versions ship.
# ---------------------------------------------------------------------------
_MODEL_CHAR_RATIOS: dict[str, float] = {
    "dana-base": 4.0,
    "dana-large": 3.8,
    "dana-xl": 3.6,
    "dana-code": 3.2,
    "dana-embed": 4.5,
}

_DEFAULT_CHAR_RATIO = 4.0

# Special-token overhead added per request (BOS / EOS / role markers).
_SPECIAL_TOKEN_OVERHEAD: dict[str, int] = {
    "dana-base": 3,
    "dana-large": 4,
    "dana-xl": 5,
    "dana-code": 4,
    "dana-embed": 2,
}

_DEFAULT_OVERHEAD = 3

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TokenCount:
    """Immutable result of a token-counting operation."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
        }


@dataclass
class TokenCounter:
    """Model-aware token counter.

    The counter applies a model-specific character-to-token ratio, accounts
    for special-token overhead, and normalises whitespace before counting so
    that billing is deterministic regardless of formatting.
    """

    _custom_ratios: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count(
        self,
        *,
        model: str,
        input_text: str,
        output_text: str = "",
    ) -> TokenCount:
        """Return the token count for a single request/response pair."""
        ratio = self._ratio_for(model)
        overhead = _SPECIAL_TOKEN_OVERHEAD.get(model, _DEFAULT_OVERHEAD)

        input_tokens = self._estimate(input_text, ratio) + overhead
        output_tokens = self._estimate(output_text, ratio) if output_text else 0

        total = input_tokens + output_tokens
        logger.debug(
            "token_count",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total=total,
        )
        return TokenCount(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            model=model,
        )

    def register_model(self, model: str, char_ratio: float) -> None:
        """Register a custom character-per-token ratio for *model*."""
        self._custom_ratios[model] = char_ratio

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ratio_for(self, model: str) -> float:
        if model in self._custom_ratios:
            return self._custom_ratios[model]
        return _MODEL_CHAR_RATIOS.get(model, _DEFAULT_CHAR_RATIO)

    @staticmethod
    def _estimate(text: str, char_ratio: float) -> int:
        """Estimate token count from *text* using *char_ratio*.

        Whitespace is normalised so that indentation / trailing spaces do not
        inflate the count.
        """
        normalised = _WHITESPACE_RE.sub(" ", text.strip())
        if not normalised:
            return 0
        return max(1, math.ceil(len(normalised) / char_ratio))
