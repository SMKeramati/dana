"""DanaInferencePipeline — wires all 6 engine libraries into a single entry point.

Orchestration order per forward step:
  1. ExpertCache primes predictive hints from RouterPredictor
  2. MoeSelfDrafter runs in top-1 mode (draft)
  3. SelfDraftVerifier runs in top-2 mode (verify)
  4. Accepted tokens returned

CPU mode throughout (feature-flagged CUDA paths marked TODO).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import torch

from dana_engine.model.config import TinyMoEConfig
from dana_engine.model.transformer import TinyMoETransformer
from dana_engine.naive_inference import greedy_generate, NaiveGenerationResult


@dataclass
class PipelineConfig:
    """Configuration for the full inference pipeline."""
    model_config: TinyMoEConfig = field(default_factory=TinyMoEConfig.micro)
    enable_spec_decode: bool = True
    num_draft_tokens: int = 4
    spec_tree_depth: int = 2
    spec_tree_width: int = 2
    enable_prefetch: bool = True
    prefetch_steps: int = 2
    max_new_tokens: int = 64
    temperature: float = 1.0


@dataclass
class GenerationResult:
    """Output from a generation request."""
    tokens: list[int]
    tokens_generated: int
    tokens_per_second: float
    finish_reason: str  # "length" or "stop"
    spec_decode_used: bool = False
    avg_tokens_per_step: float = 1.0


class DanaInferencePipeline:
    """Full inference pipeline composing all Dana engine components.

    Initialises once; shared across requests. Thread-safe for sequential
    requests (concurrent requests require batch scheduler — Phase 7 extension).

    Usage:
        pipeline = DanaInferencePipeline(PipelineConfig())
        result = pipeline.generate(input_ids, max_new_tokens=64)
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._model = TinyMoETransformer(self.config.model_config)
        self._model.eval()

        # Lazy import engine libraries to keep startup fast
        self._drafter = None
        self._verifier = None
        self._predictor = None

        if self.config.enable_spec_decode:
            self._init_spec_decode()

        if self.config.enable_prefetch:
            self._init_prefetch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        """Generate tokens autoregressively with optional speculative decoding.

        Args:
            input_ids: (1, T) integer token tensor
            max_new_tokens: override config default

        Returns:
            GenerationResult with tokens and timing metadata
        """
        limit = max_new_tokens or self.config.max_new_tokens

        if self.config.enable_spec_decode and self._drafter is not None:
            return self._generate_speculative(input_ids, limit)
        else:
            return self._generate_naive(input_ids, limit)

    async def generate_async(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        """Async wrapper around generate() — yields control back to event loop."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.generate, input_ids, max_new_tokens
        )

    async def stream_async(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream generated tokens one-at-a-time.

        Yields individual token IDs as strings (in production this would be
        detokenized text; for the test model they are integer strings).
        """
        limit = max_new_tokens or self.config.max_new_tokens
        current_ids = input_ids.clone()
        generated = 0

        self._model.eval()
        with torch.no_grad():
            while generated < limit:
                out = self._model(current_ids)
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                token_id = int(next_token.item())
                yield str(token_id)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                generated += 1
                # Yield control back to event loop between tokens
                import asyncio
                await asyncio.sleep(0)

    def health(self) -> dict:
        """Return health status for /health endpoint."""
        return {
            "healthy": True,
            "model": self.config.model_config.__class__.__name__,
            "spec_decode": self.config.enable_spec_decode,
            "prefetch": self.config.enable_prefetch,
        }

    # ------------------------------------------------------------------
    # Internal — speculative decode path
    # ------------------------------------------------------------------

    def _generate_speculative(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> GenerationResult:
        """Speculative decode loop using MoeSelfDrafter + SelfDraftVerifier."""
        from moe_self_draft.self_draft import MoeSelfDrafter
        from moe_self_draft.verify import SelfDraftVerifier

        drafter = self._drafter
        verifier = self._verifier

        current_ids = input_ids.clone()
        all_tokens: list[int] = []
        total_steps = 0
        t0 = time.perf_counter()

        self._model.eval()
        while len(all_tokens) < max_new_tokens:
            remaining = max_new_tokens - len(all_tokens)
            num_draft = min(self.config.num_draft_tokens, remaining)

            # Optionally prefetch experts predicted by draft router logits
            draft_result = drafter.draft(current_ids, num_draft_tokens=num_draft)

            if self.config.enable_prefetch and self._predictor is not None:
                self._prefetch_from_draft(draft_result)

            # Verify: run full model, accept draft tokens
            accepted = verifier.verify(current_ids, draft_result)

            # Truncate to remaining budget
            accepted = accepted[:remaining]
            all_tokens.extend(accepted)

            # Append accepted tokens to context
            acc_ids = torch.tensor(accepted, dtype=torch.long).unsqueeze(0)
            current_ids = torch.cat([current_ids, acc_ids], dim=1)
            total_steps += 1

        elapsed = time.perf_counter() - t0
        tps = len(all_tokens) / elapsed if elapsed > 0 else 0.0
        avg_per_step = len(all_tokens) / total_steps if total_steps > 0 else 1.0

        return GenerationResult(
            tokens=all_tokens,
            tokens_generated=len(all_tokens),
            tokens_per_second=tps,
            finish_reason="length",
            spec_decode_used=True,
            avg_tokens_per_step=avg_per_step,
        )

    # ------------------------------------------------------------------
    # Internal — naive fallback path
    # ------------------------------------------------------------------

    def _generate_naive(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> GenerationResult:
        """Standard autoregressive loop without spec decode."""
        t0 = time.perf_counter()
        result: NaiveGenerationResult = greedy_generate(
            self._model,
            input_ids,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.perf_counter() - t0
        # result.tokens is the full sequence (prompt + generated); extract new tokens
        new_tokens = result.tokens[0, input_ids.shape[1]:].tolist()
        tps = len(new_tokens) / elapsed if elapsed > 0 else 0.0

        return GenerationResult(
            tokens=new_tokens,
            tokens_generated=len(new_tokens),
            tokens_per_second=tps,
            finish_reason="length",
            spec_decode_used=False,
        )

    # ------------------------------------------------------------------
    # Internal — init helpers
    # ------------------------------------------------------------------

    def _init_spec_decode(self) -> None:
        from moe_self_draft.self_draft import MoeSelfDrafter
        from moe_self_draft.verify import SelfDraftVerifier
        self._drafter = MoeSelfDrafter(self._model, num_active_override=1)
        self._verifier = SelfDraftVerifier(self._model)

    def _init_prefetch(self) -> None:
        try:
            from moe_router_predict.predictor import RouterPredictor
            self._predictor = RouterPredictor(self._model, num_steps=self.config.prefetch_steps)
        except Exception:
            self._predictor = None  # graceful degradation

    def _prefetch_from_draft(self, draft_result: object) -> None:
        """Use predicted experts from draft router logits as prefetch hints.

        In CPU mode this is a no-op beyond bookkeeping; in GPU mode these
        hints would fire async H2D transfers via AsyncExpertLoader.
        """
        # TODO(gpu): wire to AsyncExpertLoader.enqueue() with priority scheduling
        pass
